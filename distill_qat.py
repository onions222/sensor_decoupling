import os
import argparse
import json
import copy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.ao.quantization as tq
from models import make_student_model
import json as _json

# ---------------------------------------------------------
# [配置] 强制使用 'qnnpack' 后端 (为 MCU 准备对称量化参数)
# ---------------------------------------------------------
torch.backends.quantized.engine = 'qnnpack'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -----------------------------
# 模型与数据定义 (保持不变)
# -----------------------------

class _PatchNet_V13_Large_Teacher(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, 1)
        )
    def forward(self, x): return self.net(x)

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.odd_net = _PatchNet_V13_Large_Teacher()
        self.even_net = _PatchNet_V13_Large_Teacher()
    def forward(self, x_patches, is_odd_flags):
        out_odd = self.odd_net(x_patches)
        out_even = self.even_net(x_patches)
        is_odd_mask = is_odd_flags.view(-1, 1, 1, 1) > 0.5
        out = torch.where(is_odd_mask, out_odd, out_even)
        return torch.relu(out)


class _PatchNet_V14_Student(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 4, 1, bias=False), nn.BatchNorm2d(4), nn.ReLU(inplace=True),
            nn.Conv2d(4, out_channels, 1)
        )
    def forward(self, x): return self.net(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.odd_net = _PatchNet_V14_Student()
        self.even_net = _PatchNet_V14_Student()
    def forward(self, x_patches, is_odd_flags):
        out_odd = self.odd_net(x_patches)
        out_even = self.even_net(x_patches)
        is_odd_mask = is_odd_flags.view(-1, 1, 1, 1) > 0.5
        out = torch.where(is_odd_mask, out_odd, out_even)
        return torch.relu(out)

# ---------------------------------
# CoM modules
# ---------------------------------
class DifferentiableCoM_Patch_3x5(nn.Module):
    def __init__(self, height=3, width=5):
        super(DifferentiableCoM_Patch_3x5, self).__init__()
        self.height, self.width = height, width
        y, x = torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)
        x_grid, y_grid = x.view(1, -1).repeat(height, 1), y.view(-1, 1).repeat(1, width)
        self.register_buffer('x_grid_buf', x_grid)
        self.register_buffer('y_grid_buf', y_grid)
    def forward(self, image):
        img = image.squeeze(1); eps = 1e-8
        mass = torch.sum(img, dim=(1, 2)) + eps
        x_weighted = torch.sum(img * self.x_grid_buf, dim=(1, 2))
        y_weighted = torch.sum(img * self.y_grid_buf, dim=(1, 2))
        center_x, center_y = x_weighted / mass, y_weighted / mass
        return torch.stack([center_x, center_y], dim=1)

class CoM_from_Patch_V12(nn.Module):
    def __init__(self, patch_h=3, patch_w=5):
        super(CoM_from_Patch_V12, self).__init__()
        self.local_com_calc = DifferentiableCoM_Patch_3x5(patch_h, patch_w)
        self.ph_offset = patch_h // 2
        self.pw_offset_10col = patch_w // 2
        odd_grid_10_np = np.array([0.5, 2.5, 4.5, 6.5, 8.0, 9.0, 10.5, 12.5, 14.5, 16.5], dtype=np.float32)
        even_grid_10_np = np.array([0.0, 1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.0], dtype=np.float32)
        self.register_buffer('odd_grid_10', torch.from_numpy(odd_grid_10_np))
        self.register_buffer('even_grid_10', torch.from_numpy(even_grid_10_np))
    def forward(self, patches, is_odd_flags, peak_rs, peak_cs_10_col):
        local_coords_3x5 = self.local_com_calc(patches)
        local_x_10col_scalar = local_coords_3x5[:, 0]
        local_y_scalar = local_coords_3x5[:, 1]
        global_x_10col_scalar = local_x_10col_scalar + peak_cs_10_col - self.pw_offset_10col
        global_y_scalar = local_y_scalar + peak_rs - self.ph_offset
        x_clamped = torch.clamp(global_x_10col_scalar, 0, 9)
        x_floor, x_ceil = torch.floor(x_clamped).long(), torch.ceil(x_clamped).long()
        x_floor, x_ceil = torch.clamp(x_floor, 0, 9), torch.clamp(x_ceil, 0, 9)
        frac = x_clamped - x_floor.float()
        grids = torch.stack([self.even_grid_10, self.odd_grid_10], dim=0)
        selected_grids = grids[is_odd_flags.long()]
        val_floor = torch.gather(selected_grids, 1, x_floor.unsqueeze(-1)).squeeze(-1)
        val_ceil = torch.gather(selected_grids, 1, x_ceil.unsqueeze(-1)).squeeze(-1)
        global_x_18col_scalar = val_floor + (val_ceil - val_floor) * frac
        return torch.stack([global_x_18col_scalar, global_y_scalar], dim=1)

class CoM_from_18col_Patch_V15(nn.Module):
    def __init__(self, patch_h=3, patch_w=5):
        super(CoM_from_18col_Patch_V15, self).__init__()
        self.local_com_calc = DifferentiableCoM_Patch_3x5(patch_h, patch_w)
        self.ph_offset = patch_h // 2
        self.pw_offset_18col = patch_w // 2
    def forward(self, patches, peak_rs, peak_cs_18_col):
        local_coords_3x5 = self.local_com_calc(patches)
        local_x_scalar = local_coords_3x5[:, 0]
        local_y_scalar = local_coords_3x5[:, 1]
        global_x_18col_scalar = local_x_scalar + peak_cs_18_col - self.pw_offset_18col
        global_y_scalar = local_y_scalar + peak_rs - self.ph_offset
        return torch.stack([global_x_18col_scalar, global_y_scalar], dim=1)

class SensorDataset_V16_Final(Dataset):
    def __init__(self, data_dir, patch_size):
        self.patch_h, self.patch_w = patch_size
        self.odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
        self.odd_single = [(8,4), (9,5)]; self.odd_map_18_to_10 = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:5, 10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9}
        self.even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
        self.even_single= [(0,0), (17,9)]; self.even_map_18_to_10 = {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:5, 11:6, 12:6, 13:7, 14:7, 15:8, 16:8, 17:9}

        samples = []
        all_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.json')])
        pad2d_18_col = (self.patch_w//2, self.patch_w//2, self.patch_h//2, self.patch_h//2)
        pad2d_10_col = pad2d_18_col
        eps = 1e-8
        for filename in all_files:
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f: aligned_data = json.load(f)
                for point_id, pair_data in aligned_data.items():
                    merging_18 = np.array(pair_data['merging']['normalized_matrix'], dtype=np.float32)
                    target_18 = np.array(pair_data['nonmerging']['normalized_matrix'], dtype=np.float32)
                    if merging_18.shape != (32, 18):
                        continue
                    peak_r_gt, peak_c_gt_18 = np.unravel_index(np.argmax(target_18), (32, 18))
                    target_t_18 = F.pad(torch.from_numpy(target_18), pad2d_18_col)
                    r0_t, r1_t = peak_r_gt, peak_r_gt + self.patch_h
                    c0_t, c1_t = peak_c_gt_18, peak_c_gt_18 + self.patch_w
                    clean_patch = target_t_18[r0_t:r1_t, c0_t:c1_t]

                    # compress to 10 col
                    effective_merging_10 = np.zeros((32,10), dtype=np.float32)
                    for r in range(32):
                        is_odd = (r % 2 == 1)
                        pairs = self.odd_pairs if is_odd else self.even_pairs
                        map_18_to_10 = self.odd_map_18_to_10 if is_odd else self.even_map_18_to_10
                        singles = self.odd_single if is_odd else self.even_single
                        for (c_left, c_right) in pairs:
                            idx = map_18_to_10[c_left]
                            effective_merging_10[r, idx] = (merging_18[r, c_left] + merging_18[r, c_right]) / 2.0
                        for (c18, idx) in singles:
                            effective_merging_10[r, idx] = merging_18[r, c18]

                    peak_r_m, peak_c_m_10 = np.unravel_index(np.argmax(effective_merging_10), (32, 10))
                    is_odd_flag = float(peak_r_m % 2 == 1)
                    merging_t_10 = F.pad(torch.from_numpy(effective_merging_10), pad2d_10_col)
                    r0_m, r1_m = peak_r_m, peak_r_m + self.patch_h
                    c0_m, c1_m = peak_c_m_10, peak_c_m_10 + self.patch_w
                    merging_patch = merging_t_10[r0_m:r1_m, c0_m:c1_m]
                    merging_patch = merging_patch.clamp_min(0); merging_patch = merging_patch / (merging_patch.sum() + eps)
                    clean_patch = clean_patch.clamp_min(0); clean_patch = clean_patch / (clean_patch.sum() + eps)

                    samples.append({
                        'raw_patch': merging_patch.unsqueeze(0), 'clean_patch': clean_patch.unsqueeze(0), 'is_odd': is_odd_flag,
                        'peak_r_gt': peak_r_gt, 'peak_c_gt_18': peak_c_gt_18,
                        'peak_r_m': peak_r_m, 'peak_c_m_10': peak_c_m_10,
                    })
            except Exception as e:
                print(f"跳过 {filename}: {e}")
        if len(samples) == 0:
            raise RuntimeError(f"未能从 {data_dir} 加载任何样本")
        # stack tensors; ensure channel dim exists -> shape (N, 1, H, W)
        self.raw_patches = torch.cat([s['raw_patch'] for s in samples], dim=0).unsqueeze(1)
        self.clean_patches = torch.cat([s['clean_patch'] for s in samples], dim=0).unsqueeze(1)
        self.is_odd = torch.tensor([s['is_odd'] for s in samples], dtype=torch.float32)

        # collect peak anchors for coordinate distillation
        peak_r_gt_list = [s['peak_r_gt'] for s in samples]
        peak_c_18_gt_list = [s['peak_c_gt_18'] for s in samples]
        self.peak_coords_18_gt = torch.tensor(list(zip(peak_r_gt_list, peak_c_18_gt_list)), dtype=torch.float32)
        peak_r_m_list = [s['peak_r_m'] for s in samples]
        peak_c_10_m_list = [s['peak_c_m_10'] for s in samples]
        self.peak_coords_10_m = torch.tensor(list(zip(peak_r_m_list, peak_c_10_m_list)), dtype=torch.float32)

    def __len__(self): return self.raw_patches.shape[0]
    def __getitem__(self, idx):
        return (
            self.raw_patches[idx], self.clean_patches[idx], self.is_odd[idx],
            self.peak_coords_18_gt[idx][0], self.peak_coords_18_gt[idx][1],
            self.peak_coords_10_m[idx][0], self.peak_coords_10_m[idx][1]
        )


# -----------------------------
# QAT 训练与导出辅助函数
# -----------------------------

def fuse_student_modules(model: StudentModel):
    # fuse conv-bn-relu triplets
    for name in ['odd_net', 'even_net']:
        module = getattr(model, name)
        module.eval()
        fuse_list = [['net.0','net.1','net.2'], ['net.3','net.4','net.5'], ['net.6','net.7','net.8']]
        torch.quantization.fuse_modules(module, fuse_list, inplace=True)

def export_onnx(model_original, args):
    """
    导出带有 FakeQuantize 节点的 ONNX 模型 (QDQ 格式)。
    修复方案 V4 (终极版): 使用物理替换法 (Hard Replacement)。
    将所有的 FakeQuantize 模块替换为自定义的静态 Wrapper，
    彻底杜绝 'fused_moving_avg' 算子出现的可能性。
    """
    print(">>> 正在准备导出 QAT ONNX 模型 (使用静态算子替换法)...")
    
    # ---------------------------------------------------------
    # 内部类：自定义的静态 QDQ 包装器
    # ---------------------------------------------------------
    class QDQExportWrapper(nn.Module):
        def __init__(self, fq_module):
            super().__init__()
            # 1. 复制量化边界
            self.quant_min = fq_module.quant_min
            self.quant_max = fq_module.quant_max
            
            # 2. 复制并注册 Scale 和 ZeroPoint
            # 注意：必须 detach() 以断开与原图的梯度联系
            self.register_buffer('scale', fq_module.scale.detach().clone())
            self.register_buffer('zero_point', fq_module.zero_point.detach().clone())
            
            # 3. 识别 Per-Channel 还是 Per-Tensor
            # qnnpack 通常权重是 Per-Channel，激活是 Per-Tensor
            self.ch_axis = getattr(fq_module, 'ch_axis', -1)
            
            # 4. 检查开关状态
            # 如果 fake_quant 被禁用，我们应当直接返回输入
            self.enabled = True
            if hasattr(fq_module, 'fake_quant_enabled'):
                if fq_module.fake_quant_enabled.item() == 0:
                    self.enabled = False

        def forward(self, X):
            # 如果未启用，直接透传 (相当于 Identity)
            if not self.enabled:
                return X
                
            # 显式调用静态伪量化算子
            # 这些算子会被 ONNX 导出器识别为 QuantizeLinear + DequantizeLinear
            if self.ch_axis != -1:
                return torch.fake_quantize_per_channel_affine(
                    X, self.scale, self.zero_point, 
                    self.ch_axis, self.quant_min, self.quant_max
                )
            else:
                return torch.fake_quantize_per_tensor_affine(
                    X, self.scale, self.zero_point, 
                    self.quant_min, self.quant_max
                )

    # ---------------------------------------------------------
    # 辅助函数：递归替换模块
    # ---------------------------------------------------------
    def replace_fq_with_static(module):
        for name, child in module.named_children():
            if isinstance(child, tq.FakeQuantize):
                # 发现 FakeQuantize，执行替换
                print(f"  [替换] replacing {name} with static QDQ node...")
                static_fq = QDQExportWrapper(child)
                setattr(module, name, static_fq)
            else:
                # 递归搜索
                replace_fq_with_static(child)

    # ---------------------------------------------------------
    # 1. 创建模型副本 (Deep Copy)
    # ---------------------------------------------------------
    model_original.cpu()
    try:
        model_export = copy.deepcopy(model_original)
    except Exception as e:
        print(f"Warning: Deepcopy failed ({e}), creating new instance.")
        # Fallback 重建逻辑
        if args.student_channels:
            chs = [int(x) for x in args.student_channels.split(',') if x.strip()]
            model_export = make_student_model(channels=chs)
        else:
            mult = float(args.student_mult) if args.student_mult is not None else None
            model_export = make_student_model(multiplier=mult)
        fuse_student_modules(model_export)
        model_export.qconfig = tq.get_default_qat_qconfig('qnnpack')
        tq.prepare_qat(model_export, inplace=True)
        model_export.load_state_dict(model_original.state_dict())
    
    # 恢复原模型到 GPU，避免影响后续训练
    model_original.to(device)

    # ---------------------------------------------------------
    # 2. 执行“手术”：物理替换所有 FakeQuantize
    # ---------------------------------------------------------
    model_export.to('cpu')
    model_export.eval()
    
    print(">>> 开始替换 FakeQuantize 模块...")
    replace_fq_with_static(model_export)
    print(">>> 替换完成。所有统计更新算子已被移除。")

    # ---------------------------------------------------------
    # 3. 导出 ONNX
    # ---------------------------------------------------------
    dummy_input_patch = torch.randn(1, 1, 3, 5)
    dummy_input_odd = torch.tensor([1.0])
    
    # 刷新一次 (虽然对于静态模块不是必须的，但保险起见)
    with torch.no_grad():
        model_export(dummy_input_patch, dummy_input_odd)

    onnx_path = os.path.join(args.out_dir, 'student_qat.onnx')
    
    try:
        print(f">>> 开始导出 ONNX: {onnx_path}")
        torch.onnx.export(
            model_export,
            (dummy_input_patch, dummy_input_odd),
            onnx_path,
            verbose=False,
            opset_version=13,  # 必须 >= 13
            input_names=['input_patch', 'input_is_odd'],
            output_names=['output'],
            do_constant_folding=True,
            training=torch.onnx.TrainingMode.EVAL,
            keep_initializers_as_inputs=False,
            dynamic_axes={
                'input_patch': {0: 'batch_size'},
                'input_is_odd': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f">>> ONNX 导出成功！(已包含静态 QDQ 节点)")
        return True
    except Exception as e:
        print(f"!!! ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        del model_export

def train_qat(args):
    # 再次设置引擎
    torch.backends.quantized.engine = 'qnnpack'
    
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        cfg = vars(args).copy()
        if cfg.get('student_channels') is None: cfg['student_channels'] = None
        with open(os.path.join(args.out_dir, 'config.json'), 'w') as _cf:
            _json.dump(cfg, _cf, indent=2)
    except Exception: pass
    
    dataset = SensorDataset_V16_Final(args.data_dir, PATCH_SIZE)
    n = len(dataset)
    train_n = int(0.8 * n)
    val_n = n - train_n
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_n, val_n])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    teacher = TeacherModel().to(device)
    if args.teacher_path and os.path.isfile(args.teacher_path):
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
    teacher.eval();
    for p in teacher.parameters(): p.requires_grad = False

    if args.student_channels:
        chs = [int(x) for x in args.student_channels.split(',') if x.strip()]
        student = make_student_model(channels=chs).to(device)
    else:
        mult = float(args.student_mult) if args.student_mult is not None else None
        student = make_student_model(multiplier=mult).to(device)
    
    if args.init_student and os.path.isfile(args.init_student):
        student.load_state_dict(torch.load(args.init_student, map_location=device))
        print(f"初始化学生模型: {args.init_student}")

    # Fuse
    fuse_student_modules(student)

    # Prepare QAT (QNNPACK config)
    student.qconfig = tq.get_default_qat_qconfig('qnnpack')
    tq.prepare_qat(student, inplace=True)
    student.to(device)

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    mse_loss = nn.MSELoss()
    ALPHA = args.alpha

    com_calc_from_10col_patch = CoM_from_Patch_V12(patch_h=3, patch_w=5).to(device)
    com_calc_from_18col_patch = CoM_from_18col_Patch_V15(patch_h=3, patch_w=5).to(device)

    best_val = float('inf')

    for epoch in range(args.epochs):
        student.train()
        total_loss = total_coords = total_distill = 0.0
        for batch in tqdm(train_loader, desc=f"Train E{epoch+1}"):
            raw_patch, clean_patch, is_odd, peak_r_gt, peak_c_gt_18, peak_r_m, peak_c_m_10 = batch
            raw_patch = raw_patch.to(device); clean_patch = clean_patch.to(device); is_odd = is_odd.to(device)
            peak_r_gt = peak_r_gt.to(device); peak_c_gt_18 = peak_c_gt_18.to(device)
            peak_r_m = peak_r_m.to(device); peak_c_m_10 = peak_c_m_10.to(device)
            
            with torch.no_grad():
                pred_teacher = teacher(raw_patch, is_odd)
            pred_student = student(raw_patch, is_odd)
            
            loss_distill = mse_loss(pred_student, pred_teacher)
            if args.distill_coord:
                pred_global_coords_student = com_calc_from_10col_patch(pred_student, is_odd, peak_r_m, peak_c_m_10)
                with torch.no_grad():
                    target_global_coords = com_calc_from_18col_patch(clean_patch, peak_r_gt, peak_c_gt_18)
                loss_coords = mse_loss(pred_global_coords_student, target_global_coords)
                loss = (ALPHA * loss_coords) + ((1.0 - ALPHA) * loss_distill)
            else:
                loss = (1.0 - ALPHA) * loss_distill
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item(); total_distill += loss_distill.item()
        scheduler.step(total_distill / max(1, len(train_loader)))

        student.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                raw_patch, clean_patch, is_odd, peak_r_gt, peak_c_gt_18, peak_r_m, peak_c_m_10 = batch
                raw_patch = raw_patch.to(device); clean_patch = clean_patch.to(device); is_odd = is_odd.to(device)
                peak_r_gt = peak_r_gt.to(device); peak_c_gt_18 = peak_c_gt_18.to(device)
                peak_r_m = peak_r_m.to(device); peak_c_m_10 = peak_c_m_10.to(device)
                pred_teacher = teacher(raw_patch, is_odd)
                pred_student = student(raw_patch, is_odd)
                loss_distill = mse_loss(pred_student, pred_teacher)
                if args.distill_coord:
                    pred_global_coords_student = com_calc_from_10col_patch(pred_student, is_odd, peak_r_m, peak_c_m_10)
                    target_global_coords = com_calc_from_18col_patch(clean_patch, peak_r_gt, peak_c_gt_18)
                    loss_coords = mse_loss(pred_global_coords_student, target_global_coords)
                    val_loss += (ALPHA * loss_coords + (1.0 - ALPHA) * loss_distill).item()
                else:
                    val_loss += loss_distill.item()
        val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{args.epochs}  TrainLoss: {total_loss/len(train_loader):.6f}  ValDistill: {val_loss:.6f}")

        float_path = os.path.join(args.out_dir, 'best_qat_float.pt')
        torch.save(student.state_dict(), float_path)

        if val_loss < best_val:
            best_val = val_loss
            # 使用新定义的 Robust 导出函数
            export_success = export_onnx(student, args)

            try:
                student.cpu()
                student_int8 = tq.convert(student, inplace=False)
                int8_path = os.path.join(args.out_dir, 'best_qat_int8.pt')
                torch.save(student_int8.state_dict(), int8_path)
                print(f"Saved best int8 -> {int8_path}")
            except Exception as e:
                print(f"警告: 无法在当前平台转换 Int8 PT 模型: {e}")
            finally:
                student.to(device)

    print("QAT 训练完成。请使用 student_qat.onnx 进行后续部署。")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=JSON_DATA_DIR_DEFAULT)
    p.add_argument('--teacher-path', default='/work/hwc/SPARSE/distill/decoupler_model_v16_teacher_best.pth')
    p.add_argument('--init-student', default='/work/hwc/SPARSE/distill/pths/decoupler_model_v16_student_best.pth')
    p.add_argument('--out-dir', default='/work/hwc/SPARSE/distill/qat_student_runs')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--student-channels', type=str, default=None)
    p.add_argument('--student-mult', type=float, default=None)
    p.add_argument('--distill-coord', action='store_true')
    p.add_argument('--alpha', type=float, default=0.3)
    return p.parse_args()

if __name__ == '__main__':
    JSON_DATA_DIR_DEFAULT = '/work/hwc/SPARSE/training_data/aligned_data_for_training_int'
    PATCH_SIZE = (3, 5)
    args = parse_args()
    print(f"Device: {device}. Using quant backend qnnpack.\nData dir: {args.data_dir}")
    train_qat(args)