#!/usr/bin/env python3
# QAT 训练脚本（针对 V16 学生模型），使用 fbgemm 后端（适用于 x86），在 GPU 上运行
"""
Usage:
  python3 train_student_qat.py --epochs 10 --batch-size 64 --out-dir runs_sparse_v2_3x6_qat_minmax

该脚本会：
- 加载训练数据（与 `student_distill.py` 中相同的预处理流程）
- 实例化 TeacherModel、StudentModel；加载教师权重（若指定）并冻结
- 对 StudentModel 进行模块融合并准备 QAT（fbgemm）
- 运行若干 epoch 的微调训练
- 将量化后的 int8 模型状态字典保存为 `best_qat_int8.pt`（并保存 float 版本）
"""
import os
import argparse
import json
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

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


# -----------------------------
# 复制（简化）Student/Teacher/数据加载定义
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
# CoM modules (copy minimal implementations used by other scripts)
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
# QAT 训练逻辑
# -----------------------------

def fuse_student_modules(model: StudentModel):
    # fuse conv-bn-relu triplets inside each submodule's sequential
    for name in ['odd_net', 'even_net']:
        module = getattr(model, name)
        # ensure eval mode for fusion
        module.eval()
        # net is an nn.Sequential with pattern: Conv(0),BN(1),ReLU(2), Conv(3),BN(4),ReLU(5), Conv(6),BN(7),ReLU(8), Conv(9)
        fuse_list = [['net.0','net.1','net.2'], ['net.3','net.4','net.5'], ['net.6','net.7','net.8']]
        torch.quantization.fuse_modules(module, fuse_list, inplace=True)
        # keep module in train mode will be set by caller if needed


def train_qat(args):
    torch.backends.quantized.engine = 'fbgemm'
    # save config for reproducibility
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        cfg = vars(args).copy()
        # ensure any non-serializable entries are converted
        if cfg.get('student_channels') is None:
            cfg['student_channels'] = None
        with open(os.path.join(args.out_dir, 'config.json'), 'w') as _cf:
            _json.dump(cfg, _cf, indent=2)
    except Exception:
        pass
    # dataset
    dataset = SensorDataset_V16_Final(args.data_dir, PATCH_SIZE)
    n = len(dataset)
    train_n = int(0.8 * n)
    val_n = n - train_n
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_n, val_n])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # models
    teacher = TeacherModel().to(device)
    if args.teacher_path and os.path.isfile(args.teacher_path):
        teacher.load_state_dict(torch.load(args.teacher_path, map_location=device))
        print(f"Loaded teacher weights: {args.teacher_path}")
    teacher.eval();
    for p in teacher.parameters(): p.requires_grad = False
    # Build student from factory (supports --student-channels or --student-mult)
    if args.student_channels:
        chs = [int(x) for x in args.student_channels.split(',') if x.strip()]
        student = make_student_model(channels=chs).to(device)
    else:
        mult = float(args.student_mult) if args.student_mult is not None else None
        student = make_student_model(multiplier=mult).to(device)
    
    print("Student Model Structure:")
    print(student)
    print("\nDetailed model structure:")
    for name, module in student.named_modules():
        print(f"{name}: {module}")
    # optionally init from pre-trained student
    if args.init_student and os.path.isfile(args.init_student):
        student.load_state_dict(torch.load(args.init_student, map_location=device))
        print(f"初始化学生模型来自: {args.init_student}")

    # Fuse modules for QAT
    fuse_student_modules(student)

    # QAT config
    student.qconfig = tq.get_default_qat_qconfig('fbgemm')
    tq.prepare_qat(student, inplace=True)
    student.to(device)

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    mse_loss = nn.MSELoss()
    ALPHA = args.alpha

    # CoM modules for optional coordinate distillation
    com_calc_from_10col_patch = CoM_from_Patch_V12(patch_h=3, patch_w=5).to(device)
    com_calc_from_18col_patch = CoM_from_18col_Patch_V15(patch_h=3, patch_w=5).to(device)

    best_val = float('inf')
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        student.train()
        total_loss = total_coords = total_distill = 0.0
        for batch in tqdm(train_loader, desc=f"Train E{epoch+1}"):
            raw_patch, clean_patch, is_odd, peak_r_gt, peak_c_gt_18, peak_r_m, peak_c_m_10 = batch
            raw_patch = raw_patch.to(device); clean_patch = clean_patch.to(device); is_odd = is_odd.to(device)
            peak_r_gt = peak_r_gt.to(device); peak_c_gt_18 = peak_c_gt_18.to(device)
            peak_r_m = peak_r_m.to(device); peak_c_m_10 = peak_c_m_10.to(device)
            # teacher pred
            with torch.no_grad():
                pred_teacher = teacher(raw_patch, is_odd)
            pred_student = student(raw_patch, is_odd)
            # pixel-wise distillation
            loss_distill = mse_loss(pred_student, pred_teacher)
            if args.distill_coord:
                # compute coordinate (hard) loss
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

        # validation
        student.eval(); val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                raw_patch, clean_patch, is_odd, peak_r_gt, peak_c_gt_18, peak_r_m, peak_c_m_10 = batch
                raw_patch = raw_patch.to(device); clean_patch = clean_patch.to(device); is_odd = is_odd.to(device)
                peak_r_gt = peak_r_gt.to(device); peak_c_gt_18 = peak_c_gt_18.to(device)
                peak_r_m = peak_r_m.to(device); peak_c_m_10 = peak_c_m_10.to(device)
                pred_teacher = teacher(raw_patch, is_odd)
                pred_student = student(raw_patch, is_odd)
                if args.distill_coord:
                    pred_global_coords_student = com_calc_from_10col_patch(pred_student, is_odd, peak_r_m, peak_c_m_10)
                    target_global_coords = com_calc_from_18col_patch(clean_patch, peak_r_gt, peak_c_gt_18)
                    loss_coords = mse_loss(pred_global_coords_student, target_global_coords)
                    loss_distill = mse_loss(pred_student, pred_teacher)
                    val_loss += (ALPHA * loss_coords + (1.0 - ALPHA) * loss_distill).item()
                else:
                    loss_distill = mse_loss(pred_student, pred_teacher)
                    val_loss += loss_distill.item()
        val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{args.epochs}  TrainLoss: {total_loss/len(train_loader):.6f}  ValDistill: {val_loss:.6f}")

        # save best float (still in QAT prepared state — convert a floating copy later)
        float_path = os.path.join(args.out_dir, 'best_qat_float.pt')
        torch.save(student.state_dict(), float_path)

        if val_loss < best_val:
            best_val = val_loss
            # convert and save int8
            student.cpu()
            student_int8 = tq.convert(student, inplace=False)
            int8_path = os.path.join(args.out_dir, 'best_qat_int8.pt')
            torch.save(student_int8.state_dict(), int8_path)
            print(f"Saved best int8 -> {int8_path} (val {best_val:.6f})")
            student.to(device)

    print("QAT 训练完成。导出文件位于:", args.out_dir)

# Copy minimal dataset preprocessor (same behavior as student_distill)
JSON_DATA_DIR_DEFAULT = '/work/hwc/SPARSE/training_data/aligned_data_for_training_int'
PATCH_SIZE = (3, 5)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', default=JSON_DATA_DIR_DEFAULT)
    p.add_argument('--teacher-path', default='/work/hwc/SPARSE/distill/decoupler_model_v16_teacher_best.pth')
    p.add_argument('--init-student', default='/work/hwc/SPARSE/distill/pths/decoupler_model_v16_student_best.pth')
    p.add_argument('--out-dir', default='/work/hwc/SPARSE/distill/qat_student_runs')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--student-channels', type=str, default=None,
                   help='Comma-separated intermediate channels for student, e.g. "2,4,2"')
    p.add_argument('--student-mult', type=float, default=None,
                   help='Multiplier to scale default channels [2,4,2]')
    p.add_argument('--distill-coord', action='store_true', help='Enable coordinate (CoM) distillation loss')
    p.add_argument('--alpha', type=float, default=0.3, help='Weight for coordinate loss when distilling')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(f"Device: {device}. Using quant backend fbgemm.\nData dir: {args.data_dir}")
    train_qat(args)
