# -*- coding: utf-8 -*-
"""
V17 Teacher 训练脚本 (Residual 版):
- 核心改进 1: 引入 Anchor Jitter 和 噪声注入 (增强鲁棒性)。
- 核心改进 2: CoordConv (3通道输入)，辅助空间感知。
- 核心改进 3 [NEW]: Residual Learning (残差学习)。
  模型不再直接预测色块，而是预测“修正量”。
  Output = Input_Intensity + Network(Input_3Ch)
  这极大地降低了小模型的学习难度，更适合 MCU 部署。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ==========================================
# 1. CoM 计算模块 (物理规则层，保持不变)
# ==========================================

class DifferentiableCoM_Patch_3x5(nn.Module):
    """(子模块) 在 3x5 色块局部坐标系中计算 (x, y) [0-4, 0-2]"""
    def __init__(self, height=3, width=5):
        super(DifferentiableCoM_Patch_3x5, self).__init__()
        self.height, self.width = height, width
        y, x = torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)
        x_grid, y_grid = x.view(1, -1).repeat(height, 1), y.view(-1, 1).repeat(1, width)
        self.register_buffer('x_grid_buf', x_grid)
        self.register_buffer('y_grid_buf', y_grid)

    def forward(self, image):
        # image: [Batch, 1, 3, 5] 或 [Batch, 3, 5]
        if image.dim() == 4 and image.shape[1] == 3:
            img = image[:, 0, :, :] # 取 Intensity 通道
        elif image.dim() == 4:
            img = image.squeeze(1)
        else:
            img = image
            
        eps = 1e-8
        mass = torch.sum(img, dim=(1, 2)) + eps
        x_weighted = torch.sum(img * self.x_grid_buf, dim=(1, 2))
        y_weighted = torch.sum(img * self.y_grid_buf, dim=(1, 2))
        center_x, center_y = x_weighted / mass, y_weighted / mass
        return torch.stack([center_x, center_y], dim=1)

class CoM_from_Patch_V12(nn.Module):
    """
    (V16/V17 Pred 管线)
    (3x5 色块, 10-col 峰值, 奇偶标记) -> 18-col 全局坐标
    """
    def __init__(self, patch_h=3, patch_w=5):
        super(CoM_from_Patch_V12, self).__init__()
        self.local_com_calc = DifferentiableCoM_Patch_3x5(patch_h, patch_w)
        self.ph_offset = patch_h // 2
        self.pw_offset_10col = patch_w // 2
        # 物理 Grid 查找表
        odd_grid_10_np = np.array([0.5, 2.5, 4.5, 6.5, 8.0, 9.0, 10.5, 12.5, 14.5, 16.5], dtype=np.float32)
        even_grid_10_np = np.array([0.0, 1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.0], dtype=np.float32)
        self.register_buffer('odd_grid_10', torch.from_numpy(odd_grid_10_np))
        self.register_buffer('even_grid_10', torch.from_numpy(even_grid_10_np))

    def forward(self, patches, is_odd_flags, peak_rs, peak_cs_10_col):
        local_coords_3x5 = self.local_com_calc(patches)
        local_x_10col_scalar = local_coords_3x5[:, 0]
        local_y_scalar = local_coords_3x5[:, 1]
        
        # 计算全局 10-col 坐标
        global_x_10col_scalar = local_x_10col_scalar + peak_cs_10_col - self.pw_offset_10col
        global_y_scalar = local_y_scalar + peak_rs - self.ph_offset
        
        # 查表映射
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
    """(V16/V17 GT 管线)"""
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


# ==========================================
# 2. 模型定义 (V17 残差版)
# ==========================================

class _PatchNet_V17_Residual(nn.Module):
    """
    V17 Network:
    - 输入: 3 通道 (Intensity, Rel_X, Rel_Y)
    - 输出: 1 通道 (Residual Map)
    - 逻辑: Output = Input_Intensity + Residual_Map
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # 保持轻量级，便于 Student 模仿
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # 最后一层没有 ReLU，因为残差可以是负的
            nn.Conv2d(16, out_channels, 1) 
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        residual = self.net(x) # [B, 1, H, W]
        
        # 提取输入的强度通道作为 Base
        input_intensity = x[:, 0:1, :, :] # [B, 1, H, W]
        
        # 残差连接: 修正后的 Patch = 原始 Patch + 修正量
        out = input_intensity + residual
        
        return out

class TeacherModelV17(nn.Module):
    def __init__(self):
        super(TeacherModelV17, self).__init__()
        self.odd_net = _PatchNet_V17_Residual(in_channels=3)
        self.even_net = _PatchNet_V17_Residual(in_channels=3)

    def forward(self, x_patches, is_odd_flags):
        # x_patches shape: [B, 3, 3, 5]
        out_odd = self.odd_net(x_patches)
        out_even = self.even_net(x_patches)
        
        is_odd_mask = is_odd_flags.view(-1, 1, 1, 1) > 0.5
        is_odd_mask = is_odd_mask.to(out_odd.device)
        out = torch.where(is_odd_mask, out_odd, out_even)
        
        # 物理约束: 电容值/光强不能为负
        return torch.relu(out)


# ==========================================
# 3. 数据集 (保持 Jitter + CoordConv)
# ==========================================

JSON_DATA_DIR = '/work/hwc/SPARSE/training_data/aligned_data_for_training_int'
PATCH_SIZE = (3, 5)

class SensorDataset_V17(Dataset):
    def __init__(self, data_dir, patch_size, device, augment=False):
        self.device = device
        self.patch_h, self.patch_w = patch_size
        self.augment = augment 
        
        print(f"V17 Dataset Init | Augment={augment} | Dir: {data_dir}")

        self.odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
        self.odd_single = [(8,4), (9,5)]; self.odd_map_18_to_10 = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:5, 10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9}
        self.even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
        self.even_single= [(0,0), (17,9)]; self.even_map_18_to_10 = {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:5, 11:6, 12:6, 13:7, 14:7, 15:8, 16:8, 17:9}

        # CoordConv Grid
        y_coords = torch.linspace(-1, 1, self.patch_h)
        x_coords = torch.linspace(-1, 1, self.patch_w)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        self.coord_grid = torch.stack([grid_x, grid_y], dim=0).to(device) 

        samples_cpu = self._load_and_process_full_matrices(data_dir)
        if not samples_cpu: 
            print("Error: No samples loaded.")
            self.n_samples = 0
            return

        print("Moving full matrices to GPU for dynamic cropping...")
        self.merging_matrices = torch.stack([s["merging_10"] for s in samples_cpu]).to(device) 
        self.target_matrices = torch.stack([s["target_18"] for s in samples_cpu]).to(device)
        
        self.initial_peaks_m = torch.tensor([s["peak_m"] for s in samples_cpu], dtype=torch.long).to(device)
        self.initial_peaks_gt = torch.tensor([s["peak_gt"] for s in samples_cpu], dtype=torch.long).to(device)

        self.n_samples = len(self.merging_matrices)
        print(f"Dataset ready. {self.n_samples} samples.")

    def _compress_to_10_col(self, matrix_18_col):
        matrix_10_col = np.zeros((32, 10), dtype=np.float32)
        for r in range(32):
            is_odd = (r % 2 == 1)
            pairs = self.odd_pairs if is_odd else self.even_pairs
            singles = self.odd_single if is_odd else self.even_single
            map_18_to_10 = self.odd_map_18_to_10 if is_odd else self.even_map_18_to_10
            for (c_left, c_right) in pairs:
                col_10_idx = map_18_to_10[c_left]
                avg = (matrix_18_col[r, c_left] + matrix_18_col[r, c_right]) / 2.0
                matrix_10_col[r, col_10_idx] = avg
            for (c_18, col_10_idx) in singles: 
                matrix_10_col[r, col_10_idx] = matrix_18_col[r, c_18]
        return matrix_10_col

    def _load_and_process_full_matrices(self, data_dir):
        samples = []
        all_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.json')])
        ph, pw = self.patch_h // 2, self.patch_w // 2
        pad_18 = (pw, pw, ph, ph)
        pad_10 = (pw, pw, ph, ph)

        for filename in tqdm(all_files, desc="Loading JSONs"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f: aligned_data = json.load(f)
                for point_id, pair_data in aligned_data.items():
                    merging_18 = np.array(pair_data['merging']['normalized_matrix'], dtype=np.float32)
                    target_18 = np.array(pair_data['nonmerging']['normalized_matrix'], dtype=np.float32)
                    if merging_18.shape != (32, 18): continue

                    target_tensor = torch.from_numpy(target_18)
                    peak_r_gt, peak_c_gt = np.unravel_index(np.argmax(target_18), (32, 18))
                    target_padded = F.pad(target_tensor, pad_18) 

                    effective_merging_10 = self._compress_to_10_col(merging_18)
                    merging_tensor = torch.from_numpy(effective_merging_10)
                    peak_r_m, peak_c_m = np.unravel_index(np.argmax(effective_merging_10), (32, 10))
                    merging_padded = F.pad(merging_tensor, pad_10)

                    samples.append({
                        "merging_10": merging_padded,
                        "target_18": target_padded,
                        "peak_m": [peak_r_m, peak_c_m],
                        "peak_gt": [peak_r_gt, peak_c_gt]
                    })
            except Exception as e: print(f"Skip {filename}: {e}")
        return samples

    def __len__(self): return self.n_samples

    def __getitem__(self, idx):
        merging_full = self.merging_matrices[idx]
        target_full = self.target_matrices[idx]
        
        peak_r_m_base, peak_c_m_base = self.initial_peaks_m[idx]
        peak_r_gt, peak_c_gt = self.initial_peaks_gt[idx]
        
        # Anchor Jitter
        r_offset, c_offset = 0, 0
        if self.augment:
            r_offset = random.randint(-1, 1)
            c_offset = random.randint(-1, 1)
        
        current_peak_r = peak_r_m_base + r_offset
        current_peak_c = peak_c_m_base + c_offset
        
        r_start = current_peak_r
        c_start = current_peak_c
        
        merging_patch = merging_full[r_start : r_start + self.patch_h, 
                                     c_start : c_start + self.patch_w]
        
        r_start_gt = peak_r_gt
        c_start_gt = peak_c_gt
        clean_patch = target_full[r_start_gt : r_start_gt + self.patch_h,
                                  c_start_gt : c_start_gt + self.patch_w]

        # Noise Injection
        if self.augment:
            noise = torch.randn_like(merging_patch) * 0.02 
            merging_patch = merging_patch + noise
            
        eps = 1e-8
        merging_patch = merging_patch.clamp_min(0)
        merging_patch = merging_patch / (merging_patch.sum() + eps)
        
        clean_patch = clean_patch.clamp_min(0)
        clean_patch = clean_patch / (clean_patch.sum() + eps)

        # CoordConv
        input_tensor = torch.cat([merging_patch.unsqueeze(0), self.coord_grid], dim=0)
        clean_patch_out = clean_patch.unsqueeze(0)
        
        is_odd = float(current_peak_r % 2 == 1)

        return (
            input_tensor, clean_patch_out, is_odd,
            peak_r_gt, peak_c_gt,
            current_peak_r, current_peak_c
        )


# ==========================================
# 4. 训练流程
# ==========================================

try:
    full_dataset_train = SensorDataset_V17(JSON_DATA_DIR, patch_size=PATCH_SIZE, device=device, augment=True)
    full_dataset_val = SensorDataset_V17(JSON_DATA_DIR, patch_size=PATCH_SIZE, device=device, augment=False)
    
    total_len = len(full_dataset_train)
    indices = list(range(total_len))
    split = int(0.8 * total_len)
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    train_subset = torch.utils.data.Subset(full_dataset_train, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset_val, val_indices)
    
    BATCH_SIZE = 64
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"V17 Data Loaders Ready. Train: {len(train_subset)}, Val: {len(val_subset)}")
    
except Exception as e:
    print(f"Data loading failed: {e}")
    train_loader = None

com_calc_pred = CoM_from_Patch_V12(patch_h=3, patch_w=5).to(device)
com_calc_gt = CoM_from_18col_Patch_V15(patch_h=3, patch_w=5).to(device)
mse_loss_fn = nn.MSELoss()

def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    total_loss = 0
    
    for input_tensor, clean_patch, is_odd, gt_r, gt_c, in_r, in_c in loader:
        optimizer.zero_grad()
        
        pred_patch_1ch = model(input_tensor, is_odd) 
        
        # Pred using Jittered Anchor
        pred_coords = com_calc_pred(pred_patch_1ch, is_odd, in_r, in_c)
        
        # GT using True Anchor
        with torch.no_grad():
            target_coords = com_calc_gt(clean_patch, gt_r, gt_c)
            
        loss = mse_loss_fn(pred_coords, target_coords)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / max(1, len(loader))
    scheduler.step(avg_loss)
    return avg_loss

def validate_epoch(model, loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for input_tensor, clean_patch, is_odd, gt_r, gt_c, in_r, in_c in loader:
            pred_patch_1ch = model(input_tensor, is_odd)
            pred_coords = com_calc_pred(pred_patch_1ch, is_odd, in_r, in_c)
            target_coords = com_calc_gt(clean_patch, gt_r, gt_c)
            loss = mse_loss_fn(pred_coords, target_coords)
            total_loss += loss.item()
    return total_loss / max(1, len(loader))

# ==========================================
# 5. 主执行块
# ==========================================

MODEL_SAVE_PATH_V17 = '/work/hwc/SPARSE/distill/decoupler_model_v17_teacher_best.pth'
NUM_EPOCHS = 100

if train_loader:
    print("\n--- Starting V17 Training (Residual Mode) ---")
    model = TeacherModelV17().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss = validate_epoch(model, val_loader)
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | LR: {lr:.1e} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH_V17)
            print("  -> Best Model Saved")
            
    print(f"Training Complete. Best Val Loss: {best_val_loss:.5f}")
    print(f"Model saved to: {MODEL_SAVE_PATH_V17}")

# ==========================================
# 6. 可视化验证
# ==========================================

def visualize_v17(json_path, model_path):
    if not os.path.exists(json_path): return
    print(f"\nVisualizing {os.path.basename(json_path)}...")
    
    viz_model = TeacherModelV17().to(device)
    viz_model.load_state_dict(torch.load(model_path, map_location=device))
    viz_model.eval()
    
    with open(json_path, 'r') as f: data = json.load(f)
    
    y_coords = torch.linspace(-1, 1, 3)
    x_coords = torch.linspace(-1, 1, 5)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    coord_grid = torch.stack([grid_x, grid_y], dim=0).to(device) 
    
    gt_x, gt_y = [], []
    pred_x, pred_y = [], []
    raw_x, raw_y = [], []
    
    com_pred = CoM_from_Patch_V12(3, 5).to(device)
    com_gt = CoM_from_18col_Patch_V15(3, 5).to(device)
    
    odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
    odd_single = [(8,4), (9,5)]; odd_map = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:5, 10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9}
    even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    even_single= [(0,0), (17,9)]; even_map = {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:5, 11:6, 12:6, 13:7, 14:7, 15:8, 16:8, 17:9}

    def compress(m18):
        m10 = np.zeros((32, 10), dtype=np.float32)
        for r in range(32):
            is_odd = (r%2==1)
            pairs = odd_pairs if is_odd else even_pairs
            singles = odd_single if is_odd else even_single
            mapping = odd_map if is_odd else even_map
            for c1, c2 in pairs:
                m10[r, mapping[c1]] = (m18[r, c1] + m18[r, c2])/2
            for c1, c10 in singles:
                m10[r, c10] = m18[r, c1]
        return m10

    for pid, rec in data.items():
        m18 = np.array(rec['merging']['normalized_matrix'], dtype=np.float32)
        t18 = np.array(rec['nonmerging']['normalized_matrix'], dtype=np.float32)
        
        m10 = compress(m18)
        pr_m, pc_m = np.unravel_index(np.argmax(m10), (32, 10))
        m10_t = torch.from_numpy(m10).to(device)
        m10_pad = F.pad(m10_t, (2,2,1,1)) 
        patch = m10_pad[pr_m:pr_m+3, pc_m:pc_m+5]
        patch = patch / (patch.sum() + 1e-8)
        
        inp = torch.cat([patch.unsqueeze(0), coord_grid], dim=0).unsqueeze(0) 
        is_odd = torch.tensor([float(pr_m%2==1)]).to(device)
        
        with torch.no_grad():
            out_patch = viz_model(inp, is_odd)
            raw_coords = com_pred(inp[:,0:1,:,:], is_odd, torch.tensor([pr_m]).to(device), torch.tensor([pc_m]).to(device))
            final_coords = com_pred(out_patch, is_odd, torch.tensor([pr_m]).to(device), torch.tensor([pc_m]).to(device))
            
            pr_gt, pc_gt = np.unravel_index(np.argmax(t18), (32, 18))
            t18_t = torch.from_numpy(t18).to(device)
            t18_pad = F.pad(t18_t, (2,2,1,1))
            gt_p = t18_pad[pr_gt:pr_gt+3, pc_gt:pc_gt+5].unsqueeze(0).unsqueeze(0)
            gt_coords = com_gt(gt_p, torch.tensor([pr_gt]).to(device), torch.tensor([pc_gt]).to(device))
            
        raw_x.append(raw_coords[0,0].item()); raw_y.append(raw_coords[0,1].item())
        pred_x.append(final_coords[0,0].item()); pred_y.append(final_coords[0,1].item())
        gt_x.append(gt_coords[0,0].item()); gt_y.append(gt_coords[0,1].item())
        
    plt.figure(figsize=(10, 10))
    
    def tr(x): return np.array(x) * 64 + 32
    
    plt.scatter(tr(gt_x), tr(gt_y), c='g', marker='*', s=100, label='GT')
    plt.scatter(tr(raw_x), tr(raw_y), c='r', marker='x', label='Raw')
    plt.scatter(tr(pred_x), tr(pred_y), c='b', marker='.', label='V17 Pred')
    
    for i in range(len(gt_x)):
        plt.plot([tr(raw_x)[i], tr(gt_x)[i]], [tr(raw_y)[i], tr(gt_y)[i]], 'r--', alpha=0.3)
        plt.plot([tr(pred_x)[i], tr(gt_x)[i]], [tr(pred_y)[i], tr(gt_y)[i]], 'b--', alpha=0.5)
        
    plt.legend()
    plt.title(f"V17 Validation (Residual): {os.path.basename(json_path)}")
    plt.grid(True)
    os.makedirs('/work/hwc/SPARSE/figs_val/v17', exist_ok=True)
    save_path = f'/work/hwc/SPARSE/figs_val/v17/{os.path.basename(json_path).replace(".json", ".png")}'
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

if os.path.exists(MODEL_SAVE_PATH_V17):
    val_file = '/work/hwc/SPARSE/training_data/validation_int/aligned_g26.json'
    visualize_v17(val_file, MODEL_SAVE_PATH_V17)