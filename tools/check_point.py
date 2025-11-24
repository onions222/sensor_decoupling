#!/usr/bin/env python3
"""Check a single JSON file and single point with the V16 teacher model.

Usage example:
  python3 tools/check_point.py --json training_data/aligned_data_for_training_int/aligned_g24.json \
    --point-id 34 --model-path distill/decoupler_model_v16_teacher_best.pth \
    --out-dir figs_val/check_point --device cuda

The script will save a PNG highlighting the chosen point and a small text log
with predicted coords, GT coords and pixel error.
"""
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def compress_18_to_10(matrix_18):
    odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
    odd_single = [(8,4), (9,5)]
    odd_map_18_to_10 = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:5, 10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9}
    even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    even_single = [(0,0), (17,9)]
    even_map_18_to_10 = {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:5, 11:6, 12:6, 13:7, 14:7, 15:8, 16:8, 17:9}

    matrix_10 = np.zeros((32,10), dtype=np.float32)
    for r in range(32):
        is_odd = (r % 2 == 1)
        pairs = odd_pairs if is_odd else even_pairs
        singles = odd_single if is_odd else even_single
        map_18_to_10 = odd_map_18_to_10 if is_odd else even_map_18_to_10
        for (c_left, c_right) in pairs:
            col_10_idx = map_18_to_10[c_left]
            avg = (matrix_18[r, c_left] + matrix_18[r, c_right]) / 2.0
            matrix_10[r, col_10_idx] = avg
        for (c_18, col_10_idx) in singles:
            matrix_10[r, col_10_idx] = matrix_18[r, c_18]
    return matrix_10


class DifferentiableCoM_Patch_3x5(nn.Module):
    def __init__(self, height=3, width=5):
        super().__init__()
        y = torch.linspace(0, height - 1, height)
        x = torch.linspace(0, width - 1, width)
        x_grid = x.view(1, -1).repeat(height, 1)
        y_grid = y.view(-1, 1).repeat(1, width)
        self.register_buffer('x_grid_buf', x_grid)
        self.register_buffer('y_grid_buf', y_grid)
    def forward(self, image):
        img = image.squeeze(1)
        eps = 1e-8
        mass = torch.sum(img, dim=(1,2)) + eps
        x_weighted = torch.sum(img * self.x_grid_buf, dim=(1,2))
        y_weighted = torch.sum(img * self.y_grid_buf, dim=(1,2))
        center_x = x_weighted / mass
        center_y = y_weighted / mass
        return torch.stack([center_x, center_y], dim=1)


class CoM_from_Patch_V12(nn.Module):
    def __init__(self, patch_h=3, patch_w=5):
        super().__init__()
        self.local_com_calc = DifferentiableCoM_Patch_3x5(patch_h, patch_w)
        self.ph_offset = patch_h // 2
        self.pw_offset_10col = patch_w // 2
        odd_grid_10_np = np.array([0.5, 2.5, 4.5, 6.5, 8.0, 9.0, 10.5, 12.5, 14.5, 16.5], dtype=np.float32)
        even_grid_10_np = np.array([0.0, 1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.0], dtype=np.float32)
        self.register_buffer('odd_grid_10', torch.from_numpy(odd_grid_10_np))
        self.register_buffer('even_grid_10', torch.from_numpy(even_grid_10_np))
    def forward(self, patches, is_odd_flags, peak_rs, peak_cs_10_col):
        local_coords_3x5 = self.local_com_calc(patches)
        local_x_10col_scalar = local_coords_3x5[:,0]
        local_y_scalar = local_coords_3x5[:,1]
        global_x_10col_scalar = local_x_10col_scalar + peak_cs_10_col - self.pw_offset_10col
        global_y_scalar = local_y_scalar + peak_rs - self.ph_offset
        x_clamped = torch.clamp(global_x_10col_scalar, 0, 9)
        x_floor = torch.floor(x_clamped).long()
        x_ceil = torch.ceil(x_clamped).long()
        x_floor = torch.clamp(x_floor, 0, 9)
        x_ceil = torch.clamp(x_ceil, 0, 9)
        frac = x_clamped - x_floor.float()
        grids = torch.stack([self.even_grid_10, self.odd_grid_10], dim=0)
        selected_grids = grids[is_odd_flags.long()]
        val_floor = torch.gather(selected_grids, 1, x_floor.unsqueeze(-1)).squeeze(-1)
        val_ceil = torch.gather(selected_grids, 1, x_ceil.unsqueeze(-1)).squeeze(-1)
        global_x_18col_scalar = val_floor + (val_ceil - val_floor) * frac
        return torch.stack([global_x_18col_scalar, global_y_scalar], dim=1)


class CoM_from_18col_Patch_V15(nn.Module):
    def __init__(self, patch_h=3, patch_w=5):
        super().__init__()
        self.local_com_calc = DifferentiableCoM_Patch_3x5(patch_h, patch_w)
        self.ph_offset = patch_h // 2
        self.pw_offset_18col = patch_w // 2
    def forward(self, patches, peak_rs, peak_cs_18_col):
        local_coords_3x5 = self.local_com_calc(patches)
        local_x_scalar = local_coords_3x5[:,0]
        local_y_scalar = local_coords_3x5[:,1]
        global_x_18col_scalar = local_x_scalar + peak_cs_18_col - self.pw_offset_18col
        global_y_scalar = local_y_scalar + peak_rs - self.ph_offset
        return torch.stack([global_x_18col_scalar, global_y_scalar], dim=1)


class _PatchNet_V13_Large_Teacher(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, 1)
        )
    def forward(self, x):
        return self.net(x)


class TeacherModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.odd_net = _PatchNet_V13_Large_Teacher()
        self.even_net = _PatchNet_V13_Large_Teacher()
    def forward(self, x_patches, is_odd_flags):
        out_odd = self.odd_net(x_patches)
        out_even = self.even_net(x_patches)
        is_odd_mask = is_odd_flags.view(-1,1,1,1) > 0.5
        out = torch.where(is_odd_mask, out_odd, out_even)
        return torch.relu(out)


def extract_point_data(json_path, point_id, patch_size=(3,5)):
    pad_ph, pad_pw = patch_size[0]//2, patch_size[1]//2
    pad2d_18_col = (pad_pw, pad_pw, pad_ph, pad_ph)
    pad2d_10_col = (pad_pw, pad_pw, pad_ph, pad_ph)
    eps = 1e-8

    with open(json_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)
    key = str(point_id)
    if key not in aligned_data:
        raise KeyError(f"point_id {point_id} not found in {json_path}")
    pair_data = aligned_data[key]
    merging_18 = np.array(pair_data['merging']['normalized_matrix'], dtype=np.float32)
    target_18 = np.array(pair_data['nonmerging']['normalized_matrix'], dtype=np.float32)
    if merging_18.shape != (32,18) or target_18.shape != (32,18):
        raise ValueError('unexpected matrix shape')

    # GT peak from nonmerging (18-col)
    peak_r_gt, peak_c_gt_18 = np.unravel_index(np.argmax(target_18), (32,18))
    target_t_18 = F.pad(torch.from_numpy(target_18), pad2d_18_col)
    r0_t, r1_t = peak_r_gt, peak_r_gt + patch_size[0]
    c0_t, c1_t = peak_c_gt_18, peak_c_gt_18 + patch_size[1]
    clean_patch = target_t_18[r0_t:r1_t, c0_t:c1_t]

    # Pred pipeline: compress merging 18->10 and find merging peak
    effective_merging_10 = compress_18_to_10(merging_18)
    peak_r_m, peak_c_m_10 = np.unravel_index(np.argmax(effective_merging_10), (32,10))
    is_odd = float(peak_r_m % 2 == 1)
    merging_t_10 = F.pad(torch.from_numpy(effective_merging_10), pad2d_10_col)
    r0_m, r1_m = peak_r_m, peak_r_m + patch_size[0]
    c0_m, c1_m = peak_c_m_10, peak_c_m_10 + patch_size[1]
    merging_patch = merging_t_10[r0_m:r1_m, c0_m:c1_m]

    merging_patch = merging_patch.clamp_min(0); merging_patch = merging_patch / (merging_patch.sum() + eps)
    clean_patch = clean_patch.clamp_min(0); clean_patch = clean_patch / (clean_patch.sum() + eps)

    return {
        'merging_patch': merging_patch.unsqueeze(0).unsqueeze(0),
        'clean_patch': clean_patch.unsqueeze(0).unsqueeze(0),
        'is_odd': torch.tensor([is_odd], dtype=torch.float32),
        'peak_18_gt': torch.tensor([[float(peak_r_gt), float(peak_c_gt_18)]], dtype=torch.float32),
        'peak_10_merging': torch.tensor([[float(peak_r_m), float(peak_c_m_10)]], dtype=torch.float32),
    }


def transform_coords(coords_np):
    x = coords_np[0]*64.0 + 32.0
    y = coords_np[1]*64.0 + 32.0
    return float(x), float(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', default='/work/hwc/SPARSE/training_data/aligned_data_for_training_int/aligned_g24.json')
    parser.add_argument('--point-id', default=20, type=int)
    parser.add_argument('--model-path', default=os.path.join('/work/hwc/SPARSE','distill','decoupler_model_v16_teacher_best.pth'))
    parser.add_argument('--out-dir', default=os.path.join('/work/hwc/SPARSE','figs_val','check_point'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # load model
    model = TeacherModel().to(device)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
        model.load_state_dict(state['state_dict'])
    else:
        model.load_state_dict(state)
    model.eval()

    # coM calculators
    com_calc_from_10col_patch = CoM_from_Patch_V12(patch_h=3, patch_w=5).to(device)
    com_calc_from_18col_patch = CoM_from_18col_Patch_V15(patch_h=3, patch_w=5).to(device)

    # extract data for the target point
    d = extract_point_data(args.json, args.point_id, patch_size=(3,5))
    merging_patch = d['merging_patch'].to(device)
    clean_patch = d['clean_patch'].to(device)
    is_odd = d['is_odd'].to(device)
    peak_18_gt = d['peak_18_gt'].to(device)
    peak_10_merging = d['peak_10_merging'].to(device)

    with torch.no_grad():
        pred_patch = model(merging_patch, is_odd)
        pred_global = com_calc_from_10col_patch(pred_patch, is_odd, peak_10_merging[:,0], peak_10_merging[:,1])
        clean_global = com_calc_from_18col_patch(clean_patch, peak_18_gt[:,0], peak_18_gt[:,1])
        raw_global = com_calc_from_10col_patch(merging_patch, is_odd, peak_10_merging[:,0], peak_10_merging[:,1])

    pred_np = pred_global.cpu().numpy()[0]
    clean_np = clean_global.cpu().numpy()[0]
    raw_np = raw_global.cpu().numpy()[0]
    pred_x, pred_y = transform_coords(pred_np)
    clean_x, clean_y = transform_coords(clean_np)
    raw_x, raw_y = transform_coords(raw_np)
    err = float(np.sqrt((pred_x - clean_x)**2 + (pred_y - clean_y)**2))

    base = os.path.splitext(os.path.basename(args.json))[0]
    out_png = os.path.join(args.out_dir, f"{base}_pt{args.point_id}_check.png")
    out_log = os.path.join(args.out_dir, f"{base}_pt{args.point_id}_check.txt")

    # simple viz: overlay raw, gt, pred (single point)
    plt.figure(figsize=(6,6))
    plt.scatter([clean_x],[clean_y], marker='*', s=200, c='lime', edgecolors='black', label='GT')
    plt.scatter([raw_x],[raw_y], marker='x', s=120, c='red', label='Raw (from merging)')
    plt.scatter([pred_x],[pred_y], marker='o', s=120, c='blue', label='Pred')
    plt.plot([raw_x, clean_x],[raw_y, clean_y], 'r--', linewidth=1.0, alpha=0.7)
    plt.plot([pred_x, clean_x],[pred_y, clean_y], 'b--', linewidth=1.0, alpha=0.7)
    plt.text(pred_x + 4.0, pred_y + 4.0, f"err={err:.2f}px", color='magenta', fontsize=12)
    plt.title(f"Check point {args.point_id} in {base}")
    plt.xlabel('X'); plt.ylabel('Y'); plt.legend(); plt.grid(True, linestyle=':', alpha=0.6); plt.axis('equal')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()

    with open(out_log, 'w', encoding='utf-8') as wf:
        wf.write(f"json_file,{os.path.basename(args.json)}\n")
        wf.write(f"point_id,{args.point_id}\n")
        wf.write(f"pred_x,pred_y,{pred_x:.6f},{pred_y:.6f}\n")
        wf.write(f"gt_x,gt_y,{clean_x:.6f},{clean_y:.6f}\n")
        wf.write(f"raw_x,raw_y,{raw_x:.6f},{raw_y:.6f}\n")
        wf.write(f"error_pixels,{err:.6f}\n")

    print('Saved png:', out_png)
    print('Saved log:', out_log)


if __name__ == '__main__':
    main()
