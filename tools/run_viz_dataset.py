#!/usr/bin/env python3
"""
Run visualization (PNG + bad_points logs) over entire dataset directories.
Default directories processed:
 - /work/hwc/SPARSE/training_data/aligned_data_for_training_int
 - /work/hwc/SPARSE/training_data/validation_int

Outputs saved under: /work/hwc/SPARSE/figs_val/v16_dataset/<dir_basename>/
A summary CSV is produced at: /work/hwc/SPARSE/figs_val/v16_dataset/bad_points_summary_dataset.csv

This script instantiates the TeacherModel, loads weights, runs inference per JSON file,
computes pred vs GT errors, saves PNG and *_bad_points.txt, and aggregates results.
"""
import os
import json
import glob
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/work/hwc/SPARSE'
DATA_DIRS = [
    os.path.join(ROOT, 'training_data', 'aligned_data_for_training_int'),
    os.path.join(ROOT, 'training_data', 'validation_int')
]
MODEL_PATH = os.path.join(ROOT, 'distill', 'decoupler_model_v16_teacher_best.pth')
OUT_ROOT = os.path.join(ROOT, 'figs_val', 'v16_dataset')
VIZ_PATCH_SIZE = (3,5)
TOP_K = 3
os.makedirs(OUT_ROOT, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)

# Minimal model/CoM definitions (same as in notebook)
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

# Model (must match saved weights)
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

# mapping helpers (same as notebook)
odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
odd_single = [(8,4), (9,5)]
odd_map_18_to_10 = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:5, 10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9}

even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
even_single = [(0,0), (17,9)]
even_map_18_to_10 = {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:5, 11:6, 12:6, 13:7, 14:7, 15:8, 16:8, 17:9}

def compress_18_to_10(matrix_18):
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

# loader per JSON

def load_data_for_viz_v16(json_path, patch_size, device):
    raw_patch_list, clean_patch_list, is_odd_list = [], [], []
    peak_coords_18_gt_list, peak_coords_10_merging_list = [], []
    point_id_list = []

    pad_ph, pad_pw = patch_size[0]//2, patch_size[1]//2
    pad2d_18_col = (pad_pw, pad_pw, pad_ph, pad_ph)
    pad2d_10_col = (pad_pw, pad_pw, pad_ph, pad_ph)
    eps = 1e-8

    with open(json_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)
    for point_id, pair_data in aligned_data.items():
        merging_18 = np.array(pair_data['merging']['normalized_matrix'], dtype=np.float32)
        target_18 = np.array(pair_data['nonmerging']['normalized_matrix'], dtype=np.float32)
        if merging_18.shape != (32,18):
            continue
        # GT
        peak_r_gt, peak_c_gt_18 = np.unravel_index(np.argmax(target_18), (32,18))
        target_t_18 = F.pad(torch.from_numpy(target_18), pad2d_18_col)
        r0_t, r1_t = peak_r_gt, peak_r_gt + patch_size[0]
        c0_t, c1_t = peak_c_gt_18, peak_c_gt_18 + patch_size[1]
        clean_patch = target_t_18[r0_t:r1_t, c0_t:c1_t]

        # Pred pipeline: compress merging 18->10
        effective_merging_10 = compress_18_to_10(merging_18)
        peak_r_m, peak_c_m_10 = np.unravel_index(np.argmax(effective_merging_10), (32,10))
        is_odd = float(peak_r_m % 2 == 1)
        merging_t_10 = F.pad(torch.from_numpy(effective_merging_10), pad2d_10_col)
        r0_m, r1_m = peak_r_m, peak_r_m + patch_size[0]
        c0_m, c1_m = peak_c_m_10, peak_c_m_10 + patch_size[1]
        merging_patch = merging_t_10[r0_m:r1_m, c0_m:c1_m]

        merging_patch = merging_patch.clamp_min(0); merging_patch = merging_patch / (merging_patch.sum() + eps)
        clean_patch = clean_patch.clamp_min(0); clean_patch = clean_patch / (clean_patch.sum() + eps)

        raw_patch_list.append(merging_patch)
        clean_patch_list.append(clean_patch)
        is_odd_list.append(is_odd)
        peak_coords_18_gt_list.append([peak_r_gt, peak_c_gt_18])
        peak_coords_10_merging_list.append([peak_r_m, peak_c_m_10])
        point_id_list.append(str(point_id))

    if len(raw_patch_list) == 0:
        return [None]*6
    raw_tensor = torch.stack(raw_patch_list).unsqueeze(1).to(device)
    clean_tensor = torch.stack(clean_patch_list).unsqueeze(1).to(device)
    is_odd_tensor = torch.tensor(is_odd_list, dtype=torch.float32).to(device)
    peak_18_gt_tensor = torch.tensor(peak_coords_18_gt_list, dtype=torch.float32).to(device)
    peak_10_merging_tensor = torch.tensor(peak_coords_10_merging_list, dtype=torch.float32).to(device)
    return raw_tensor, clean_tensor, is_odd_tensor, peak_18_gt_tensor, peak_10_merging_tensor, point_id_list

# instantiate model
model = TeacherModel().to(device)
state = torch.load(MODEL_PATH, map_location=device)
if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
    model.load_state_dict(state['state_dict'])
else:
    model.load_state_dict(state)
model.eval()

com_calc_from_10col_patch = CoM_from_Patch_V12(patch_h=VIZ_PATCH_SIZE[0], patch_w=VIZ_PATCH_SIZE[1]).to(device)
com_calc_from_18col_patch = CoM_from_18col_Patch_V15(patch_h=VIZ_PATCH_SIZE[0], patch_w=VIZ_PATCH_SIZE[1]).to(device)

all_rows = []

for data_dir in DATA_DIRS:
    if not os.path.isdir(data_dir):
        print('Missing directory, skipping:', data_dir)
        continue
    json_files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.json')])
    out_dir = os.path.join(OUT_ROOT, os.path.basename(data_dir.rstrip('/')))
    os.makedirs(out_dir, exist_ok=True)
    print('Processing directory:', data_dir, 'found', len(json_files), 'json files')
    for json_path in json_files:
        try:
            result = load_data_for_viz_v16(json_path, VIZ_PATCH_SIZE, device)
            if result[0] is None:
                print('No samples in', json_path, 'skipping')
                continue
            raw_data, clean_data, is_odd_data, peak_18_gt_data, peak_10_merging_data, point_id_list = result
            with torch.no_grad():
                pred_patches = model(raw_data, is_odd_data)
                pred_global_coords = com_calc_from_10col_patch(pred_patches, is_odd_data, peak_10_merging_data[:,0], peak_10_merging_data[:,1])
                clean_global_coords = com_calc_from_18col_patch(clean_data, peak_18_gt_data[:,0], peak_18_gt_data[:,1])
            raw_global_coords = com_calc_from_10col_patch(raw_data, is_odd_data, peak_10_merging_data[:,0], peak_10_merging_data[:,1])

            raw_coords_np = raw_global_coords.cpu().numpy(); clean_coords_np = clean_global_coords.cpu().numpy(); pred_coords_np = pred_global_coords.cpu().numpy()
            def transform_coords(coords_np):
                x = coords_np[:,0]*64.0 + 32.0
                y = coords_np[:,1]*64.0 + 32.0
                return x,y
            raw_x_viz, raw_y_viz = transform_coords(raw_coords_np)
            clean_x_viz, clean_y_viz = transform_coords(clean_coords_np)
            pred_x_viz, pred_y_viz = transform_coords(pred_coords_np)

            errors = np.sqrt((pred_x_viz - clean_x_viz)**2 + (pred_y_viz - clean_y_viz)**2)
            worst_idx = np.argsort(errors)[::-1][:TOP_K]

            plt.figure(figsize=(10,10))
            plt.scatter(clean_x_viz, clean_y_viz, marker='*', s=120, c='lime', edgecolors='black', label='GT', zorder=5)
            plt.scatter(raw_x_viz, raw_y_viz, marker='x', s=60, c='red', label='Raw', zorder=4)
            plt.scatter(pred_x_viz, pred_y_viz, marker='o', s=60, c='blue', alpha=0.8, label='Pred', zorder=3)
            for i in range(len(clean_x_viz)):
                plt.plot([raw_x_viz[i], clean_x_viz[i]],[raw_y_viz[i], clean_y_viz[i]], 'r--', linewidth=0.4, alpha=0.5)
                plt.plot([pred_x_viz[i], clean_x_viz[i]],[pred_y_viz[i], clean_y_viz[i]], 'b--', linewidth=0.4, alpha=0.5)
            bad_lines = []
            for idx in worst_idx:
                pid = point_id_list[idx] if idx < len(point_id_list) else str(idx)
                err_val = errors[idx]
                plt.scatter(pred_x_viz[idx], pred_y_viz[idx], s=200, facecolors='none', edgecolors='magenta', linewidths=2, zorder=6)
                plt.text(pred_x_viz[idx] + 2.0, pred_y_viz[idx] + 2.0, f"{pid}:{err_val:.1f}", color='magenta', fontsize=9, zorder=7)
                bad_lines.append((pid, float(err_val)))
            plt.title(f"V16 (Teacher) global coordinate (from {os.path.basename(json_path)})")
            plt.xlabel('X (transformed)'); plt.ylabel('Y (transformed)'); plt.legend(); plt.grid(True, linestyle=':', alpha=0.6); plt.axis('equal')
            out_png = os.path.join(out_dir, os.path.basename(json_path).replace('.json', '.png'))
            plt.savefig(out_png, dpi=150, bbox_inches='tight')
            plt.close()

            out_log = out_png.replace('.png', '_bad_points.txt')
            with open(out_log, 'w', encoding='utf-8') as wf:
                wf.write(f"json_file,{os.path.basename(json_path)}\n")
                wf.write('point_id,error_pixels\n')
                for pid, err in bad_lines:
                    wf.write(f"{pid},{err:.4f}\n")
            # append to summary rows
            for pid, err in bad_lines:
                all_rows.append({'json_file': os.path.basename(json_path), 'point_id': pid, 'error_pixels': err, 'source_log': os.path.basename(out_log)})
            print('Saved:', out_png, 'log:', out_log)
        except Exception as e:
            print('Error processing', json_path, e)

# write aggregated CSV
out_csv = os.path.join(OUT_ROOT, 'bad_points_summary_dataset.csv')
with open(out_csv, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['json_file','point_id','error_pixels','source_log']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for r in all_rows:
        writer.writerow(r)
print('Wrote aggregated CSV:', out_csv)
print('Processed total entries:', len(all_rows))
