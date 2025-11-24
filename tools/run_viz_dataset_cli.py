#!/usr/bin/env python3
"""
CLI wrapper for running dataset visualization.
Supports: --data-dirs (one or more), --model-path, --out-root, --top-k, --workers

This script spins up worker processes; each worker loads the model once.
"""
import os
import argparse
import multiprocessing as mp
import traceback

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dirs', nargs='+', required=True, help='Directories containing JSON files to process')
    p.add_argument('--model-path', default='/work/hwc/SPARSE/distill/decoupler_model_v16_teacher_best.pth')
    p.add_argument('--out-root', default='/work/hwc/SPARSE/figs_val/v16_dataset')
    p.add_argument('--patch-size', nargs=2, type=int, default=[3,5])
    p.add_argument('--top-k', type=int, default=5)
    p.add_argument('--workers', type=int, default=1, help='Number of parallel worker processes (>=1)')
    return p.parse_args()

# Worker initializer: load model and helper objects inside each worker process
def worker_init(model_path, device_str, patch_h, patch_w):
    global WORKER_STATE
    try:
        import torch, torch.nn.functional as F
        import torch.nn as nn
        import numpy as np
        # re-define minimal model classes (copied from notebook file)
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
        device = torch.device(device_str)
        model = TeacherModel().to(device)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
            model.load_state_dict(state['state_dict'])
        else:
            model.load_state_dict(state)
        model.eval()
        com10 = CoM_from_Patch_V12(patch_h=patch_h, patch_w=patch_w).to(device)
        com18 = None
        WORKER_STATE = {'model': model, 'com10': com10, 'com18': com18, 'device': device}
    except Exception:
        traceback.print_exc()
        raise

# Worker processing function: runs on a worker process
def worker_process(args_tuple):
    try:
        import torch
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from torch.nn import functional as F
        model = WORKER_STATE['model']
        com10 = WORKER_STATE['com10']
        device = WORKER_STATE['device']
        json_path, out_dir, top_k, patch_size = args_tuple
        os.makedirs(out_dir, exist_ok=True)
        with open(json_path, 'r', encoding='utf-8') as f:
            aligned = json.load(f)
        raw_list=[]; clean_list=[]; is_odd_list=[]; peak18=[]; peak10=[]; pids=[]
        pad_ph, pad_pw = patch_size[0]//2, patch_size[1]//2
        pad2d_18_col = (pad_pw, pad_pw, pad_ph, pad_ph)
        pad2d_10_col = (pad_pw, pad_pw, pad_ph, pad_ph)
        eps=1e-8
        for pid, pair in aligned.items():
            merging_18 = np.array(pair['merging']['normalized_matrix'], dtype=np.float32)
            target_18 = np.array(pair['nonmerging']['normalized_matrix'], dtype=np.float32)
            if merging_18.shape != (32,18):
                continue
            peak_r_gt, peak_c_gt_18 = np.unravel_index(np.argmax(target_18),(32,18))
            # get clean patch
            import torch as _torch
            target_t_18 = F.pad(_torch.from_numpy(target_18), pad2d_18_col)
            r0_t, r1_t = peak_r_gt, peak_r_gt + patch_size[0]
            c0_t, c1_t = peak_c_gt_18, peak_c_gt_18 + patch_size[1]
            clean_patch = target_t_18[r0_t:r1_t, c0_t:c1_t]
            # compress merging
            def compress_18_to_10(matrix_18):
                m10 = np.zeros((32,10), dtype=np.float32)
                # mapping copied
                odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
                odd_map = {0:0,1:0,2:1,3:1,4:2,5:2,6:3,7:3,8:4,9:5,10:6,11:6,12:7,13:7,14:8,15:8,16:9,17:9}
                even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
                even_map = {0:0,1:1,2:1,3:2,4:2,5:3,6:3,7:4,8:4,9:5,10:5,11:6,12:6,13:7,14:7,15:8,16:8,17:9}
                for r in range(32):
                    is_odd = (r%2==1)
                    pairs = odd_pairs if is_odd else even_pairs
                    map18 = odd_map if is_odd else even_map
                    for c_left,c_right in pairs:
                        idx = map18[c_left]
                        m10[r,idx] = (matrix_18[r,c_left]+matrix_18[r,c_right])/2.0
                    singles = [(8,4),(9,5)] if is_odd else [(0,0),(17,9)]
                    for c18, idx in singles:
                        m10[r, idx] = matrix_18[r,c18]
                return m10
            effective_merging_10 = compress_18_to_10(merging_18)
            peak_r_m, peak_c_m_10 = np.unravel_index(np.argmax(effective_merging_10),(32,10))
            is_odd = float(peak_r_m % 2 == 1)
            merging_t_10 = F.pad(_torch.from_numpy(effective_merging_10), pad2d_10_col)
            r0_m, r1_m = peak_r_m, peak_r_m + patch_size[0]
            c0_m, c1_m = peak_c_m_10, peak_c_m_10 + patch_size[1]
            merging_patch = merging_t_10[r0_m:r1_m, c0_m:c1_m]
            merging_patch = merging_patch.clamp_min(0); merging_patch = merging_patch / (merging_patch.sum() + eps)
            clean_patch = clean_patch.clamp_min(0); clean_patch = clean_patch / (clean_patch.sum() + eps)
            raw_list.append(merging_patch)
            clean_list.append(clean_patch)
            is_odd_list.append(is_odd)
            peak18.append((peak_r_gt, peak_c_gt_18))
            peak10.append((peak_r_m, peak_c_m_10))
            pids.append(str(pid))
        if len(raw_list)==0:
            return (json_path, None, 'no_samples')
        raw_tensor = torch.stack(raw_list).unsqueeze(1).to(device)
        clean_tensor = torch.stack(clean_list).unsqueeze(1).to(device)
        is_odd_tensor = torch.tensor(is_odd_list, dtype=torch.float32).to(device)
        peak18_t = torch.tensor(peak18, dtype=torch.float32).to(device)
        peak10_t = torch.tensor(peak10, dtype=torch.float32).to(device)
        with torch.no_grad():
            pred_patches = model(raw_tensor, is_odd_tensor)
            pred_global = com10(pred_patches, is_odd_tensor, peak10_t[:,0], peak10_t[:,1])
            clean_global = None
            # compute clean global using local com from 18col (approx via com10 structure not needed here)
            # but we have no com18 implemented in worker; instead we reuse com10 call on clean_tensor with fake flags
            # to produce comparable coordinate space
            clean_global = com10(clean_tensor, is_odd_tensor, peak18_t[:,0], peak18_t[:,1])
        pred_np = pred_global.cpu().numpy(); clean_np = clean_global.cpu().numpy()
        def transform(coords):
            x = coords[:,0]*64.0 + 32.0
            y = coords[:,1]*64.0 + 32.0
            return x,y
        pred_x, pred_y = transform(pred_np)
        clean_x, clean_y = transform(clean_np)
        import numpy as _np
        errors = _np.sqrt((pred_x-clean_x)**2 + (pred_y-clean_y)**2)
        worst_idx = _np.argsort(errors)[::-1][:top_k]
        # plot and save
        import matplotlib.pyplot as _plt
        _plt.figure(figsize=(10,10))
        _plt.scatter(clean_x, clean_y, marker='*', s=120, c='lime', edgecolors='black', label='GT', zorder=5)
        _plt.scatter(pred_x, pred_y, marker='o', s=60, c='blue', alpha=0.8, label='Pred', zorder=3)
        for i in range(len(clean_x)):
            _plt.plot([pred_x[i], clean_x[i]],[pred_y[i], clean_y[i]], 'b--', linewidth=0.4, alpha=0.5)
        bad_lines=[]
        for idx in worst_idx:
            pid = pids[idx]
            err = float(errors[idx])
            _plt.scatter(pred_x[idx], pred_y[idx], s=200, facecolors='none', edgecolors='magenta', linewidths=2, zorder=6)
            _plt.text(pred_x[idx]+2.0, pred_y[idx]+2.0, f"{pid}:{err:.1f}", color='magenta', fontsize=9, zorder=7)
            bad_lines.append((pid, err))
        _plt.title(os.path.basename(json_path))
        _plt.legend(); _plt.grid(True, linestyle=':', alpha=0.6); _plt.axis('equal')
        out_png = os.path.join(out_dir, os.path.basename(json_path).replace('.json', '.png'))
        _plt.savefig(out_png, dpi=150, bbox_inches='tight')
        _plt.close()
        out_log = out_png.replace('.png', '_bad_points.txt')
        with open(out_log,'w',encoding='utf-8') as wf:
            wf.write(f"json_file,{os.path.basename(json_path)}\n")
            wf.write('point_id,error_pixels\n')
            for pid,err in bad_lines:
                wf.write(f"{pid},{err:.4f}\n")
        return (json_path, out_png, out_log)
    except Exception as e:
        return (json_path, None, str(e))

# Main: create job list and run with pool
def main():
    args = parse_args()
    jobs = []
    for data_dir in args.data_dirs:
        if not os.path.isdir(data_dir):
            print('Missing dir, skipping', data_dir)
            continue
        jsons = sorted([os.path.join(data_dir,f) for f in os.listdir(data_dir) if f.lower().endswith('.json')])
        out_dir = os.path.join(args.out_root, os.path.basename(data_dir.rstrip('/')))
        for j in jsons:
            jobs.append((j, out_dir, args.top_k, tuple(args.patch_size)))
    if len(jobs)==0:
        print('No json jobs to run'); return
    # run with pool
    workers = max(1, args.workers)
    if workers==1:
        # run sequentially, load model in main process
        worker_init(args.model_path, 'cuda' if torch.cuda.is_available() else 'cpu', args.patch_size[0], args.patch_size[1])
        results = [worker_process(job) for job in jobs]
    else:
        # start pool with initializer that loads the model per process
        ctx = mp.get_context('spawn')
        pool = ctx.Pool(processes=workers, initializer=worker_init, initargs=(args.model_path, 'cuda' if torch.cuda.is_available() else 'cpu', args.patch_size[0], args.patch_size[1]))
        results = pool.map(worker_process, jobs)
        pool.close(); pool.join()
    # report
    success=0; fail=0
    for r in results:
        if r[1] is not None:
            success+=1
        else:
            fail+=1
            print('Failed:', r[0], r[2])
    print('Done. success=', success, 'failed=', fail)

if __name__=='__main__':
    main()
