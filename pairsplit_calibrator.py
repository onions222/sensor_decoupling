# -*- coding: utf-8 -*-
"""
PairSplit Calibrator (X-only): robust centroid-x correction for shorted touch sensors
Author: ChatGPT
Date: 2025-10-20

Usage examples:
  python pairsplit_calibrator_xonly.py --data_dir /path/to/jsons --epochs 40 --batch_size 64 \
      --save runs/pairsplit_small_xonly.pth

  # eval-only (load weights then evaluate)
  python pairsplit_calibrator_xonly.py --data_dir /path/to/jsons --epochs 0 --eval_only 1 \
      --save runs/pairsplit_small_xonly.pth
"""

import os, json, random, argparse
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# ------------------------------
# Shorting prior (row-wise pairs)
# ------------------------------

def get_short_pairs_for_row(row: int, cols: int = 18):
    """Return (paired_indices, singletons) for a given row index (0-based)."""
    odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
    odd_single = [8,9]
    even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    even_single= [0,17]
    is_odd = (row % 2 == 1)
    return (odd_pairs, odd_single) if is_odd else (even_pairs, even_single)


def side_map_trinary(rows: int=32, cols: int=18) -> torch.Tensor:
    """(1,H,W) map: left=-1, right=+1, singleton=0"""
    side = np.zeros((rows, cols), dtype=np.float32)
    for i in range(rows):
        pairs, singles = get_short_pairs_for_row(i, cols)
        for l, r in pairs:
            side[i, l] = -1.0
            side[i, r] = +1.0
        for s in singles:
            side[i, s] = 0.0
    return torch.from_numpy(side).unsqueeze(0).contiguous().float()

# ------------------------------
# Data
# ------------------------------

class TouchJSONDataset(Dataset):
    """
    Load all JSON files in a directory. Each JSON has keys '0','1',...
    record['merging']['normalized_matrix'] and record['nonmerging']['normalized_matrix'] are 32x18 lists.
    We DO NOT renormalize matrices; we keep raw intensities and compute intensity-domain centroids.
    """
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.side = side_map_trinary(32, 18)  # (1,32,18)
        self.samples = self._load_all()

    def _load_all(self):
        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.json')]
        files.sort()
        buf = []
        for fname in files:
            path = os.path.join(self.data_dir, fname)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                for k, rec in obj.items():
                    m = np.array(rec['merging']['normalized_matrix'], dtype=np.float32)
                    t = np.array(rec['nonmerging']['normalized_matrix'], dtype=np.float32)
                    buf.append({'merging': m, 'target': t, 'info': {'file': fname, 'id': k}})
            except Exception as e:
                print(f"[WARN] skip {fname}: {e}")
        print(f"[INFO] loaded {len(buf)} samples from {self.data_dir}")
        return buf

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        m = torch.from_numpy(rec['merging']).clamp_min(0)
        t = torch.from_numpy(rec['target']).clamp_min(0)
        # keep raw intensities (no probability normalization)
        # centroid_x handles normalization internally via sum-weighted ratio
        return m.unsqueeze(0), t.unsqueeze(0), self.side, rec['info']  # (1,32,18), ...

# ------------------------------
# Metrics helpers
# ------------------------------

def centroid_x(img: torch.Tensor) -> torch.Tensor:
    """img: (B,1,H,W) -> (B,) intensity-domain x-centroid (sum-weighted)."""
    _,_,H,W = img.shape
    x = torch.linspace(0, W-1, W, device=img.device).view(1,1,1,W)
    num = (img * x).sum(dim=(-1,-2)).squeeze(1)
    den = (img).sum(dim=(-1,-2)).squeeze(1) + 1e-8
    return num / den

# ------------------------------
# Model: DeltaNet (tiny CNN)
# ------------------------------

class DeltaNet(nn.Module):
    """Tiny CNN predicting delta in [-1,1] via tanh."""
    def __init__(self, in_ch=1, base=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1, bias=True),
        )

    def forward(self, x):
        return torch.tanh(self.net(x))

# ------------------------------
# Physics-consistent split
# ------------------------------

def pairsplit_reconstruct(merging: torch.Tensor, delta_map: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
    """
    Vectorized reconstruction with per-pair symmetric split:
      for pair (l,r) with observed equal value m:
        x_l = m*(1+delta),  x_r = m*(1-delta)
      for singletons s: x_s = m_s
    """
    B, C, H, W = merging.shape
    assert C == 1
    side = side.to(merging.device)
    neg_mask = (side == -1).float()  # left
    pos_mask = (side == +1).float()  # right
    sing_mask= (side == 0).float()

    m = merging
    d = delta_map

    negm = neg_mask.expand(B,1,H,W)
    posm = pos_mask.expand(B,1,H,W)
    singm= sing_mask.expand(B,1,H,W)

    xl = m * (1 + d) * negm
    xr = m * (1 - d) * posm
    xs = m * singm

    return xl + xr + xs

# ------------------------------
# Regularizers
# ------------------------------

def tv_loss_2d(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).mean()
    dy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).mean()
    return dx + dy

# ------------------------------
# Train / Eval (X-only)
# ------------------------------

def train_epoch(model, loader, device, side, w):
    """X-only objective (no probabilities): ax*MAE_x + tv_w*TV + l2_w*||delta||^2 + hx*hinge_x"""
    model.train()
    ax, tv_w, l2_w, hx = w['alpha_x'], w['tv'], w['l2'], w['hinge_x']
    opt = w['opt']

    loss_agg = defaultdict(float)
    n_batches = 0

    for mp, tp, _, _ in loader:
        mp = mp.to(device)
        tp = tp.to(device)

        delta_map = model(mp)
        rec = pairsplit_reconstruct(mp, delta_map, side)

        cx_in, cx_gt, cx_pr = centroid_x(mp), centroid_x(tp), centroid_x(rec)
        mae_x = torch.abs(cx_pr - cx_gt).mean()
        tv    = tv_loss_2d(delta_map)
        l2    = (delta_map**2).mean()

        hinge_x = F.relu(torch.abs(cx_pr - cx_gt) - torch.abs(cx_in - cx_gt)).mean()
        loss = ax*mae_x + tv_w*tv + l2_w*l2 + hx*hinge_x

        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        loss_agg['total']   += float(loss.item())
        loss_agg['mae_x']   += float(mae_x.item())
        loss_agg['tv']      += float(tv.item())
        loss_agg['l2']      += float(l2.item())
        loss_agg['hinge_x'] += float(hinge_x.item())
        n_batches += 1

    for k in loss_agg:
        loss_agg[k] /= max(1, n_batches)
    return dict(loss_agg)


@torch.no_grad()
def evaluate(model, loader, device, side):
    model.eval()
    errs_x_b, errs_x_a = [], []
    improved = []

    for mp, tp, _, _ in loader:
        mp = mp.to(device); tp = tp.to(device)
        delta_map = model(mp)
        rec = pairsplit_reconstruct(mp, delta_map, side)

        cx_in, cx_gt, cx_pr = centroid_x(mp), centroid_x(tp), centroid_x(rec)
        exb = torch.abs(cx_in - cx_gt); exa = torch.abs(cx_pr - cx_gt)

        errs_x_b += exb.cpu().tolist(); errs_x_a += exa.cpu().tolist()
        improved += (exa < exb).cpu().tolist()

    def mean(x):
        return float(np.mean(x)) if len(x)>0 else float('nan')

    out = {
        'MAE_x_before'      : mean(errs_x_b),
        'MAE_x_after'       : mean(errs_x_a),
        'improved_ratio_x_%': 100.0 * mean(improved),
    }
    return out

# ------------------------------
# Runner
# ------------------------------

def run(data_dir: str,
        save_path: str = 'runs/pairsplit_small_xonly.pth',
        batch_size: int = 64,
        epochs: int = 60,
        lr: float = 3e-3,
        weight_decay: float = 1e-5,
        val_split: float = 0.2,
        seed: int = 42,
        eval_only: bool = False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device: {device}")

    ds_full = TouchJSONDataset(data_dir)
    if len(ds_full) == 0:
        raise RuntimeError(f"No JSON samples found in {data_dir}")

    val_size = int(len(ds_full) * val_split)
    tr_size  = len(ds_full) - val_size
    g = torch.Generator().manual_seed(seed)
    idxs = range(len(ds_full))
    tr_idx, va_idx = random_split(idxs, [tr_size, val_size], generator=g)
    ds_tr = torch.utils.data.Subset(ds_full, tr_idx.indices)
    ds_va = torch.utils.data.Subset(ds_full, va_idx.indices)

    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=0)

    model = DeltaNet(in_ch=1, base=8).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    side = side_map_trinary(32, 18).to(device)

    weights = dict(alpha_x=1.0, tv=1e-2, l2=1e-3, hinge_x=0.8, opt=opt)

    if not eval_only and epochs>0:
        best = float('inf')
        for ep in range(1, epochs+1):
            tr = train_epoch(model, dl_tr, device, side, weights)
            va = evaluate(model, dl_va, device, side)
            after = va['MAE_x_after']; before = va['MAE_x_before']
            impr  = (before - after) / (before + 1e-12)
            print(f"[Ep {ep:03d}] loss={tr['total']:.6f}  MAE_x: {before:.4f} -> {after:.4f} (↑{impr*100:.2f}%),  "
                  f"improved_x={va['improved_ratio_x_%']:.1f}%")
            if after < best:
                best = after
                os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
                torch.save(model.state_dict(), save_path)

    # final eval on validation split
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    va = evaluate(model, dl_va, device, side)

    print("=== Final Evaluation (validation split) ===")
    for k,v in va.items():
        print(f"{k:>18s}: {v:.6f}")

    return va


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/work/hwc/SENSOR_STAGE3/training_data/aligned_data_for_training_int')
    parser.add_argument('--save', type=str, default='runs/pairsplit_small_xonly.pth')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--eval_only', type=int, default=0)
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        save_path=args.save,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_only=bool(args.eval_only)
    )


if __name__ == "__main__":
    main()