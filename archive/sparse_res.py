# -*- coding: utf-8 -*-
import os, re, json, math, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

# (新增) 导入量化相关的包
import torch.nn.quantized as nnq
import torch.nn.intrinsic.quantized as nniq
import torch.ao.quantization as tq

from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# =========================
# 路径与全局配置
# =========================
TRAIN_DIR = "training_data/aligned_data_for_training_int"
VAL_DIR   = "training_data/validation_int"

OUT_DIR   = Path("runs_sparse_v2_3x6_qat_minmax") # (修改) 新的输出目录, 标明 MinMax
MODEL_FLOAT_PATH = OUT_DIR / "best_qat_float.pt"
MODEL_INT8_PATH  = OUT_DIR / "best_qat_int8.pt"
QP_JSON_PATH     = OUT_DIR / "offset_qparams.json"

# (不变) 3x6 窗口, K=8 稀疏点
LOCAL_PATCH_H = 3
LOCAL_PATCH_W = 6
K_SPARSE_POINTS = 8

LEARNING_RATE = 1e-3

BATCH_TR, BATCH_VA = 64, 256
EPOCHS             = 50
SEED               = 0
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WIDTH              = 8

# (不变) Y 轴处理
MAX_DY_ABS = 0.25
LOSS_W_X   = 6.0
LOSS_W_Y   = 0.0
ALPHA_Y    = 0.0

# (不变) 风险厌恶
TAIL_ALPHA = 0.30
CVAR_WARMUP_EPOCHS = 5
EARLY_STOP_PATIENCE = 10

# (不变) 激活 QAT
USE_QAT   = True
QENGINE   = "fbgemm"

# =========================
# 工具与先验 (不变)
# =========================
def centroid_xy(arr: np.ndarray):
    """(不变) 计算质心"""
    H, W = arr.shape
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    s = float(arr.sum())
    if s <= 1e-12:
        return float((W-1)/2), float((H-1)/2)
    cx = float((arr * x_idx).sum() / s)
    cy = float((arr * y_idx).sum() / s)
    return cx, cy

def safe_stem(name: str):
    """(不变)"""
    stem = Path(name).stem
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)

def crop_local_window(arr: np.ndarray, r_peak: int, c_peak: int, ph: int, pw: int):
    """(不变) 裁剪局部窗口"""
    H, W = arr.shape
    hr, hc = ph // 2, pw // 2

    r0 = np.clip(r_peak - hr, 0, H - 1)
    r1 = np.clip(r_peak + hr + 1, 0, H)
    c0 = np.clip(c_peak - hc, 0, W - 1)
    c1 = np.clip(c_peak + hc + 1, 0, W)

    if r1 <= r0: r1 = r0 + 1
    if c1 <= c0: c1 = c0 + 1

    local_arr = arr[r0:r1, c0:c1]

    return local_arr, r0, c0

def export_qparams(model: nn.Module, fpath: Path):
    """(不变) 导出量化参数"""
    qparams = {}
    for name, m in model.named_modules():
        if isinstance(m, (nnq.Linear, nniq.LinearReLU)):
            if m.weight().qscheme() == torch.per_tensor_affine:
                w_scale = float(m.weight().q_scale())
                w_zero = int(m.weight().q_zero_point())
                qparams[f"{name}.weight"] = {'scale': w_scale, 'zero_point': w_zero, 'scheme': 'per_tensor_affine'}
            elif m.weight().qscheme() == torch.per_channel_affine:
                 w_scales = m.weight().q_per_channel_scales().tolist()
                 w_zeros = m.weight().q_per_channel_zero_points().tolist()
                 qparams[f"{name}.weight"] = {'scales': w_scales, 'zero_points': w_zeros, 'scheme': 'per_channel_affine', 'axis': m.weight().q_per_channel_axis()}
            else:
                 print(f"[WARN] Unknown weight qscheme for {name}: {m.weight().qscheme()}")
                 continue

            a_scale, a_zero = float(m.scale), int(m.zero_point)
            qparams[f"{name}.activation"] = {'scale': a_scale, 'zero_point': a_zero}

            if m.bias() is not None:
                if m.weight().qscheme() == torch.per_tensor_affine:
                    b_scale = w_scale * a_scale
                    qparams[f"{name}.bias"] = {'scale': b_scale, 'zero_point': 0}
                else: # per-channel
                     b_scales = [ws * a_scale for ws in w_scales]
                     qparams[f"{name}.bias"] = {'scales': b_scales, 'zero_points': [0]*len(b_scales)}


        elif isinstance(m, tq.QuantStub):
            a_scale, a_zero = float(m.scale), int(m.zero_point)
            qparams[f"{name}.activation"] = {'scale': a_scale, 'zero_point': a_zero}

    with open(fpath, 'w') as f:
        json.dump(qparams, f, indent=2)

# =========================
# Data
# =========================
class TouchJSONDataset(Dataset):
    """(不变) 扫描目录，加载 JSON"""
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
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
                    buf.append({'merging': m, 'target': t, 'file': fname, 'id': k})
            except Exception as e:
                print(f"[WARN] skip {fname}: {e}")
        print(f"[INFO] loaded {len(buf)} samples from {self.data_dir}")
        return buf

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        rec = self.samples[idx]
        m = rec['merging']
        t = rec['target']
        return m, t, rec['file']

class SparseCompensationDataset(Dataset):
    """(不变) 3x6 局部裁剪 + K=8 稀疏点提取"""
    def __init__(self, base_ds: TouchJSONDataset,
                 k_points=K_SPARSE_POINTS,
                 patch_h=LOCAL_PATCH_H,
                 patch_w=LOCAL_PATCH_W):
        self.base = base_ds
        self.k_points = k_points
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.H = 32
        self.W = 18

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        M, T, file_str = self.base[idx]

        r_peak, c_peak = np.unravel_index(np.argmax(M), M.shape)

        M_local, r0, c0 = crop_local_window(M, r_peak, c_peak, self.patch_h, self.patch_w)
        T_local, _,  _  = crop_local_window(T, r_peak, c_peak, self.patch_h, self.patch_w)

        local_H, local_W = M_local.shape

        x_raw_local, y_raw_local = centroid_xy(M_local)
        x_gt_local,  y_gt_local  = centroid_xy(T_local)

        dx = x_gt_local - x_raw_local
        dy = y_gt_local - y_raw_local
        target = np.array([dx, dy], dtype=np.float32)

        flat_M_local = M_local.flatten()
        k_actual = min(self.k_points, len(flat_M_local))

        if k_actual == 0:
            indices_1d = np.array([], dtype=int)
            values = np.array([], dtype=np.float32)
        elif len(flat_M_local) > k_actual:
            indices_1d = np.argpartition(flat_M_local, -k_actual)[-k_actual:]
            sorted_k_indices = indices_1d[np.argsort(-flat_M_local[indices_1d])]
            values = flat_M_local[sorted_k_indices]
            indices_1d = sorted_k_indices
        else:
            indices_1d = np.argsort(-flat_M_local)
            values = flat_M_local[indices_1d]

        rows_local, cols_local = np.unravel_index(indices_1d, (local_H, local_W))

        # --- (修改) 使用 Min-Max 归一化代替 Z-score ---
        norm_vals = np.zeros(k_actual, dtype=np.float32)
        if k_actual > 0:
            log_vals = np.log1p(values).astype(np.float32)
            min_log = np.min(log_vals)
            max_log = np.max(log_vals)
            range_log = max_log - min_log
            if range_log > 1e-6:
                norm_vals = (log_vals - min_log) / range_log # Scale to [0, 1]
            else:
                # If range is zero (all values the same), map to 0.5
                norm_vals[:] = 0.5
        # --- 结束修改 ---

        norm_rows_local = rows_local.astype(np.float32) / (self.patch_h - 1.0)
        norm_cols_local = cols_local.astype(np.float32) / (self.patch_w - 1.0)

        x_in = np.zeros((self.k_points, 3), dtype=np.float32)
        x_in[:k_actual, 0] = norm_vals
        x_in[:k_actual, 1] = norm_rows_local
        x_in[:k_actual, 2] = norm_cols_local
        x_in = x_in.flatten()

        meta = np.array([c_peak / (self.W - 1.0), r_peak / (self.H - 1.0)], dtype=np.float32)

        x_raw_g = c0 + x_raw_local
        y_raw_g = r0 + y_raw_local
        x_gt_g  = c0 + x_gt_local
        y_gt_g  = r0 + y_gt_local

        return {
            "x": torch.from_numpy(x_in),
            "y": torch.from_numpy(target),
            "meta": torch.from_numpy(meta),
            "raw_g": torch.tensor([x_raw_g, y_raw_g], dtype=torch.float32),
            "gt_g":  torch.tensor([x_gt_g,  y_gt_g ], dtype=torch.float32),
            "file":  file_str
        }

# =========================
# Model (不变)
# =========================
class SparseCompensationMLP(nn.Module):
    """(不变) MLP + QuantStub/DeQuantStub"""
    def __init__(self, k_points=K_SPARSE_POINTS, width=WIDTH, hidden=16, max_dx=4.0, max_dy=MAX_DY_ABS):
        super().__init__()

        sparse_features = k_points * 3
        meta_features = 2
        input_dim = sparse_features + meta_features

        self.quant_in = tq.QuantStub()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2)
        )

        self.dequant_out = tq.DeQuantStub()

        self.max_dx, self.max_dy = max_dx, max_dy

    def forward(self, x, meta):
        h = torch.cat([x, meta], dim=1)

        h = self.quant_in(h)
        h = self.net(h)
        out = self.dequant_out(h)

        out = torch.tanh(out)
        out = torch.stack([out[:, 0] * self.max_dx, out[:, 1] * self.max_dy], dim=1)
        return out

    def fuse_model(self):
        """(不变) 融合 (Linear, ReLU) 对"""
        if self.net:
            torch.ao.quantization.fuse_modules(
                self.net, [('0', '1'), ('2', '3')], inplace=True
            )

# =========================
# Train / Eval / Infer-Plot (不变)
# =========================
def train_one_epoch(model, loader, opt, device, epoch_idx: int, total_epochs: int):
    """(不变)"""
    if epoch_idx < CVAR_WARMUP_EPOCHS:
        alpha_now = 1.0
    else:
        span = max(1, total_epochs - CVAR_WARMUP_EPOCHS)
        t = min(1.0, (epoch_idx - CVAR_WARMUP_EPOCHS + 1) / span)
        alpha_now = 1.0 + (TAIL_ALPHA - 1.0) * t

    model.train(); s = 0.0
    for b in loader:
        x = b["x"].to(device, dtype=torch.float32)
        y = b["y"].to(device, dtype=torch.float32)
        meta = b["meta"].to(device, dtype=torch.float32)
        pred = model(x, meta)

        loss_x_i = _per_sample_smooth_l1(pred[:,0] - y[:,0], beta=1.0) * LOSS_W_X
        loss_y_i = _per_sample_smooth_l1(pred[:,1] - y[:,1], beta=1.0) * LOSS_W_Y
        loss_i = loss_x_i + loss_y_i

        loss = _cvar_mean(loss_i, alpha_now)

        opt.zero_grad()
        loss.backward()
        opt.step()

        s += loss_i.mean().item() * x.size(0)

    return s / len(loader.dataset), float(alpha_now)


@torch.no_grad()
def validate_mae(model, loader, device):
    """(不变)"""
    model.eval()
    ax, ay = [], []
    for b in loader:
        x = b["x"].to(device, dtype=torch.float32)
        y = b["y"].to(device, dtype=torch.float32)
        meta = b["meta"].to(device, dtype=torch.float32)
        p = model(x, meta)
        ax.append(torch.abs(p[:,0]-y[:,0]).cpu().numpy())
        ay.append(torch.abs(p[:,1]-y[:,1]).cpu().numpy())
    ax = float(np.mean(np.concatenate(ax))) if ax else 0.0
    ay = float(np.mean(np.concatenate(ay))) if ay else 0.0
    return {"MAE_dx": ax, "MAE_dy": ay}

def _per_sample_smooth_l1(diff: torch.Tensor, beta: float = 1.0):
    """(不变)"""
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < beta, 0.5 * (abs_diff ** 2) / beta, abs_diff - 0.5 * beta)
    while loss.dim() > 1:
        loss = loss.mean(dim=-1)
    return loss

def _cvar_mean(loss_vec: torch.Tensor, tail_alpha: float) -> torch.Tensor:
    """(不变)"""
    B = loss_vec.numel()
    k = max(1, int(math.ceil(tail_alpha * B)))
    topk_vals, _ = torch.topk(loss_vec, k=k, largest=True, sorted=False)
    return topk_vals.mean()


@torch.no_grad()
def infer_and_plot_per_file(model, loader, device, out_dir: Path, alpha_y=ALPHA_Y):
    """(不变) 可视化逻辑"""
    out_dir.mkdir(parents=True, exist_ok=True)
    mae_rx, mae_cx, mae_ry, mae_cy = [], [], [], []
    buckets = defaultdict(lambda: {"gt": [], "raw": [], "cor": []})

    all_gt_points, all_raw_points, all_cor_points = [], [], []

    for b in loader:
        x  = b["x"].to(device, dtype=torch.float32)
        meta = b["meta"].to(device, dtype=torch.float32)

        rg = b["raw_g"].numpy()
        gg = b["gt_g"].numpy()
        files = b["file"]

        pd = model(x, meta).cpu().numpy()
        pd[:,1] = np.clip(pd[:,1], -MAX_DY_ABS, MAX_DY_ABS)

        for j in range(x.size(0)):
            xr, yr = float(rg[j,0]), float(rg[j,1])
            xg, yg = float(gg[j,0]), float(gg[j,1])
            dx, dy = float(pd[j,0]), float(pd[j,1])

            xc, yc = xr + dx, yr + alpha_y * dy

            mae_rx.append(abs(xr - xg)); mae_cx.append(abs(xc - xg))
            mae_ry.append(abs(yr - yg)); mae_cy.append(abs(yc - yg))

            key = safe_stem(os.path.basename(files[j]))
            buckets[key]["gt"].append((xg, yg))
            buckets[key]["raw"].append((xr, yr))
            buckets[key]["cor"].append((xc, yc))

            all_gt_points.append((xg, yg))
            all_raw_points.append((xr, yr))
            all_cor_points.append((xc, yc))

    # --- 可视化 1: 每文件一张图 (不变) ---
    def A(v): return v*64.0 + 32.0
    for key, D in buckets.items():
        gt  = np.array(D["gt"],  dtype=np.float32)
        raw = np.array(D["raw"], dtype=np.float32)
        cor = np.array(D["cor"], dtype=np.float32)

        plt.figure(figsize=(6,6), dpi=150)
        plt.scatter(A(gt[:,0]),  A(gt[:,1]),  marker='s', s=16, label='gt')
        plt.scatter(A(raw[:,0]), A(raw[:,1]), marker='o', s=14, label='raw', alpha=0.85)
        plt.scatter(A(cor[:,0]), A(cor[:,1]), marker='^', s=14, label='corrected', alpha=0.85)

        LINE_STEP = max(1, len(gt)//200)
        for k in range(0, len(gt), LINE_STEP):
            plt.plot([A(gt[k,0]), A(raw[k,0])], [A(gt[k,1]), A(raw[k,1])],
                     linestyle='--', linewidth=0.6, alpha=0.45)
            plt.plot([A(gt[k,0]), A(cor[k,0])], [A(gt[k,1]), A(cor[k,1])],
                     linestyle='--', linewidth=0.6, alpha=0.8)

        plt.title(key)
        plt.xlabel("x (*64+32)"); plt.ylabel("y (*64+32)")
        plt.legend(loc='best', frameon=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}.png")
        plt.close()

    # --- 可视化 2: 全局汇总图 (不变) ---
    if all_gt_points:
        gt_all  = np.array(all_gt_points,  dtype=np.float32)
        raw_all = np.array(all_raw_points, dtype=np.float32)
        cor_all = np.array(all_cor_points, dtype=np.float32)

        plt.figure(figsize=(10, 8), dpi=150)
        plt.scatter(A(gt_all[:,0]),  A(gt_all[:,1]),  marker='s', s=16, label=f'gt ({len(gt_all)} pts)', alpha=0.5)
        plt.scatter(A(raw_all[:,0]), A(raw_all[:,1]), marker='o', s=14, label='raw', alpha=0.5)
        plt.scatter(A(cor_all[:,0]), A(cor_all[:,1]), marker='^', s=14, label='corrected', alpha=0.5)

        LINE_STEP = max(1, len(gt_all)//500)
        for k in range(0, len(gt_all), LINE_STEP):
            plt.plot([A(gt_all[k,0]), A(raw_all[k,0])], [A(gt_all[k,1]), A(raw_all[k,1])],
                     'r-', linewidth=0.3, alpha=0.2)
            plt.plot([A(gt_all[k,0]), A(cor_all[k,0])], [A(gt_all[k,1]), A(cor_all[k,1])],
                     'g-', linewidth=0.3, alpha=0.3)

        plt.title(f"Global Summary: All {len(gt_all)} Validation Points")
        plt.xlabel("x"); plt.ylabel("y")
        plt.legend(loc='best', frameon=True)
        plt.grid(True, linestyle=':', alpha=0.5)
        plt.tight_layout()
        plt.savefig(out_dir / "_GLOBAL_SUMMARY.png")
        plt.close()

    # --- 可视化 3: X 轴误差直方图 (不变) ---
    if mae_rx:
        plt.figure(figsize=(10, 5), dpi=120)
        max_err = max(0.5, np.max(mae_rx) * 1.1)
        bins = np.linspace(0, max_err, 100)
        plt.hist(np.clip(mae_rx, bins[0], bins[-1]), bins=bins, alpha=0.7, label=f'Raw Error (MAE={np.mean(mae_rx):.4f})', color='red')
        plt.hist(np.clip(mae_cx, bins[0], bins[-1]), bins=bins, alpha=0.7, label=f'Corrected Error (MAE={np.mean(mae_cx):.4f})', color='blue')
        plt.title('X-Axis Absolute Error Distribution (Global)')
        plt.xlabel('Error (dx)')
        plt.ylabel('Count')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "_GLOBAL_ERROR_HIST_X.png")
        plt.close()

    def avg(v): return float(np.mean(v)) if v else 0.0
    rep = {
        "MAE_x_raw": avg(mae_rx), "MAE_x_corr": avg(mae_cx),
        "MAE_y_raw": avg(mae_ry), "MAE_y_corr": avg(mae_cy),
    }
    rep["Imp_x(%)"] = (rep["MAE_x_raw"]-rep["MAE_x_corr"])/(rep["MAE_x_raw"]+1e-9)*100.0
    rep["Imp_y(%)"] = (rep["MAE_y_raw"]-rep["MAE_y_corr"])/(rep["MAE_y_raw"]+1e-9)*100.0
    return rep

# =========================
# Main (不变)
# =========================
def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tr_base = TouchJSONDataset(TRAIN_DIR)
    va_base = TouchJSONDataset(VAL_DIR)
    ds_tr = SparseCompensationDataset(tr_base, k_points=K_SPARSE_POINTS)
    ds_va = SparseCompensationDataset(va_base, k_points=K_SPARSE_POINTS)

    loader_tr = DataLoader(ds_tr, batch_size=BATCH_TR, shuffle=True,  num_workers=0)
    loader_va = DataLoader(ds_va, batch_size=BATCH_VA, shuffle=False, num_workers=0)

    model = SparseCompensationMLP(k_points=K_SPARSE_POINTS, width=WIDTH, hidden=16)

    model.fuse_model()

    if USE_QAT:
        print(f"[INFO] Using QAT with engine: {QENGINE}")
        model.cpu()
        torch.backends.quantized.engine = QENGINE
        model.qconfig = tq.get_default_qat_qconfig(QENGINE)
        tq.prepare_qat(model, inplace=True)

    model = model.to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)

    best = math.inf
    best_state = None
    bad = 0
    for ep in range(EPOCHS):
        tr_loss, alpha_now = train_one_epoch(model, loader_tr, opt, DEVICE, epoch_idx=ep, total_epochs=EPOCHS)

        model.eval()
        ev = validate_mae(model, loader_va, DEVICE)
        if USE_QAT:
            model.train()

        score = ev["MAE_dx"] + ev["MAE_dy"]
        sched.step(score)
        print(f"Epoch {ep:02d} | train_loss(mean)={tr_loss:.4f} | val={ev} | CVaR alpha={alpha_now:.2f}")
        if score < best - 1e-6:
            best = score
            bad = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, MODEL_FLOAT_PATH)
        else:
            bad += 1
            if bad >= EARLY_STOP_PATIENCE:
                print("[INFO] Early stop."); break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        torch.save(best_state, MODEL_FLOAT_PATH)
    print(f"[INFO] best proxy score={best:.6f} | saved -> {MODEL_FLOAT_PATH}")

    model_fp32 = SparseCompensationMLP(k_points=K_SPARSE_POINTS, width=WIDTH, hidden=16)
    model_fp32.fuse_model()
    model_fp32.qconfig = tq.get_default_qat_qconfig(QENGINE)
    tq.prepare_qat(model_fp32, inplace=True)
    model_fp32.load_state_dict(best_state)
    model_fp32.eval()

    if USE_QAT:
        print("[INFO] Converting to INT8 model...")
        model_cpu = model_fp32.cpu()
        quant_model = tq.convert(model_cpu, inplace=False)
        quant_model.eval()

        torch.save(quant_model.state_dict(), MODEL_INT8_PATH)
        print(f"[INFO] saved int8 model -> {MODEL_INT8_PATH}")

        try:
             export_qparams(quant_model, QP_JSON_PATH)
             print(f"[INFO] exported quant params -> {QP_JSON_PATH}")
        except Exception as e:
             print(f"[ERROR] Failed to export qparams: {e}")
             print(f"[INFO] Skipping qparams export due to error.")


        eval_model = quant_model
        eval_device = torch.device("cpu")
    else:
        eval_model = model_fp32.to(DEVICE)
        eval_device = DEVICE

    print(f"[INFO] Running final validation on {eval_device}...")
    rep = infer_and_plot_per_file(eval_model, loader_va, eval_device, OUT_DIR / "figs_val", alpha_y=ALPHA_Y)
    print("[REPORT on validation_int]", rep)

    with open(OUT_DIR / "figs_val" / "_GLOBAL_REPORT.json", 'w') as f:
        json.dump(rep, f, indent=2)

if __name__ == "__main__":
    main()

