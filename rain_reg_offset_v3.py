# -*- coding: utf-8 -*-
"""
Train (CVaR/Top-k) → FX QAT (prepare/convert) → INT8 convert →
validation + per-file visualization.

最终修正版:
- 确保 QAT observer 状态在 convert_fx 时被使用。
- 简化转换流程，直接使用内存中训练好的 QAT 模型进行转换，
  不再依赖易出错的 QAT state_dict 保存/加载。
"""

import os, re, json, math, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Hide common QAT warnings if desired

# =====================
# CONFIG
# =====================
TRAIN_DIR  = "/work/hwc/SENSOR_STAGE3/training_data/aligned_data_for_training_int" # 假设训练数据在 data/
VAL_DIR    = "training_data/validation_int" # 假设验证数据在 data/ (请修改为您正确的路径)
OUT_DIR    = Path("runs_offset_v12_final") # Changed output dir name
MODEL_FLOAT_PATH = OUT_DIR / "best_float.pt"
# MODEL_QAT_PATH   = OUT_DIR / "best_qat_fakequant.pt" # 不再需要
MODEL_INT8_PATH  = OUT_DIR / "best_int8.pt" # 最终产物
JIT_INT8_PATH    = OUT_DIR / "best_int8_script.pt"

PATCH_H, PATCH_W   = 3, 7
BATCH_TR, BATCH_VA = 64, 256
EPOCHS             = 60 # 浮点预训练的总轮数
SEED               = 0
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Y handling (default: not correcting Y)
MAX_DY_ABS = 0.25
LOSS_W_X   = 6.0
LOSS_W_Y   = 0.0
ALPHA_Y    = 0.0

LEARNING_RATE = 2e-3

# ============ Global Normalization (Plan B) ============
# Use fixed global mean/std for the intensity feature channel to minimize MCU cost.
# If True: feature = (raw_patch - MEAN_G) * INV_STD_G
# If False: keep original per-patch log1p + z-score (more expensive, not MCU-friendly)
USE_GLOBAL_NORM = True
# Whether to apply log1p before computing/applying global stats. For MCU simplicity, keep False.
GLOBAL_NORM_USE_LOG = False
# Paths to cache/load global stats
GLOBAL_NORM_CACHE = OUT_DIR / "global_norm.json"

# Defaults; will be filled by scanning training data if USE_GLOBAL_NORM
MEAN_G = None
INV_STD_G = None  # 1/std_g
WIDTH      = 4

# Risk-averse loss
TAIL_ALPHA = 0.10
CVAR_WARMUP_EPOCHS = 5
EARLY_STOP_PATIENCE = 10

# Quantization settings
QENGINE             = "fbgemm"  # x86 服务器使用 "fbgemm"
QAT_START_EPOCH     = 25 # 浮点训练达到此轮数时开始 QAT
QAT_FINE_TUNE_EPOCHS= 20 # QAT 微调的轮数

# =====================
# Priors & utilities
# =====================
# ... (所有辅助函数 get_short_pairs_for_row, centroid_xy,
#      crop_patch_with_indices, build_side_prior_patch, safe_stem 保持不变) ...
def get_short_pairs_for_row(row: int, cols: int = 18):
    odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
    even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    return (odd_pairs if (row % 2 == 1) else even_pairs)
def centroid_xy(arr: np.ndarray):
    H, W = arr.shape
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    s = float(arr.sum())
    if s <= 1e-12: return float((W-1)/2), float((H-1)/2)
    cx = float((arr * x_idx).sum() / s)
    cy = float((arr * y_idx).sum() / s)
    return cx, cy
def crop_patch_with_indices(arr: np.ndarray, cy: int, cx: int, ph=3, pw=7):
    H, W = arr.shape
    hy, hx = ph//2, pw//2
    y0, y1 = cy - hy, cy + hy + 1
    x0, x1 = cx - hx, cx + hx + 1
    ys = np.clip(np.arange(y0, y1), 0, H-1)
    xs = np.clip(np.arange(x0, x1), 0, W-1)
    patch = arr[np.ix_(ys, xs)]
    return patch, ys, xs
def build_side_prior_patch(ys, xs, cols=18):
    ph, pw = len(ys), len(xs)
    side = np.zeros((ph, pw), dtype=np.float32)
    for i, r in enumerate(ys):
        paired = {c for a,b in get_short_pairs_for_row(int(r), cols) for c in (a,b)}
        for j, c in enumerate(xs):
            if int(c) in paired: side[i, j] = 1.0
    return side
def safe_stem(name: str):
    stem = Path(name).stem
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', stem)


# =====================
# Dataset
# =====================
class TouchJSONDataset(Dataset):
# ... (rest of the class is the same) ...
    def __init__(self, data_dir: str):
        super().__init__()
        self.data_dir = data_dir
        self.samples = self._load_all()
    def _load_all(self):
        files = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.json')]
        if not files:
            print(f"[ERROR] No .json files found in {self.data_dir}")
            print(f"        Please check the TRAIN_DIR/VAL_DIR paths.")
        files.sort(); buf = []
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
        if not buf:
             print(f"[ERROR] No samples loaded from {self.data_dir}")
        else:
             print(f"[INFO] loaded {len(buf)} samples from {self.data_dir}")
        return buf
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        rec = self.samples[idx]
        m = torch.from_numpy(rec['merging']).clamp_min(0)
        t = torch.from_numpy(rec['target']).clamp_min(0)
        return m.unsqueeze(0), t.unsqueeze(0), rec['file']


class PatchOffsetDataset(Dataset):
# ... (rest of the class is the same) ...
    def __init__(self, base_ds: TouchJSONDataset, ph=3, pw=7,
                 use_global_norm: bool = USE_GLOBAL_NORM,
                 mean_g: float | None = MEAN_G,
                 inv_std_g: float | None = INV_STD_G,
                 global_norm_use_log: bool = GLOBAL_NORM_USE_LOG):
        self.base = base_ds; self.ph, self.pw = ph, pw
        self.use_global_norm = use_global_norm
        self.mean_g = mean_g
        self.inv_std_g = inv_std_g
        self.global_norm_use_log = global_norm_use_log
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        m, t, file_str = self.base[idx]
        M = m[0].numpy(); T = t[0].numpy()
        H_full, W_full = M.shape # Get full shape
        cx_full, cy_full = centroid_xy(M)
        cy, cx = int(round(cy_full)), int(round(cx_full))
        pM, ys, xs = crop_patch_with_indices(M, cy, cx, self.ph, self.pw)
        pT, _,  _  = crop_patch_with_indices(T, cy, cx, self.ph, self.pw)
        x_raw_p, y_raw_p = centroid_xy(pM)
        x_gt_p,  y_gt_p  = centroid_xy(pT)
        dx, dy = x_gt_p - x_raw_p, y_gt_p - y_raw_p
        target = np.array([dx, dy], dtype=np.float32)
        # --- Feature engineering (Input X, channel 0) ---
        if self.use_global_norm and (self.mean_g is not None) and (self.inv_std_g is not None):
            feat_src = np.log1p(pM).astype(np.float32) if self.global_norm_use_log else pM.astype(np.float32)
            feat = (feat_src - float(self.mean_g)) * float(self.inv_std_g)
        else:
            # Fallback to original (more expensive) per-patch normalization
            feat = np.log1p(pM).astype(np.float32)
            feat_mean = feat.mean(); feat_std = feat.std()
            feat = (feat - feat_mean) / (feat_std + 1e-6)
        side = build_side_prior_patch(ys, xs, cols=W_full).astype(np.float32)
        x_in = np.stack([feat, side], axis=0) # Shape [2, ph, pw]
        x_raw_g = xs[0] + x_raw_p; y_raw_g = ys[0] + y_raw_p
        meta_x = x_raw_g / (W_full - 1) if W_full > 1 else 0.0
        meta_y = y_raw_g / (H_full - 1) if H_full > 1 else 0.0
        meta = np.array([meta_x, meta_y], dtype=np.float32)
        x_gt_g  = xs[0] + x_gt_p;  y_gt_g  = ys[0] + y_gt_p
        return {
            "x": torch.from_numpy(x_in),
            "meta": torch.from_numpy(meta),
            "y": torch.from_numpy(target),
            "raw_g": torch.tensor([x_raw_g, y_raw_g], dtype=torch.float32),
            "gt_g":  torch.tensor([x_gt_g,  y_gt_g ], dtype=torch.float32),
            "file":  file_str,
        }

# =====================
# Model
# =====================
class DWConvBlock(nn.Module):
# ... (rest of the class is the same) ...
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, padding=k//2, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class OffsetNetWithMeta(nn.Module):
# ... (rest of the class is the same - no stubs needed) ...
    def __init__(self, in_c=2, width=4, max_dx=4.0, max_dy=MAX_DY_ABS):
        super().__init__()
        self.block1 = DWConvBlock(in_c,  width)
        self.block2 = DWConvBlock(width, width)
        self.gap    = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Linear(width+2, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 2)
        )
        self.max_dx, self.max_dy = max_dx, max_dy

    def forward(self, x, meta):
        z = self.block1(x); z = self.block2(z); z = self.gap(z).flatten(1)
        h = torch.cat([z, meta], dim=1)
        out_intermediate = self.head(h)
        out = torch.tanh(out_intermediate)
        out = torch.stack([out[:,0]*self.max_dx, out[:,1]*self.max_dy], dim=1)
        return out

# =====================
# Training / Eval / Plot
# =====================

def _per_sample_smooth_l1(diff: torch.Tensor, beta: float = 1.0):
# ... (rest of the function is the same) ...
    abs_diff = diff.abs()
    loss = torch.where(abs_diff < beta, 0.5 * (abs_diff ** 2) / beta, abs_diff - 0.5 * beta)
    while loss.dim() > 1: loss = loss.mean(dim=-1)
    return loss
def _cvar_mean(loss_vec: torch.Tensor, tail_alpha: float) -> torch.Tensor:
# ... (rest of the function is the same) ...
    B = loss_vec.numel(); k = max(1, int(math.ceil(tail_alpha * B)))
    topk_vals, _ = torch.topk(loss_vec, k=k, largest=True, sorted=False)
    return topk_vals.mean()
def train_one_epoch(model, loader, opt, device, epoch_idx: int, total_epochs: int):
# ... (rest of the function is the same) ...
    if epoch_idx < CVAR_WARMUP_EPOCHS: alpha_now = 1.0
    else: span = max(1, total_epochs - CVAR_WARMUP_EPOCHS); t = min(1.0, (epoch_idx - CVAR_WARMUP_EPOCHS + 1) / span); alpha_now = 1.0 + (TAIL_ALPHA - 1.0) * t
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
        opt.zero_grad(); loss.backward(); opt.step()
        s += loss_i.mean().item() * x.size(0)
    return s / len(loader.dataset), float(alpha_now)
@torch.no_grad()
def validate_mae(model, loader, device):
# ... (rest of the function is the same) ...
    model.eval(); ax, ay = [], []
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
@torch.no_grad()
def infer_and_plot_per_file(model, loader, device, out_dir: Path, alpha_y=ALPHA_Y):
# ... (rest of the function is the same) ...
    out_dir.mkdir(parents=True, exist_ok=True)
    mae_rx, mae_cx, mae_ry, mae_cy = [], [], [], []
    buckets = defaultdict(lambda: {"gt": [], "raw": [], "cor": []})
    model.eval()
    for b in loader:
        x  = b["x"].to(device, dtype=torch.float32)
        meta = b["meta"].to(device, dtype=torch.float32)
        rg = b["raw_g"].numpy(); gg = b["gt_g"].numpy(); files = b["file"]
        pd = model(x, meta).cpu().numpy();
        for j in range(x.size(0)):
            xr, yr = float(rg[j,0]), float(rg[j,1]); xg, yg = float(gg[j,0]), float(gg[j,1])
            dx, dy = float(pd[j,0]), float(pd[j,1])
            xc, yc = xr + dx, yr + alpha_y * dy
            mae_rx.append(abs(xr - xg)); mae_cx.append(abs(xc - xg)); mae_ry.append(abs(yr - yg)); mae_cy.append(abs(yc - yg))
            key = safe_stem(os.path.basename(files[j]))
            buckets[key]["gt"].append((xg, yg)); buckets[key]["raw"].append((xr, yr)); buckets[key]["cor"].append((xc, yc))
    def A(v): return v*64.0 + 32.0
    for key, D in buckets.items():
        gt  = np.array(D["gt"],  dtype=np.float32); raw = np.array(D["raw"], dtype=np.float32); cor = np.array(D["cor"], dtype=np.float32)
        if gt.size == 0: continue
        plt.figure(figsize=(6,6), dpi=150)
        plt.scatter(A(gt[:,0]),  A(gt[:,1]),  marker='s', s=16, label='gt', zorder=3)
        plt.scatter(A(raw[:,0]), A(raw[:,1]), marker='o', s=14, label='raw', alpha=0.7, zorder=2)
        plt.scatter(A(cor[:,0]), A(cor[:,1]), marker='^', s=14, label='corrected', alpha=0.8, zorder=2)
        LINE_STEP = max(1, len(gt)//100)
        for k in range(0, len(gt), LINE_STEP):
            plt.plot([A(gt[k,0]), A(raw[k,0])], [A(gt[k,1]), A(raw[k,1])], color='gray', linestyle='--', linewidth=0.6, alpha=0.45, zorder=1)
            plt.plot([A(gt[k,0]), A(cor[k,0])], [A(gt[k,1]), A(cor[k,1])], color='orange', linestyle='--', linewidth=0.6, alpha=0.6, zorder=1)
        plt.title(key); plt.xlabel("x (*64+32)"); plt.ylabel("y (*64+32)")
        plt.legend(loc='best', frameon=True); plt.grid(True, linestyle=':', alpha=0.5); plt.tight_layout(); plt.savefig(out_dir / f"{key}.png"); plt.close()
    def avg(v): return float(np.mean(v)) if v else 0.0
    rep = {"MAE_x_raw": avg(mae_rx), "MAE_x_corr": avg(mae_cx), "MAE_y_raw": avg(mae_ry), "MAE_y_corr": avg(mae_cy)}
    if rep["MAE_x_raw"] > 1e-9: rep["Imp_x(%)"] = (rep["MAE_x_raw"]-rep["MAE_x_corr"]) / rep["MAE_x_raw"] * 100.0
    else: rep["Imp_x(%)"] = 0.0
    if rep["MAE_y_raw"] > 1e-9: rep["Imp_y(%)"] = (rep["MAE_y_raw"]-rep["MAE_y_corr"]) / rep["MAE_y_raw"] * 100.0
    else: rep["Imp_y(%)"] = 0.0
    return rep

# =====================
# FX QAT helpers
# =====================
from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
from torch.ao.quantization.fuse_modules import fuse_modules_qat

def make_fx_qconfig_mapping(backend: str = QENGINE):
    torch.backends.quantized.engine = backend
    qconfig = get_default_qat_qconfig(backend)
    return QConfigMapping().set_global(qconfig)

def get_fusable_module_list(model):
# ... (rest of the function is the same) ...
    """ Helper to get list of module names to fuse (Conv-BN-ReLU pattern) """
    fusable_list = []
    for name, module in model.named_modules():
        if isinstance(module, DWConvBlock):
            # Fuse PW-BN-ReLU
            fusable_list.append([f"{name}.pw", f"{name}.bn", f"{name}.act"])
        elif isinstance(module, nn.Sequential):
            if name == "head":
                fusable_list.append(["head.0", "head.1"]) # Fuse Linear-ReLU
    print(f"[INFO] Modules identified for fusion: {fusable_list}")
    return fusable_list


# =====================
# Main
# =====================

def _load_or_estimate_global_norm(train_dir: str, ph: int, pw: int,
                                  use_log: bool, cache_path: Path) -> tuple[float, float]:
    """Compute global mean/std over training patches (channel 0), or load from cache.
       Returns (mean_g, inv_std_g)."""
    # Try load cache first
    try:
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            mean_g = float(obj["mean_g"]) ; std_g = float(obj["std_g"]) ; inv_std_g = 1.0/ max(1e-6, std_g)
            print(f"[INFO] Loaded global norm stats from {cache_path}: mean={mean_g:.6f}, std={std_g:.6f}")
            return mean_g, inv_std_g
    except Exception as e:
        print(f"[WARN] Failed to load global norm cache: {e}")

    # Estimate by single pass over training JSONs
    base = TouchJSONDataset(train_dir)
    if not base.samples:
        print(f"[ERROR] No samples in {train_dir}, cannot estimate global norm. Fallback to per-patch norm.")
        return None, None

    count = 0
    s = 0.0
    ss = 0.0
    for i in range(len(base)):
        m, t, file_str = base[i]
        M = m[0].numpy()
        H_full, W_full = M.shape
        cx_full, cy_full = centroid_xy(M)
        cy, cx = int(round(cy_full)), int(round(cx_full))
        pM, ys, xs = crop_patch_with_indices(M, cy, cx, ph, pw)
        feat_src = np.log1p(pM).astype(np.float32) if use_log else pM.astype(np.float32)
        # accumulate
        s  += float(feat_src.sum())
        ss += float((feat_src.astype(np.float64)**2).sum())
        count += feat_src.size
    if count == 0:
        print("[ERROR] Global norm count=0; fallback to per-patch norm.")
        return None, None
    mean_g = s / count
    var_g = max(0.0, ss / count - mean_g*mean_g)
    std_g = float(np.sqrt(var_g))
    inv_std_g = 1.0 / max(1e-6, std_g)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({"mean_g": mean_g, "std_g": std_g, "use_log": use_log, "ph": ph, "pw": pw}, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Saved global norm stats -> {cache_path}")
    except Exception as e:
        print(f"[WARN] Failed to save global norm cache: {e}")
    print(f"[INFO] Estimated global norm: mean={mean_g:.6f}, std={std_g:.6f}")
    return mean_g, inv_std_g

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data
    tr_base = TouchJSONDataset(TRAIN_DIR); va_base = TouchJSONDataset(VAL_DIR)
    if not tr_base.samples or not va_base.samples:
        print("[ERROR] No training or validation data loaded. Exiting.")
        return
    # Global normalization (Plan B)
    mean_g, inv_std_g = (None, None)
    if USE_GLOBAL_NORM:
        mean_g, inv_std_g = _load_or_estimate_global_norm(TRAIN_DIR, PATCH_H, PATCH_W, GLOBAL_NORM_USE_LOG, GLOBAL_NORM_CACHE)

    ds_tr = PatchOffsetDataset(tr_base, ph=PATCH_H, pw=PATCH_W,
                               use_global_norm=USE_GLOBAL_NORM,
                               mean_g=mean_g, inv_std_g=inv_std_g,
                               global_norm_use_log=GLOBAL_NORM_USE_LOG)
    ds_va = PatchOffsetDataset(va_base, ph=PATCH_H, pw=PATCH_W,
                               use_global_norm=USE_GLOBAL_NORM,
                               mean_g=mean_g, inv_std_g=inv_std_g,
                               global_norm_use_log=GLOBAL_NORM_USE_LOG)
    loader_tr = DataLoader(ds_tr, batch_size=BATCH_TR, shuffle=True,  num_workers=0)
    loader_va = DataLoader(ds_va, batch_size=BATCH_VA, shuffle=False, num_workers=0)

    # Model
    model = OffsetNetWithMeta(in_c=2, width=WIDTH).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=2e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    # --- Training Loop ---
    best_score = math.inf
    best_float_state = None
    qat_active = False
    # --- 修正: We only need to store the *in-memory* QAT model ---
    # model_prepared_for_qat = None # This will just be 'model'

    print(f"--- Starting Float Training (up to {EPOCHS} epochs) ---")
    for ep in range(EPOCHS + QAT_FINE_TUNE_EPOCHS):

        # --- Check if starting QAT ---
        if ep == QAT_START_EPOCH and not qat_active:
            print(f"\n--- Starting QAT at Epoch {ep} ---")
            qat_active = True
            
            # --- 修正: Load best float weights before QAT ---
            if best_float_state:
                print(f"Loading best float weights from epoch {ep-bad-1} (score {best_score:.4f})")
                model.load_state_dict(best_float_state)
            else:
                print("[WARN] No best float state found, starting QAT from last float epoch.")
            
            model.eval() # Set to eval before fusion/prepare
            model_to_prepare = model.to("cpu")

            # 1. Fuse Modules
            fusable_list = get_fusable_module_list(model_to_prepare)
            try:
                model_fused = fuse_modules_qat(model_to_prepare, fusable_list, inplace=True)
                print("[INFO] Modules fused successfully.")
            except Exception as e:
                print(f"[WARN] Module fusion failed: {e}. Proceeding without fusion.")
                model_fused = model_to_prepare # Fallback

            # 2. Prepare for QAT
            qmap = make_fx_qconfig_mapping(QENGINE)
            ex_x = torch.zeros(1, 2, PATCH_H, PATCH_W, dtype=torch.float32)
            ex_meta = torch.zeros(1, 2, dtype=torch.float32)
            try:
                # Prepare the fused model
                # --- 修正: model_prepared is now the main 'model' ---
                model = prepare_qat_fx(model_fused, qconfig_mapping=qmap, example_inputs=(ex_x, ex_meta))
                print("[INFO] Model prepared for QAT (FX).")
                model.to(DEVICE) # Move prepared model to device
                # Adjust optimizer for QAT model parameters
                opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE / 4, weight_decay=2e-5) # Use smaller LR
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
                # Reset best score for QAT phase
                best_score = math.inf 
                bad = 0
                print(f"--- QAT Fine-tuning for {QAT_FINE_TUNE_EPOCHS} epochs ---")
            except Exception as e:
                print(f"[ERROR] prepare_qat_fx failed: {e}")
                print("[ERROR] Cannot proceed with QAT. Exiting.")
                return

        # --- Train or Fine-tune ---
        epoch_label = f"QAT {ep - QAT_START_EPOCH:02d}" if qat_active else f"FP {ep:02d}"
        
        tr_loss, alpha_now = train_one_epoch(model, loader_tr, opt, DEVICE, epoch_idx=ep, total_epochs=EPOCHS+QAT_FINE_TUNE_EPOCHS)
        ev = validate_mae(model, loader_va, DEVICE)
        score = ev["MAE_dx"] + ev["MAE_dy"]
        sched.step(score)
        lr_current = opt.param_groups[0]['lr']

        print(f"Epoch {epoch_label} | LR={lr_current:.1e} | train_loss(mean)={tr_loss:.4f} | val={ev} | Score={score:.4f} | CVaR alpha={alpha_now:.2f}")

        # --- Save Best State (Float or QAT) ---
        if score < best_score - 1e-6:
            print(f"  New best score: {score:.4f} (was {best_score:.4f})")
            best_score = score
            bad = 0
            if qat_active:
                # --- 修正: No need to save QAT state, model is in memory ---
                # We will use the 'model' object directly after the loop
                pass
            else:
                # Save float state
                best_float_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_float_state, MODEL_FLOAT_PATH)
                print(f"  Saved best float state to {MODEL_FLOAT_PATH}")
        else:
             bad += 1
             if bad >= EARLY_STOP_PATIENCE:
                 print(f"[INFO] Early stop triggered after {bad} epochs without improvement.")
                 break

        # --- Exit QAT fine-tuning loop if done ---
        if qat_active and (ep - QAT_START_EPOCH + 1) >= QAT_FINE_TUNE_EPOCHS:
            print(f"\n--- QAT Fine-tuning finished after {QAT_FINE_TUNE_EPOCHS} epochs ---")
            break # Exit main loop after QAT fine-tuning

    # --- 修正: Simplified Conversion to INT8 ---
    if not qat_active:
        print("[WARN] QAT phase was not reached or completed. Cannot convert to INT8.")
        final_eval_state_path = MODEL_FLOAT_PATH
        model_to_eval = OffsetNetWithMeta(in_c=2, width=WIDTH) # Load float
        if best_float_state:
            model_to_eval.load_state_dict(best_float_state)
        else:
             print("[WARN] No best float state saved, using initial model state for final eval.")
             model_to_eval = model
    else:
        print("\n--- Converting final QAT model to INT8 ---")
        # --- 修正: Convert the IN-MEMORY 'model' directly ---
        # 'model' is the 'model_prepared_for_qat' instance
        model_prepared_final = model.to("cpu").eval() # Move to CPU and set to eval

        try:
            model_int8 = convert_fx(model_prepared_final)
            print("[INFO] Model converted to INT8 (FX).")

            # --- Add Inspection ---
            print("\n--- Inspecting Converted INT8 Model ---")
            n_trivial_act = 0
            n_total_act = 0
            has_quantized_weights = False
            for name, mod in model_int8.named_modules():
                 if hasattr(mod, 'scale') and hasattr(mod, 'zero_point'):
                      n_total_act += 1
                      scale = mod.scale.item() if hasattr(mod.scale, 'item') else mod.scale
                      zp = mod.zero_point.item() if hasattr(mod.zero_point, 'item') else mod.zero_point
                      print(f"  Layer (Act): {name:<20} | Scale: {scale:<12.5e} | Zero Point: {zp}")
                      if abs(scale - 1.0) < 1e-6 and zp == 0:
                           n_trivial_act += 1
                 if hasattr(mod, 'weight') and callable(getattr(mod, 'weight', None)) and mod.weight().is_quantized:
                      has_quantized_weights = True
                      wq = mod.weight()
                      if wq.qscheme() == torch.per_tensor_affine or wq.qscheme() == torch.per_tensor_symmetric:
                           scale = wq.q_scale(); zp = wq.q_zero_point()
                           print(f"  Layer (W): {name:<20} | Weight Scale: {scale:<12.5e} | Weight ZP: {zp} (Per-Tensor)")
                      elif wq.qscheme() == torch.per_channel_affine or wq.qscheme() == torch.per_channel_symmetric:
                           scales = wq.q_per_channel_scales(); zps = wq.q_per_channel_zero_points()
                           print(f"  Layer (W): {name:<20} | Weight Scales: [{scales.min().item():.3e} .. {scales.max().item():.3e}] | Weight ZPs: [{zps.min()} .. {zps.max()}] (Per-Channel)")

            if not has_quantized_weights:
                 print("[ERROR] No quantized weights found! Conversion likely failed.")
            elif n_total_act > 0 and n_trivial_act == n_total_act:
                 print("[ERROR] All activation quantization parameters are trivial (scale=1.0, zp=0).")
                 print("[ERROR] QAT likely failed. INT8 model may be inaccurate.")
            elif n_total_act == 0:
                 print("[WARN] No quantized activation parameters found during inspection.")
            else:
                 print(f"[INFO] Found {n_total_act - n_trivial_act} / {n_total_act} non-trivial activation qparams.")

            torch.save(model_int8.state_dict(), MODEL_INT8_PATH)
            print(f"[INFO] Saved INT8 state_dict -> {MODEL_INT8_PATH}")

            # Try TorchScript export
            try:
                model_int8.eval()
                scripted = torch.jit.script(model_int8)
                scripted.save(str(JIT_INT8_PATH))
                print(f"[INFO] Saved INT8 TorchScript -> {JIT_INT8_PATH}")
            except Exception as e:
                print(f"[WARN] TorchScript save failed: {e}")

        except Exception as e:
            print(f"[ERROR] convert_fx failed: {e}")
            print("[ERROR] Could not generate INT8 model.")
            return

        final_eval_state_path = MODEL_INT8_PATH
        model_to_eval = model_int8 # Use the converted model directly

    # ===== Final Eval & plots =====
    print(f"\n--- Final Evaluation using {final_eval_state_path.name} ---")
    if final_eval_state_path.exists():
        if final_eval_state_path == MODEL_INT8_PATH:
             pass # model_to_eval is already model_int8
        else: # Load float state
            try:
                 model_to_eval = OffsetNetWithMeta(in_c=2, width=WIDTH)
                 state = torch.load(final_eval_state_path, map_location=DEVICE, weights_only=True)
                 model_to_eval.load_state_dict(state)
            except Exception as e:
                 print(f"Error loading final state {final_eval_state_path}: {e}")
                 model_to_eval = model

        model_to_eval.to(DEVICE).eval()
        va_loader_device = DataLoader(ds_va, batch_size=BATCH_VA, shuffle=False, num_workers=0)
        final_report = infer_and_plot_per_file(model_to_eval, va_loader_device, DEVICE, OUT_DIR/"figs_final_eval", alpha_y=ALPHA_Y)
        print("\n[FINAL REPORT on validation_int]")
        for k, v in final_report.items():
             print(f"  {k}: {v:.4f}")
    else:
        print(f"Final model file not found: {final_eval_state_path}")


if __name__ == "__main__":
    main()

