import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore") # Suppress warnings during verification runs

# ========================================================================
# 1. 从 train_offset_QAT.py 复制必要的模型和数据处理定义
# ========================================================================
# --- 模型 ---
# (OffsetNetWithMeta, DWConvBlock)
# (与 export_weights.py 中的定义相同)
import torch.nn.functional as F
class DWConvBlock(nn.Module):
# ... (existing DWConvBlock code) ...
    def __init__(self, c_in, c_out, k=3):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, k, padding=k//2, groups=c_in, bias=False)
        self.pw = nn.Conv2d(c_in, c_out, 1, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        y = self.act(self.bn(y)) # Keep BN for prepare_fx compatibility
        return y


class OffsetNetWithMeta(nn.Module):
# ... (existing OffsetNetWithMeta code without stubs) ...
    def __init__(self, in_c=2, width=4, max_dx=4.0, max_dy=0.25):
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
        out_q = self.head(h)
        out = torch.tanh(out_q)
        out = torch.stack([out[:,0]*self.max_dx, out[:,1]*self.max_dy], dim=1)
        return out


# --- 数据加载 ---
# (TouchJSONDataset, PatchOffsetDataset)
# (centroid_xy, crop_patch_with_indices, build_side_prior_patch, get_short_pairs_for_row)
# (与 train_offset_QAT.py 中的定义相同 - 此处省略以保持简洁)
# Placeholder functions/classes - Replace with actual definitions
from torch.utils.data import Dataset # Need this import

def get_short_pairs_for_row(row: int, cols: int = 18):
    odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
    even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    return (odd_pairs if (row % 2 == 1) else even_pairs)

def centroid_xy(arr: np.ndarray):
    H, W = arr.shape
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    s = float(arr.sum())
    if s <= 1e-12:
        return float((W-1)/2), float((H-1)/2)
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
            if int(c) in paired:
                side[i, j] = 1.0
    return side

class TouchJSONDataset(Dataset):
    def __init__(self, data_dir: str, filename: str): # Modified to take specific filename
        super().__init__()
        self.data_dir = data_dir
        self.filename = filename
        self.samples = self._load_sample()
    def _load_sample(self):
        path = os.path.join(self.data_dir, self.filename)
        if not os.path.exists(path):
             print(f"[ERROR] Data file not found: {path}")
             return []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            # Use only sample "0"
            k = "1"
            if k in obj:
                 rec = obj[k]
                 m = np.array(rec['merging']['normalized_matrix'], dtype=np.float32)
                 t = np.array(rec['nonmerging']['normalized_matrix'], dtype=np.float32)
                 print(f"[INFO] Loaded sample {k} from {self.filename}")
                 return [{'merging': m, 'target': t, 'file': self.filename, 'id': k}]
            else:
                 print(f"[WARN] Sample '0' not found in {self.filename}")
                 return []
        except Exception as e:
            print(f"[WARN] Failed to load {self.filename}: {e}")
            return []
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        rec = self.samples[idx]
        m = torch.from_numpy(rec['merging']).clamp_min(0)
        t = torch.from_numpy(rec['target']).clamp_min(0)
        return m.unsqueeze(0), t.unsqueeze(0), rec['file']


class PatchOffsetDataset(Dataset):
    def __init__(self, base_ds: TouchJSONDataset, ph=3, pw=7):
        self.base = base_ds; self.ph, self.pw = ph, pw
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        # --- Using actual preprocessing from train_offset_QAT.py ---
        m, t, file_str = self.base[idx]
        M = m[0].numpy(); T = t[0].numpy()
        H_full, W_full = M.shape

        cx_full, cy_full = centroid_xy(M)
        cy, cx = int(round(cy_full)), int(round(cx_full))

        pM, ys, xs = crop_patch_with_indices(M, cy, cx, self.ph, self.pw)
        pT, _,  _  = crop_patch_with_indices(T, cy, cx, self.ph, self.pw)

        # Centroids within the patch
        x_raw_p, y_raw_p = centroid_xy(pM)
        x_gt_p,  y_gt_p  = centroid_xy(pT)

        # Target offset
        dx, dy = x_gt_p - x_raw_p, y_gt_p - y_raw_p
        target = np.array([dx, dy], dtype=np.float32)

        # Feature engineering (Input X)
        feat = np.log1p(pM).astype(np.float32)
        feat_mean = feat.mean()
        feat_std = feat.std()
        feat = (feat - feat_mean) / (feat_std + 1e-6) # Normalize
        side = build_side_prior_patch(ys, xs, cols=W_full).astype(np.float32)
        x_in = np.stack([feat, side], axis=0) # Shape [2, ph, pw]

        # Meta features (Input Meta) - global coordinates normalized
        x_raw_g = xs[0] + x_raw_p; y_raw_g = ys[0] + y_raw_p
        # Ensure division by zero is avoided if W_full/H_full is 1
        meta_x = x_raw_g / (W_full - 1) if W_full > 1 else 0.0
        meta_y = y_raw_g / (H_full - 1) if H_full > 1 else 0.0
        meta = np.array([meta_x, meta_y], dtype=np.float32)

        return {
            "x": torch.from_numpy(x_in),     # Shape [2, 3, 7]
            "meta": torch.from_numpy(meta),  # Shape [2]
            "y": torch.from_numpy(target), # Target offset (not used for golden data)
            "file": file_str,
        }

# ========================================================================
# 2. 从 export_weights.py 复制辅助函数
# ========================================================================
# --- 修正: Directly import from export_weights ---
try:
    # Assuming export_weights.py is in the same directory or Python path
    from export_weights import get_qat_converted_model
except ImportError:
    print("[ERROR] Could not import 'get_qat_converted_model' from 'export_weights.py'.")
    print("        Ensure 'export_weights.py' is in the same directory or Python path.")
    exit(1)


# Helper to quantize tensor
def quantize_tensor(x, scale, zero_point, dtype=torch.quint8):
     # Ensure scale and zero_point are floats/ints
     _scale = scale.item() if hasattr(scale, 'item') else scale
     _zero_point = zero_point.item() if hasattr(zero_point, 'item') else zero_point
     # Ensure scale is not zero
     if _scale == 0.0:
          print("[WARN] Attempting to quantize with zero scale. Returning tensor of zero_points.")
          return torch.full_like(x, _zero_point, dtype=dtype)
     # --- Use torch.quint8 based on export script usage ---
     return torch.quantize_per_tensor(x, _scale, _zero_point, torch.quint8)

# --- 修正: Helper function to save integer tensor ---
def save_int_tensor(tensor: torch.Tensor, filename: Path, dtype=np.int8):
    """ Saves the integer representation of a quantized tensor """
    if not tensor.is_quantized:
        print(f"[WARN] Tensor for {filename} is not quantized. Saving as is.")
        data = tensor.cpu().numpy().astype(dtype)
    else:
        # Get integer representation and convert to target type (e.g., int8)
        data = tensor.int_repr().cpu().numpy().astype(dtype)
    data.tofile(filename)
    print(f"  Saved INT tensor: {filename.name} (Shape: {data.shape}, Dtype: {data.dtype})")


# ========================================================================
# 3. 主函数：生成黄金数据
# ========================================================================
def generate_golden_data(model_path, data_dir, data_filename, out_dir_str):
    """
    加载 QAT 模型，运行它（模拟 FX 图），并保存
    浮点输入、最终浮点输出以及 **中间层整型输出**。
    """
    out_dir = Path(out_dir_str)
    out_dir.mkdir(exist_ok=True)

    # 1. 加载和预处理数据 (浮点)
    print(f"Loading and processing {data_filename}, sample 0...")
    base_ds = TouchJSONDataset(data_dir, data_filename)
    if len(base_ds) == 0:
         print("No samples loaded, cannot generate golden data.")
         return
    patch_ds = PatchOffsetDataset(base_ds, ph=3, pw=7) # Use ph, pw from training
    sample = patch_ds[0]
    x_f = sample['x'].unsqueeze(0)     # Shape [1, 2, 3, 7]
    meta_f = sample['meta'].unsqueeze(0) # Shape [1, 2]
    print(f"Input X shape: {x_f.shape}, Input Meta shape: {meta_f.shape}")


    # 2. 加载 FX 转换后的 QAT 模型
    print(f"Loading QAT model from {model_path}...")
    model_int8 = get_qat_converted_model(model_path)
    model_int8.eval()

    # --- 创建用于保存中间值的字典 ---
    intermediate_outputs_int = {}

    # 3. **[关键]** 模拟 FX 图的量化前向传播
    #    我们需要从模型中提取正确的 scale/zp
    print("Running Python model (float-in, float-out) & saving intermediates...")
    with torch.no_grad():
        # --- 获取输入 QParams ---
        x_scale = model_int8.block1_dw_input_scale_0.item()
        x_zp = model_int8.block1_dw_input_zero_point_0.item()
        meta_scale = model_int8._input_scale_0.item() # Used for meta
        meta_zp = model_int8._input_zero_point_0.item() # Used for meta

        # --- 获取 GAP 输出 re-quant / Cat 输入 qparams ---
        try:
             cat_input_scale = model_int8._scale_0.item()
             cat_input_zp = model_int8._zero_point_0.item()
             print(f"Found Cat Input q_params (_scale_0): ({cat_input_scale}, {cat_input_zp})")
             # Verify match with meta_scale/zp (as required by cat)
             if abs(cat_input_scale - meta_scale) > 1e-9 or cat_input_zp != meta_zp:
                  print(f"[ERROR] Mismatched quantization params for torch.cat!")
                  print(f"  GAP Output (re-quantized with _scale_0): ({cat_input_scale}, {cat_input_zp})")
                  print(f"  Meta Input (_input_scale_0):           ({meta_scale}, {meta_zp})")
                  print("  FX Graph is likely invalid or simulation will be inaccurate.")
                  # sys.exit("Exiting due to mismatched cat qparams.") # Optional: exit on error
        except AttributeError:
             print("[ERROR] Cannot find _scale_0/_zero_point_0 on converted model for CAT input.")
             print("[ERROR] Verification will likely fail. Using fallback (0.1, 0).")
             cat_input_scale = 0.1
             cat_input_zp = 0


        # --- 量化输入 ---
        q_x = quantize_tensor(x_f, x_scale, x_zp)
        q_meta = quantize_tensor(meta_f, meta_scale, meta_zp)
        intermediate_outputs_int["input_x_q"] = q_x
        intermediate_outputs_int["input_meta_q"] = q_meta

        # --- 通过模型层 (使用量化模块) ---
        q_b1dw = model_int8.block1.dw(q_x)
        intermediate_outputs_int["block1_dw_q"] = q_b1dw
        q_b1pw = model_int8.block1.pw(q_b1dw) # Includes ReLU
        intermediate_outputs_int["block1_pw_q"] = q_b1pw

        q_b2dw = model_int8.block2.dw(q_b1pw)
        intermediate_outputs_int["block2_dw_q"] = q_b2dw
        q_b2pw = model_int8.block2.pw(q_b2dw) # Includes ReLU
        intermediate_outputs_int["block2_pw_q"] = q_b2pw

        q_gap = model_int8.gap(q_b2pw)
        # Note: q_gap might still be float or quantized depending on FX graph
        # The FX graph code shows it's dequantized immediately after gap:
        # gap = self.gap(block2_pw); dequantize_6 = gap.dequantize()
        # So, save the dequantized version for consistency with graph steps
        dq_gap = q_gap.dequantize()

        # --- 模拟 FX 图中的 Dequant -> Flatten -> Requant for GAP output ---
        # dq_gap = q_gap.dequantize() # Already done based on graph code
        float_gap_flat = dq_gap.flatten(1)
        q_gap_flat_requant = quantize_tensor(float_gap_flat, cat_input_scale, cat_input_zp)
        intermediate_outputs_int["gap_requant_q"] = q_gap_flat_requant


        # --- 模拟 Cat 操作 ---
        q_cat_result = torch.cat([q_gap_flat_requant, q_meta], dim=1)
        intermediate_outputs_int["cat_q"] = q_cat_result


        # --- 通过 Head ---
        head0 = model_int8.head.get_submodule('0')
        head2 = model_int8.head.get_submodule('2')
        q_head0 = head0(q_cat_result) # head.0 (includes fused ReLU)
        intermediate_outputs_int["head0_q"] = q_head0
        q_head2 = head2(q_head0)      # head.2
        intermediate_outputs_int["head2_q"] = q_head2 # Final INT output

        # --- Dequantize 输出 ---
        dq_head2 = q_head2.dequantize()

        # --- 应用后处理 (Tanh, Scaling) ---
        tanh_out = torch.tanh(dq_head2)
        float_model_ref = OffsetNetWithMeta(in_c=2, width=4)
        final_out_f = torch.stack([
            tanh_out[:,0] * float_model_ref.max_dx,
            tanh_out[:,1] * float_model_ref.max_dy
        ], dim=1)

    # 4. 保存 Binaries
    print(f"\nSaving golden data to {out_dir_str}/")
    # Ensure tensors are on CPU and convert to numpy
    x_f_np = x_f.cpu().numpy()
    meta_f_np = meta_f.cpu().numpy()
    final_out_f_np = final_out_f.cpu().numpy()

    x_f_np.tofile(out_dir / "input_x.bin")
    meta_f_np.tofile(out_dir / "input_meta.bin")
    final_out_f_np.tofile(out_dir / "golden_output.bin")

    print(f"  Saved float input_x.bin (Shape {x_f_np.shape})")
    print(f"  Saved float input_meta.bin (Shape {meta_f_np.shape})")
    print(f"  Saved float golden_output.bin (Shape {final_out_f_np.shape})")
    print(f"Golden output (dx, dy): {final_out_f_np.flatten()}")

    # --- 修正: 保存中间 INT 输出 ---
    print("\nSaving intermediate integer tensors...")
    # Determine dtype (quint8 for activations, qint8 for weights usually)
    # Based on QConfig, activations are quint8 (0-255)
    intermediate_dtype = np.uint8 # Or np.int8 if using symmetric activation qconfig

    for name, tensor in intermediate_outputs_int.items():
        # --- Use quint8 (uint8) based on QConfig ---
        save_int_tensor(tensor, out_dir / f"golden_{name}.bin", dtype=np.uint8 if tensor.dtype==torch.quint8 else np.int8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", type=str, default="best_int8.pt",
                        help="Input INT8 state_dict (best_int8.pt)")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Directory containing input JSON data")
    parser.add_argument("--data-file", type=str, default="aligned_g1.json",
                        help="Specific input JSON filename (e.g., aligned_g1.json)")
    parser.add_argument("--out-dir", type=str, default="verify_data",
                        help="Output directory for golden files")
    args = parser.parse_args()

    # Use specified data file
    generate_golden_data(args.model_in, args.data_dir, args.data_file, args.out_dir)

