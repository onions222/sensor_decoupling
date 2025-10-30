# -*- coding: utf-8 -*-
import os
import json
import subprocess # 用于调用 C 程序
import numpy as np
import torch
import torch.ao.quantization as tq
import torch.nn.quantized as nnq
import torch.nn.intrinsic.quantized as nniq
from pathlib import Path
import argparse

# --- 从训练脚本导入必要的定义 ---
# (确保这些定义与你的训练脚本和 C 代码一致)

# 配置 (与训练脚本保持一致)
VAL_DIR = "/work/hwc/SPARSE/infer/val_data" # 验证数据目录
MODEL_INT8_PATH = Path("/work/hwc/SPARSE/runs_sparse_v2_3x6_qat_minmax/best_qat_int8.pt") # INT8 模型路径
C_EXECUTABLE_PATH = "./inference_app" # C 程序路径 (由 Makefile 生成)

LOCAL_PATCH_H = 3
LOCAL_PATCH_W = 6
K_SPARSE_POINTS = 8
WIDTH = 8
QENGINE = "fbgemm"
MAX_DY_ABS = 0.25
ALPHA_Y = 0.0
C_MLP_INPUT_SIZE = K_SPARSE_POINTS * 3 + 2
C_MLP_HIDDEN1_SIZE = WIDTH * 2 * 2
C_MLP_HIDDEN2_SIZE = WIDTH * 2
C_MLP_OUTPUT_SIZE = 2

# 模型定义 (与训练脚本一致)
from sparse_res import SparseCompensationMLP # 假设模型定义在训练脚本中

# 数据集定义 (与训练脚本一致)
from sparse_res import TouchJSONDataset, SparseCompensationDataset, centroid_xy, crop_local_window

# --- Python 推理函数 ---
@torch.no_grad()
def run_python_inference(model_int8, sample_data):
    """使用 PyTorch INT8 模型运行单个样本的推理"""
    model_int8.eval() # 确保在评估模式

    # 1. 解包数据 (与 Dataset __getitem__ 类似，但直接处理 numpy)
    M = sample_data['merging']
    # T = sample_data['target'] # 不需要目标 T 来推理
    H, W = M.shape

    # 2. 预处理 (峰值, 裁剪, 计算 baseline_cog, 提取稀疏特征)
    r_peak, c_peak = np.unravel_index(np.argmax(M), M.shape)
    M_local, r0, c0 = crop_local_window(M, r_peak, c_peak, LOCAL_PATCH_H, LOCAL_PATCH_W)
    local_H, local_W = M_local.shape

    x_raw_local, y_raw_local = centroid_xy(M_local)
    x_raw_g = c0 + x_raw_local
    y_raw_g = r0 + y_raw_local

    # 提取稀疏特征 (与 Dataset 一致)
    flat_M_local = M_local.flatten()
    k_actual = min(K_SPARSE_POINTS, len(flat_M_local))
    values = np.array([], dtype=np.float32)
    indices_1d = np.array([], dtype=int)

    if k_actual > 0 :
        non_zero_indices = np.where(flat_M_local > 1e-6)[0]
        if len(non_zero_indices) > 0:
             actual_values = flat_M_local[non_zero_indices]
             sorted_indices_local = np.argsort(-actual_values, kind='stable') # Sort descending by value (stable)

             k_to_take = min(k_actual, len(sorted_indices_local))

             top_k_indices_local = sorted_indices_local[:k_to_take]
             indices_1d = non_zero_indices[top_k_indices_local] # Get original flat indices
             values = flat_M_local[indices_1d] # Get corresponding values

             k_actual = len(values) # Update k_actual based on non-zero points found


    rows_local, cols_local = np.unravel_index(indices_1d, (local_H, local_W))

    # Min-Max 归一化 (与 Dataset 一致)
    norm_vals = np.zeros(k_actual, dtype=np.float32)
    if k_actual > 0:
        log_vals = np.log1p(values).astype(np.float32)
        min_log = np.min(log_vals)
        max_log = np.max(log_vals)
        range_log = max_log - min_log
        if range_log > 1e-6:
            norm_vals = (log_vals - min_log) / range_log
        else:
            norm_vals[:] = 0.5

    norm_rows_local = rows_local.astype(np.float32) / (LOCAL_PATCH_H - 1.0) if LOCAL_PATCH_H > 1 else np.full(k_actual, 0.5, dtype=np.float32)
    norm_cols_local = cols_local.astype(np.float32) / (LOCAL_PATCH_W - 1.0) if LOCAL_PATCH_W > 1 else np.full(k_actual, 0.5, dtype=np.float32)

    # 构建输入向量
    x_in_np = np.zeros((K_SPARSE_POINTS, 3), dtype=np.float32)
    x_in_np[:k_actual, 0] = norm_vals
    x_in_np[:k_actual, 1] = norm_rows_local
    x_in_np[:k_actual, 2] = norm_cols_local
    x_in_np = x_in_np.flatten()

    # 元特征
    meta_np = np.array([c_peak / (W - 1.0), r_peak / (H - 1.0)], dtype=np.float32)

    # 转换为 Tensor (需要 unsqueeze 添加 batch 维度)
    x_in_tensor = torch.from_numpy(x_in_np).unsqueeze(0)
    meta_tensor = torch.from_numpy(meta_np).unsqueeze(0)

    # 3. 模型推理
    pred_offset_tensor = model_int8(x_in_tensor, meta_tensor)
    pred_offset = pred_offset_tensor.squeeze(0).numpy() # 移除 batch 维度并转 numpy

    # 4. 计算最终 CoG
    dx, dy = pred_offset[0], pred_offset[1]
    # Clip dy based on MAX_DY_ABS (as done in plotting)
    dy = np.clip(dy, -MAX_DY_ABS, MAX_DY_ABS)

    xc_py = x_raw_g + dx
    yc_py = y_raw_g + ALPHA_Y * dy # 应用 ALPHA_Y

    return xc_py, yc_py, x_raw_g, y_raw_g, x_in_tensor, meta_tensor # 返回 Python 计算结果、基线以及模型输入张量

# --- C 推理调用函数 ---
def run_c_inference(json_file_path, sample_id, dump_path=None):
    """调用 C 可执行文件并解析其输出"""
    try:
        # 调用 C 程序, 传递 JSON 文件路径和样本 ID
        # C 程序应该将结果打印为 "xc,yc" 或类似格式
        # When dumping intermediates, instruct C program where to write them (dump_path passed from caller)
        env = dict(os.environ)
        if dump_path:
            env['DUMP_INTERMEDIATES_PATH'] = str(dump_path)
        result = subprocess.run(
            [C_EXECUTABLE_PATH, str(json_file_path), str(sample_id)],
            capture_output=True,
            text=True,
            env=env,
            check=True # 如果 C 程序返回非零退出码则抛出异常
        )
        # 解析 C 程序的标准输出
        stdout = result.stdout.strip()
        # Choose the last non-empty line (C may print debug lines before final result)
        last_line = None
        for ln in stdout.splitlines():
            s = ln.strip()
            if s:
                last_line = s
        if last_line is None:
            print(f"[ERROR] Empty C stdout for {json_file_path} ID {sample_id}")
            return None, None
        # Expect final line to be 'xc,yc' with two floats
        parts = last_line.split(',')
        if len(parts) == 2:
            try:
                xc_c = float(parts[0])
                yc_c = float(parts[1])
                return xc_c, yc_c
            except ValueError:
                print(f"[ERROR] Failed to parse final C output line: {last_line}")
                print(f"  Full stdout:\n{stdout}")
                return None, None
        else:
            print(f"[ERROR] Unexpected final C output format: {last_line}")
            print(f"  Full stdout:\n{stdout}")
            return None, None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] C executable failed for {json_file_path} ID {sample_id}:")
        print(f"  Return code: {e.returncode}")
        print(f"  Stdout: {e.stdout}")
        print(f"  Stderr: {e.stderr}")
        return None, None
    except FileNotFoundError:
        print(f"[ERROR] C executable not found at: {C_EXECUTABLE_PATH}")
        print("  Did you compile it using the Makefile?")
        return None, None
    except Exception as e:
        print(f"[ERROR] Failed to run or parse C inference for {json_file_path} ID {sample_id}: {e}")
        return None, None


# --- 主验证逻辑 ---
def main():
    global VAL_DIR, MODEL_INT8_PATH, C_EXECUTABLE_PATH
    parser = argparse.ArgumentParser(description="Validate C inference against Python INT8 model.")
    parser.add_argument("--val_dir", type=str, default=VAL_DIR, help="Directory containing validation JSON files.")
    parser.add_argument("--model_path", type=str, default=str(MODEL_INT8_PATH), help="Path to the trained INT8 model (.pt).")
    parser.add_argument("--c_exe", type=str, default=C_EXECUTABLE_PATH, help="Path to the compiled C inference executable.")
    parser.add_argument("--tolerance", type=float, default=1e-4, help="Absolute tolerance for float comparison.")
    parser.add_argument("--limit", type=int, default=None, help="Limit validation to the first N samples (optional).")
    parser.add_argument("--dump-intermediates", action='store_true', help="Dump PyTorch per-layer intermediate activations to JSON for first N samples.")
    parser.add_argument("--dump-dir", type=str, default="./intermediates", help="Directory to write intermediate JSON files.")
    args = parser.parse_args()

    # 更新全局变量
    
    VAL_DIR = args.val_dir
    MODEL_INT8_PATH = Path(args.model_path)
    C_EXECUTABLE_PATH = args.c_exe

    if not MODEL_INT8_PATH.exists():
        print(f"[ERROR] INT8 Model file not found: {MODEL_INT8_PATH}")
        return
    if not Path(C_EXECUTABLE_PATH).exists():
         print(f"[ERROR] C executable not found: {C_EXECUTABLE_PATH}. Compile it first.")
         return

    dump_intermediates = args.dump_intermediates
    dump_dir = Path(args.dump_dir)
    if dump_intermediates:
        dump_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载 PyTorch INT8 模型
    print(f"[INFO] Loading PyTorch INT8 model from {MODEL_INT8_PATH}...")
    try:
        # 需要模型定义来加载 state_dict
        model_def = SparseCompensationMLP(k_points=K_SPARSE_POINTS, width=WIDTH, hidden=16)
        model_def.fuse_model()
        model_load = model_def.cpu()
        model_load.qconfig = tq.get_default_qat_qconfig(QENGINE)
        tq.prepare_qat(model_load, inplace=True) # Prepare structure
        model_load.eval()
        model_int8 = tq.convert(model_load, inplace=False) # Convert to INT8 structure
        model_int8.eval()
        state_dict = torch.load(MODEL_INT8_PATH, map_location='cpu')
        model_int8.load_state_dict(state_dict)
        print("[INFO] PyTorch model loaded successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to load PyTorch model: {e}")
        return

    # Helper: run model step-by-step and capture per-module activations (useful for debugging)
    def capture_intermediates(model_q, x_tensor, meta_tensor):
        activations = {}
        # Build input batch
        h = torch.cat([x_tensor, meta_tensor], dim=1)
        # Quantize input via model's QuantStub if present
        if hasattr(model_q, 'quant_in'):
            try:
                h = model_q.quant_in(h)
            except Exception:
                # quant_in may be a no-op in some converted models
                pass
        # Capture both dequantized floats and integer representations when quantized
        if hasattr(h, 'is_quantized') and h.is_quantized:
            h_deq = h.dequantize()
            try:
                h_int = h.int_repr().detach().cpu().numpy().tolist()
            except Exception:
                h_int = None
            activations['after_quant_in_q_int'] = h_int
            activations['after_quant_in_q_scale'] = float(h.q_scale())
            activations['after_quant_in_q_zero_point'] = int(h.q_zero_point())
        else:
            h_deq = h
            activations['after_quant_in_q_int'] = None
            activations['after_quant_in_q_scale'] = None
            activations['after_quant_in_q_zero_point'] = None
        activations['after_quant_in'] = h_deq.detach().cpu().numpy().tolist()
        # Step through sequential net modules
        idx = 0
        for m in model_q.net:
            h = m(h)
            # For each layer capture dequantized floats and integer repr if quantized
            if hasattr(h, 'is_quantized') and h.is_quantized:
                h_deq = h.dequantize()
                try:
                    h_int = h.int_repr().detach().cpu().numpy().tolist()
                except Exception:
                    h_int = None
                activations[f'net_{idx}_q_int'] = h_int
                activations[f'net_{idx}_q_scale'] = float(h.q_scale())
                activations[f'net_{idx}_q_zero_point'] = int(h.q_zero_point())
            else:
                h_deq = h
                activations[f'net_{idx}_q_int'] = None
                activations[f'net_{idx}_q_scale'] = None
                activations[f'net_{idx}_q_zero_point'] = None
            activations[f'net_{idx}'] = h_deq.detach().cpu().numpy().tolist()
            idx += 1
        # Dequant_out
        if hasattr(model_q, 'dequant_out'):
            try:
                out = model_q.dequant_out(h)
            except Exception:
                out = h
        else:
            out = h
        if hasattr(out, 'is_quantized') and out.is_quantized:
            out_deq = out.dequantize()
            try:
                out_int = out.int_repr().detach().cpu().numpy().tolist()
            except Exception:
                out_int = None
            activations['after_dequant_out_q_int'] = out_int
            activations['after_dequant_out_q_scale'] = float(out.q_scale())
            activations['after_dequant_out_q_zero_point'] = int(out.q_zero_point())
        else:
            out_deq = out
            activations['after_dequant_out_q_int'] = None
            activations['after_dequant_out_q_scale'] = None
            activations['after_dequant_out_q_zero_point'] = None
        activations['after_dequant_out'] = out_deq.detach().cpu().numpy().tolist()
        # final tanh and scaling
        out_t = torch.tanh(out)
        out_t = torch.stack([out_t[:,0] * model_q.max_dx, out_t[:,1] * model_q.max_dy], dim=1)
        activations['final'] = out_t.detach().cpu().numpy().tolist()
        return activations

    def collect_weight_ints(model_q):
        wdict = {}
        for name, m in model_q.named_modules():
            # capture quantized linear weights/bias if present
            if isinstance(m, (nnq.Linear, nniq.LinearReLU)):
                try:
                    w = m.weight()
                    w_info = {}
                    w_info['qscheme'] = str(w.qscheme())
                    # per-tensor or per-channel
                    if w.qscheme() == torch.per_tensor_affine:
                        w_info['weight_int'] = w.int_repr().detach().cpu().numpy().tolist()
                        w_info['scale'] = float(w.q_scale())
                        w_info['zero_point'] = int(w.q_zero_point())
                    elif w.qscheme() == torch.per_channel_affine:
                        try:
                            w_info['weight_int'] = w.int_repr().detach().cpu().numpy().tolist()
                            w_info['scales'] = w.q_per_channel_scales().detach().cpu().numpy().tolist()
                            w_info['zero_points'] = w.q_per_channel_zero_points().detach().cpu().numpy().tolist()
                            w_info['axis'] = int(w.q_per_channel_axis())
                        except Exception:
                            w_info['weight_int'] = None
                    else:
                        w_info['weight_int'] = None

                    # bias (float) and integerized bias relative to activation/out scales if possible
                    try:
                        b = m.bias
                        if b is not None:
                            b_fp = b.detach().cpu().numpy().tolist()
                            w_info['bias_float'] = b_fp
                        else:
                            w_info['bias_float'] = None
                    except Exception:
                        w_info['bias_float'] = None

                    wdict[name] = w_info
                except Exception:
                    continue
        return wdict

    # 2. 加载验证数据 (使用原始 TouchJSONDataset 逐个加载)
    print(f"[INFO] Loading validation data from {VAL_DIR}...")
    try:
         # 只加载原始数据，不进行 Dataset 转换
         base_dataset = TouchJSONDataset(VAL_DIR)
         if not base_dataset.samples:
              print("[ERROR] No validation samples found.")
              return
    except Exception as e:
         print(f"[ERROR] Failed to load validation data: {e}")
         return

    # 3. 逐样本验证
    mismatches = 0
    total_samples = len(base_dataset)
    limit = args.limit if args.limit is not None else total_samples
    limit = min(limit, total_samples)

    print(f"[INFO] Starting validation for {limit} samples...")
    for i in range(limit):
        sample_info = base_dataset.samples[i]
        sample_id = sample_info['id']
        json_filename = sample_info['file']
        json_file_path = Path(VAL_DIR) / json_filename

        print(f"--- Validating sample {i+1}/{limit} (File: {json_filename}, ID: {sample_id}) ---")

        # a) Python 推理
        try:
            xc_py, yc_py, xr_py, yr_py, x_in_tensor, meta_tensor = run_python_inference(model_int8, sample_info)
            print(f"  Python Result: xc={xc_py:.6f}, yc={yc_py:.6f} (Baseline: xr={xr_py:.6f}, yr={yr_py:.6f})")
            if dump_intermediates:
                # Capture and dump per-layer activations for this sample
                acts = capture_intermediates(model_int8, x_in_tensor, meta_tensor)
                # include baseline CoG from Python preprocessing for direct comparison
                acts['baseline_cog'] = [float(xr_py), float(yr_py)]
                fname = dump_dir / f"intermediates_{json_filename}_id{sample_id}_py.json"
                with open(fname, 'w') as f:
                    json.dump(acts, f)
                print(f"  [INFO] Dumped intermediates -> {fname}")
        except Exception as e:
            print(f"  [ERROR] Python inference failed: {e}")
            mismatches += 1
            continue # 跳过此样本

        # b) C 推理 (可选写入中间结果到 dump_dir)
        dump_fname = None
        if dump_intermediates:
            dump_fname = dump_dir / f"intermediates_{json_filename}_id{sample_id}_c.json"
        xc_c, yc_c = run_c_inference(json_file_path, sample_id, dump_path=str(dump_fname) if dump_fname else None)
        if xc_c is None or yc_c is None:
            mismatches += 1
            continue # C 推理失败，跳过比较

        print(f"  C Result:      xc={xc_c:.6f}, yc={yc_c:.6f}")

        # c) 比较结果
        diff_x = abs(xc_py - xc_c)
        diff_y = abs(yc_py - yc_c)

        if diff_x > args.tolerance or diff_y > args.tolerance:
            print(f"  [MISMATCH DETECTED!] Diff X: {diff_x:.6f}, Diff Y: {diff_y:.6f}")
            mismatches += 1
        else:
            print("  [MATCH]")

    # 4. 总结报告
    print("\n--- Validation Summary ---")
    print(f"Total samples checked: {limit}")
    print(f"Mismatches found: {mismatches}")
    if mismatches == 0:
        print("Validation PASSED!")
    else:
        print("Validation FAILED!")

if __name__ == "__main__":
    main()
