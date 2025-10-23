import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import re
from pathlib import Path
import sys # For exit
import warnings
warnings.filterwarnings("ignore") # Suppress warnings

# =================================================================
# 1. 从 train_offset_QAT.py 复制必要的模型定义
# (必须与 train_offset_QAT.py 完全一致)
# =================================================================
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
         y = self.act(self.bn(y))
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

# =================================================================
# 2. 导出器辅助函数
# =================================================================

def format_c_array(name, data, dtype='int8_t', is_const=True):
# ... (existing format_c_array code) ...
    """将 numpy 数组格式化为 C 数组"""
    data_flat = data.flatten()
    num_elements = data_flat.size
    if num_elements == 0:
        print(f"[WARN] formatting empty C array: {name}")
        return f"const {dtype} {name}[1] = {{0}};" # Provide a dummy element?

    const_str = "const " if is_const else ""
    header = f"{const_str}{dtype} {name}[{num_elements}] = {{\n  "
    footer = "\n};"
    body_elements = []
    for x in data_flat:
        if dtype == 'float': body_elements.append(f"{x:.8e}f")
        elif 'int' in dtype: body_elements.append(str(int(x)))
        else: body_elements.append(str(x))
    max_line_len = 80
    formatted_body = ""
    current_line = ""
    for i, elem in enumerate(body_elements):
        potential_line = current_line + (", " if current_line else "") + elem
        if len(potential_line) > max_line_len and current_line:
            formatted_body += current_line + ",\n  "
            current_line = elem
        else: current_line = potential_line
        if i == num_elements - 1: formatted_body += current_line
    if not formatted_body.strip() and num_elements > 0: formatted_body = ", ".join(body_elements)
    return header + formatted_body + footer


def format_c_define(name, value):
# ... (existing format_c_define code) ...
    """格式化 C #define"""
    if isinstance(value, float):
        return f"#define {name} {value:.10e}f" if abs(value) < 1e-6 and value != 0.0 else f"#define {name} {value:.10f}f"
    elif isinstance(value, int):
        return f"#define {name} {value}"
    else:
        if hasattr(value, 'item'): item_val = value.item(); return format_c_define(name, item_val)
        return f"#define {name} {value}"


def get_requant_params(effective_scale):
# ... (existing get_requant_params code) ...
    """
    计算 TFLite 风格的 multiplier 和 shift (用于 no-int64 C 核)
    effective_scale = (in_scale * w_scale) / out_scale
    C 端计算: requantized = RoundingDivideByPOT( SaturatingRoundingDoublingHighMul(acc * 2^left_shift, multiplier), right_shift)
    或者简化版: result = ((int64_t)acc * multiplier + rounding_offset) >> shift
    我们导出适用于简化版的 multiplier 和 shift (需要 C 端模拟 64 位乘法或使用 TFLite 的 MultiplyByQuantizedMultiplier)

    此函数计算 M 和 shift，使得 effective_scale ≈ M / 2^shift (M < 2^31)
    """
    if isinstance(effective_scale, torch.Tensor):
         multipliers = []; shifts = []
         effective_scale_cpu = effective_scale.cpu()
         # --- Add Debug Print ---
         is_block1_pw_scales = (effective_scale.numel() == 4) # Heuristic check

         for i, scale_val in enumerate(effective_scale_cpu):
             # Print for the specific layer if needed
             if is_block1_pw_scales and i < 4: # Print first few channels for B1PW
                 print(f"      get_requant_params[B1PW ch={i}]: eff_scale={scale_val.item():.6e}")
             m, s = get_requant_params(scale_val.item()) # Recursive call with float
             if is_block1_pw_scales and i < 4:
                 print(f"        -> mult={m}, shift={s}")
             multipliers.append(m); shifts.append(s)
         return np.array(multipliers, dtype=np.int32), np.array(shifts, dtype=np.int32)

    # --- Original logic for single float scale ---
    if not isinstance(effective_scale, (float, np.float32, np.float64)):
         if hasattr(effective_scale, 'item'): effective_scale = effective_scale.item()
         else: print(f"[ERROR] Invalid type for effective_scale: {type(effective_scale)}"); return 0,0
    if effective_scale <= 0.0: # Check for non-positive scale
        print(f"[WARN] Effective scale is non-positive ({effective_scale:.6e}). Returning 0 multiplier/shift.")
        return 0, 0 # Multiplier and shift
    significand, exponent = np.frexp(effective_scale)
    if significand == 0.0 and effective_scale != 0.0:
         print(f"[WARN] np.frexp returned zero significand for non-zero scale {effective_scale}. Adjusting.")
         scale_adj = effective_scale; adj_exponent = 0
         while abs(scale_adj) < 1e-9 and adj_exponent < 60: scale_adj *= 2.0; adj_exponent += 1
         if abs(scale_adj) >= 1e-9: significand, exponent = np.frexp(scale_adj); exponent -= adj_exponent; print(f"  Adjusted: signif={significand}, exp={exponent}")
         else: print(f"[ERROR] Could not recover significand for scale {effective_scale}."); return 0, 0
    significand_q31 = int(round(significand * (1 << 31)))
    if significand_q31 == (1 << 31): significand_q31 //= 2; exponent += 1
    elif significand_q31 <= 0 and effective_scale > 0.0: # Check if rounding yielded non-positive for positive scale
        print(f"[WARN] Effective scale {effective_scale:.6e} resulted in zero or negative multiplier ({significand_q31}) *after* np.frexp. Check ranges.")
        # Try adjusting exponent? This requires careful thought, TFLite might handle differently.
        # Fallback to 0,0 is safer if the scale is extremely small or problematic.
        if effective_scale < 1e-9: # If scale is tiny, maybe 0,0 is acceptable
            return 0, 0
        else: # If scale is reasonable, this indicates a potential issue.
             print(f"[ERROR] Unexpected non-positive multiplier for positive scale. Returning 0,0.")
             return 0,0

    multiplier = significand_q31; shift = 31 - exponent
    shift = max(0, min(60, shift)) # Clamp shift
    # Add check for zero multiplier resulting from clamping/rounding
    if multiplier <= 0 and effective_scale > 0:
        print(f"[WARN] Final multiplier is non-positive ({multiplier}) for positive effective scale {effective_scale:.6e}. Returning 0,0.")
        return 0,0

    return multiplier, shift


def export_model_data(model_int8_path, float_model_def, out_dir_str):
# ... (existing code) ...
    """主导出函数"""
    out_dir = Path(out_dir_str)
    out_dir.mkdir(exist_ok=True)

    h_meta_file = out_dir / "model_meta.h"
    h_weights_file = out_dir / "weights.h"
    c_weights_file = out_dir / "weights.c"

    # --- 1. 重建 QAT 图并加载权重 ---
    print(f"Loading QAT state_dict from {model_int8_path}...")
    from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig
    import torch.ao.quantization.quantize_fx as qfx
    qengine = "fbgemm"; torch.backends.quantized.engine = qengine
    qconfig_mapping = QConfigMapping().set_global(get_default_qat_qconfig(qengine))
    ex_x = torch.zeros(1, 2, 3, 7, dtype=torch.float32)
    ex_meta = torch.zeros(1, 2, dtype=torch.float32)
    example_inputs = (ex_x, ex_meta)
    model = float_model_def.eval()
    prepared_model = qfx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
    converted_model = qfx.convert_fx(prepared_model)
    print("="*30); print("Inspecting converted_model:"); print(converted_model); print("="*30)
    converted_model.load_state_dict(torch.load(model_int8_path, map_location="cpu", weights_only=False))
    converted_model.eval()
    print("Model graph loaded and converted.")

    # --- 2. 准备 C 文件 ---
    h_meta = ["#ifndef MODEL_META_H", "#define MODEL_META_H", "\n#include <stdint.h>"]
    h_weights = ["#ifndef WEIGHTS_H", "#define WEIGHTS_H", "\n#include <stdint.h>"]
    c_weights = [f'#include "{h_meta_file.name}"', f'#include "{h_weights_file.name}"']

    # --- 3. 提取所有模块的参数 ---
    q_params = {}
    try:
        q_params['quant_x'] = (converted_model.block1_dw_input_scale_0.item(), converted_model.block1_dw_input_zero_point_0.item())
        print(f"Found Input 'x' q_params: {q_params['quant_x']}")
        q_params['quant_meta'] = (converted_model._input_scale_0.item(), converted_model._input_zero_point_0.item())
        print(f"Found Input 'meta' q_params: {q_params['quant_meta']}")
        output_mod = converted_model.head.get_submodule('2')
        q_params['output'] = (output_mod.scale, output_mod.zero_point)
        print(f"Found Output (pre-tanh) q_params: {q_params['output']}")
        if hasattr(converted_model, '_scale_0') and hasattr(converted_model, '_zero_point_0'):
             q_params['gap_out_requant'] = (converted_model._scale_0.item(), converted_model._zero_point_0.item())
             print(f"Found Gap Requant/Cat Input q_params (_scale_0): {q_params['gap_out_requant']}")
             if abs(q_params['gap_out_requant'][0] - q_params['quant_meta'][0]) > 1e-9 or \
                q_params['gap_out_requant'][1] != q_params['quant_meta'][1]:
                 print(f"[ERROR] Mismatched quantization params for torch.cat!") # ... (rest of error message) ...
             q_params['cat_out'] = q_params['gap_out_requant']
        else:
             print("[ERROR] Cannot find _scale_0/_zero_point_0 on converted model for GAP requant/CAT input."); print("[ERROR] Export might be incorrect. Using fallback (0.1, 0).")
             q_params['gap_out_requant'] = (0.1, 0); q_params['cat_out'] = (0.1, 0)
    except AttributeError as e: print(f"\n[错误] 查找根 QParams 失败: {e}"); raise

    # --- 4. 写入元数据 (model_meta.h) ---
    h_meta.append("\n// --- Model Shapes ---") # ... (rest of shape defines) ...
    h_meta.append(format_c_define("MODEL_INPUT_X_SHAPE_SIZE", "(1*2*3*7)"))
    h_meta.append(format_c_define("MODEL_INPUT_META_SHAPE_SIZE", "(1*2)"))
    h_meta.append(format_c_define("MODEL_OUTPUT_SHAPE_SIZE", "(1*2)"))
    h_meta.append("\n// --- Final Post-Processing Scales ---") # ... (rest of final scale defines) ...
    h_meta.append(format_c_define("MODEL_FINAL_SCALE_DX", float_model_def.max_dx))
    h_meta.append(format_c_define("MODEL_FINAL_SCALE_DY", float_model_def.max_dy))
    h_meta.append("\n// --- Activation Quantization Params ---") # ... (rest of input defines) ...
    h_meta.append(format_c_define("MODEL_INPUT_X_SCALE", q_params['quant_x'][0]))
    h_meta.append(format_c_define("MODEL_INPUT_X_ZERO_POINT", q_params['quant_x'][1]))
    h_meta.append(format_c_define("MODEL_INPUT_META_SCALE", q_params['quant_meta'][0]))
    h_meta.append(format_c_define("MODEL_INPUT_META_ZERO_POINT", q_params['quant_meta'][1]))

    mod_dict = dict(converted_model.named_modules())
    def find_output_q_params(mod_path): # ... (existing find_output_q_params, should be fine) ...
        # 辅助函数，在转换后的图中查找 scale/zp
        if mod_path == "head.0":
            head0_mod = mod_dict.get("head.0")
            if head0_mod and hasattr(head0_mod, 'scale'): return (head0_mod.scale, head0_mod.zero_point)
            else: relu_mod = mod_dict.get("head.1");
            if relu_mod and hasattr(relu_mod, 'activation_post_process'): return (relu_mod.activation_post_process.scale, relu_mod.activation_post_process.zero_point)
        if mod_path == "gap": return q_params.get('gap_out_requant', (0.1, 0))
        mod = mod_dict.get(mod_path)
        if mod and hasattr(mod, 'scale') and hasattr(mod, 'zero_point'): return (mod.scale, mod.zero_point)
        elif mod and hasattr(mod, 'activation_post_process'): return (mod.activation_post_process.scale, mod.activation_post_process.zero_point)
        print(f"[WARN] 无法找到 {mod_path} 的 output q_params."); print(f"[WARN] 检查 FX 图命名和融合状态."); return 0.1, 0

    def find_input_q_params(mod_path, q_params_dict): # ... (existing find_input_q_params, should be fine) ...
        # 辅助函数，查找模块的 *输入* q_params
        if mod_path == "block1.dw": return q_params_dict.get('quant_x', (0.1, 0))
        if mod_path == "block1.pw": return q_params_dict.get('block1_dw_out', (0.1, 0))
        if mod_path == "block2.dw": return q_params_dict.get('block1_out', (0.1, 0))
        if mod_path == "block2.pw": return q_params_dict.get('block2_dw_out', (0.1, 0))
        if mod_path == "gap":       return q_params_dict.get('block2_out', (0.1, 0))
        if mod_path == "head.0":    return q_params_dict.get('cat_out', (0.1, 0))
        if mod_path == "head.2":    return q_params_dict.get('head0_out', (0.1, 0))
        print(f"[WARN] 无法确定 {mod_path} 的 *input* q_params using simple lookup."); return 0.1, 0

    print("Extracting intermediate QParams...") # ... (rest of intermediate QParam extraction and printing) ...
    q_params["block1_dw_out"] = find_output_q_params("block1.dw")
    q_params["block1_out"] = find_output_q_params("block1.pw")
    q_params["block2_dw_out"] = find_output_q_params("block2.dw")
    q_params["block2_out"] = find_output_q_params("block2.pw")
    q_params["gap_out"] = find_output_q_params("gap")
    print(f"  block1_dw_out: {q_params.get('block1_dw_out', 'Not Found')}") # ... (rest of prints) ...
    print(f"  block1_out: {q_params.get('block1_out', 'Not Found')}")
    print(f"  block2_dw_out: {q_params.get('block2_dw_out', 'Not Found')}")
    print(f"  block2_out: {q_params.get('block2_out', 'Not Found')}")
    print(f"  gap_out (re-quant): {q_params.get('gap_out', 'Not Found')}")
    q_params["cat_out"] = find_input_q_params("head.0", q_params)
    print(f"  cat_out (head.0 input): {q_params['cat_out']}")
    q_params["head0_out"] = find_output_q_params("head.0")
    print(f"  head0_out: {q_params.get('head0_out', 'Not Found')}")

    h_meta.append(format_c_define("MODEL_BLOCK1_DW_OUT_SCALE", q_params.get("block1_dw_out", [0.1])[0])) # ... (rest of defines) ...
    h_meta.append(format_c_define("MODEL_BLOCK1_DW_OUT_ZERO_POINT", q_params.get("block1_dw_out", [0.1, 0])[1]))
    h_meta.append(format_c_define("MODEL_BLOCK1_OUT_SCALE", q_params.get("block1_out", [0.1])[0]))
    h_meta.append(format_c_define("MODEL_BLOCK1_OUT_ZERO_POINT", q_params.get("block1_out", [0.1, 0])[1]))
    h_meta.append(format_c_define("MODEL_BLOCK2_DW_OUT_SCALE", q_params.get("block2_dw_out", [0.1])[0]))
    h_meta.append(format_c_define("MODEL_BLOCK2_DW_OUT_ZERO_POINT", q_params.get("block2_dw_out", [0.1, 0])[1]))
    h_meta.append(format_c_define("MODEL_BLOCK2_OUT_SCALE", q_params.get("block2_out", [0.1])[0]))
    h_meta.append(format_c_define("MODEL_BLOCK2_OUT_ZERO_POINT", q_params.get("block2_out", [0.1, 0])[1]))
    h_meta.append(format_c_define("MODEL_GAP_OUT_SCALE", q_params.get("gap_out", [0.1])[0]))
    h_meta.append(format_c_define("MODEL_GAP_OUT_ZERO_POINT", q_params.get("gap_out", [0.1, 0])[1]))
    h_meta.append(format_c_define("MODEL_CAT_OUT_SCALE", q_params.get("cat_out", [0.1])[0]))
    h_meta.append(format_c_define("MODEL_CAT_OUT_ZERO_POINT", q_params.get("cat_out", [0.1, 0])[1]))
    h_meta.append(format_c_define("MODEL_HEAD0_OUT_SCALE", q_params.get("head0_out", [0.1])[0]))
    h_meta.append(format_c_define("MODEL_HEAD0_OUT_ZERO_POINT", q_params.get("head0_out", [0.1, 0])[1]))
    h_meta.append(format_c_define("MODEL_OUTPUT_SCALE", q_params.get("output", [0.1])[0]))
    h_meta.append(format_c_define("MODEL_OUTPUT_ZERO_POINT", q_params.get("output", [0.1, 0])[1]))

    # --- 5. 提取权重, 计算乘子 (移除偏置融合) ---
    h_meta.append("\n// --- Layer Requantization Params (Per-Tensor/Per-Channel) ---")

    for name, mod in converted_model.named_modules():
        if isinstance(mod, (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)):
            print(f"Exporting layer: {name}")
            c_name = name.replace(".", "_")
            is_per_channel = False; num_channels = 0

            # A. 获取权重和偏置(原始)
            weight_q = mod.weight(); w_s8 = weight_q.int_repr()
            bias = mod.bias()
            original_bias_s32 = torch.zeros(w_s8.shape[0], dtype=torch.int32)
            if bias is not None:
                if isinstance(bias, torch.Tensor) and bias.is_quantized: original_bias_s32 = bias.int_repr()
                elif isinstance(bias, torch.Tensor):
                    in_s, _ = find_input_q_params(name, q_params)
                    w_scales_for_bias = weight_q.q_per_channel_scales() if weight_q.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric) else weight_q.q_scale()
                    _in_s_float = in_s.item() if hasattr(in_s, 'item') else in_s
                    bias_scales = None
                    if isinstance(w_scales_for_bias, torch.Tensor): bias_scales = _in_s_float * w_scales_for_bias
                    else: _w_scales_float = w_scales_for_bias.item() if hasattr(w_scales_for_bias, 'item') else w_scales_for_bias; bias_scales = _in_s_float * _w_scales_float
                    if bias_scales is None or (isinstance(bias_scales, torch.Tensor) and torch.any(bias_scales <= 0.0)) or (isinstance(bias_scales, float) and bias_scales <= 0.0): # Check non-positive
                         print(f"[ERROR] Non-positive bias scale detected for {name}. Cannot quantize bias.") # Updated check
                    else:
                        if isinstance(bias_scales, torch.Tensor) and bias_scales.shape != bias.shape:
                            if bias_scales.numel() == bias.numel(): bias_scales = bias_scales.reshape(bias.shape)
                            else: print(f"[ERROR] Mismatched shapes for bias quantization: bias={bias.shape}, scale={bias_scales.shape}")
                        # Use torch.clamp before casting to handle potential out-of-range values
                        original_bias_s32 = torch.clamp(torch.round(bias / bias_scales), torch.iinfo(torch.int32).min, torch.iinfo(torch.int32).max).to(torch.int32)
                else: print(f"[WARN] Unexpected bias type for {name}: {type(bias)}")


            # B. 获取权重 QParams
            w_zp = 0
            if weight_q.qscheme() in (torch.per_channel_affine, torch.per_channel_symmetric):
                is_per_channel = True; w_scales = weight_q.q_per_channel_scales()
                num_channels = w_scales.numel(); print(f"  Layer {name}: Per-channel weights detected ({num_channels} channels).")
                _w_zps_check = weight_q.q_per_channel_zero_points();
                if not torch.all(_w_zps_check == 0): print(f"[WARN] Layer {name} has non-zero per-channel weight zero points. C kernel MUST handle this.")
            elif weight_q.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                w_scales = weight_q.q_scale(); w_zp = weight_q.q_zero_point()
                num_channels = 1;
                if w_zp != 0: print(f"[WARN] Layer {name} has non-zero per-tensor weight zero-point ({w_zp}). C kernel MUST handle this.")
            else: raise RuntimeError(f"Unsupported weight qscheme {weight_q.qscheme()} for layer {name}")


            # C. 获取输入/输出 QParams
            in_s, in_z = find_input_q_params(name, q_params)
            out_s, out_z = find_output_q_params(name)
            _in_s_float = in_s.item() if hasattr(in_s, 'item') else in_s
            _in_z_int = in_z.item() if hasattr(in_z, 'item') else in_z
            _out_s_float = out_s.item() if hasattr(out_s, 'item') else out_s
            _out_z_int = out_z.item() if hasattr(out_z, 'item') else out_z

            # --- Add Debug Print for Block1 PW Scales ---
            if name == "block1.pw":
                 print(f"    DEBUG Block1 PW Params for Requant:")
                 print(f"      Input Scale (block1_dw_out): {_in_s_float:.8e}, ZP: {_in_z_int}")
                 print(f"      Output Scale (block1_pw):    {_out_s_float:.8e}, ZP: {_out_z_int}")
                 if is_per_channel:
                      print(f"      Weight Scales (per-chan): [{w_scales.min().item():.8e} .. {w_scales.max().item():.8e}]")
                 else:
                      _w_scales_float = w_scales.item() if hasattr(w_scales, 'item') else w_scales
                      print(f"      Weight Scale (per-tensor): {_w_scales_float:.8e}")


            # D. 计算重量化乘子 (Multiplier/Shift)
            if _out_s_float <= 0.0: # Check non-positive
                print(f"[ERROR] Output scale for layer {name} is non-positive ({_out_s_float:.6e}). Cannot compute requant params.")
                # Assign default/zero params based on whether per-channel
                if is_per_channel: effective_scale = torch.zeros_like(w_scales)
                else: effective_scale = 0.0
            else:
                effective_scale = (_in_s_float * w_scales) / _out_s_float # w_scales can be tensor or float

             # --- Add Debug Print for Block1 PW Effective Scale ---
            if name == "block1.pw":
                 print(f"    DEBUG Block1 PW Effective Scale Calculation:")
                 if isinstance(effective_scale, torch.Tensor):
                      print(f"      Effective Scales (per-chan): [{effective_scale.min().item():.8e} .. {effective_scale.max().item():.8e}]")
                      print(f"      Effective Scales (all): {effective_scale.cpu().numpy()}") # Print all values
                 else:
                      print(f"      Effective Scale (per-tensor): {effective_scale:.8e}")


            multiplier, shift = get_requant_params(effective_scale) # Handles tensor or float

            # --- Add Debug Print for Block1 PW Multiplier/Shift ---
            if name == "block1.pw":
                 print(f"    DEBUG Block1 PW Multiplier/Shift Results:")
                 if isinstance(multiplier, np.ndarray):
                      print(f"      Multipliers: [{multiplier.min()} .. {multiplier.max()}]")
                      print(f"      Multipliers (all): {multiplier}")
                      print(f"      Shifts:      [{shift.min()} .. {shift.max()}]")
                      print(f"      Shifts (all):      {shift}")
                 else:
                      print(f"      Multiplier: {multiplier}")
                      print(f"      Shift:      {shift}")


            # E. 保存 C 代码 (Weights, Original Bias, Requant Params)
            h_weights.append(f"extern const int8_t g_{c_name}_weight[{w_s8.numel()}];")
            c_weights.append(format_c_array(f"g_{c_name}_weight", w_s8.cpu().numpy(), 'int8_t'))
            h_weights.append(f"extern const int32_t g_{c_name}_bias[{original_bias_s32.numel()}];")
            c_weights.append(format_c_array(f"g_{c_name}_bias", original_bias_s32.cpu().numpy(), 'int32_t'))

            h_meta.append(f"\n// --- Requantization Params for Layer: {name} ---") # ... (rest of requant param export is the same) ...
            if is_per_channel:
                 h_meta.append(f"#define MODEL_{c_name.upper()}_IS_PER_CHANNEL 1"); h_meta.append(f"#define MODEL_{c_name.upper()}_NUM_CHANNELS {num_channels}")
                 h_weights.append(f"extern const int32_t g_{c_name}_multiplier[{num_channels}];"); c_weights.append(format_c_array(f"g_{c_name}_multiplier", multiplier, 'int32_t'))
                 h_weights.append(f"extern const int32_t g_{c_name}_shift[{num_channels}];"); c_weights.append(format_c_array(f"g_{c_name}_shift", shift, 'int32_t'))
            else:
                 h_meta.append(f"#define MODEL_{c_name.upper()}_IS_PER_CHANNEL 0")
                 h_meta.append(format_c_define(f"MODEL_{c_name.upper()}_OUT_MULTIPLIER", multiplier)); h_meta.append(format_c_define(f"MODEL_{c_name.upper()}_OUT_SHIFT", shift))

    # --- 6. 处理 GAP (GlobalAvgPool) ---
    in_s_gap, in_z_gap = q_params.get("block2_out", (0.1, 0)) # ... (rest of GAP logic is the same) ...
    out_s_gap, out_z_gap = q_params.get("gap_out", (0.1, 0))
    pool_size = 3 * 7
    if hasattr(in_s_gap, 'item'): in_s_gap = in_s_gap.item()
    if hasattr(out_s_gap, 'item'): out_s_gap = out_s_gap.item()
    if out_s_gap <= 0.0 or pool_size == 0: # Check non-positive
        print(f"[ERROR] Invalid scale (<=0) or pool size for GAP..."); effective_scale_gap = 0.0
    else: effective_scale_gap = in_s_gap / (out_s_gap * pool_size)
    multiplier_gap, shift_gap = get_requant_params(effective_scale_gap)
    h_meta.append(f"\n// --- Requantization Params for Layer: GAP ---")
    h_meta.append(f"// Note: Maps block2_out scale to the re-quantized scale before cat (_scale_0)")
    h_meta.append(format_c_define(f"MODEL_GAP_OUT_MULTIPLIER", multiplier_gap))
    h_meta.append(format_c_define(f"MODEL_GAP_OUT_SHIFT", shift_gap))
    h_meta.append(format_c_define(f"MODEL_GAP_OUT_ZERO_POINT", q_params.get("gap_out", [0.1, 0])[1])) # GAP output ZP based on re-quant params

    # --- 7. 写入文件 ---
    h_meta.append("\n#endif // MODEL_META_H") # ... (rest of file writing is the same) ...
    h_weights.append("\n#endif // WEIGHTS_H")
    with open(h_meta_file, 'w') as f: f.write("\n".join(h_meta))
    with open(h_weights_file, 'w') as f: f.write("\n".join(h_weights))
    with open(c_weights_file, 'w') as f: f.write("\n\n".join(c_weights))
    print(f"\nSuccessfully exported weights to {out_dir_str}/")

# =================================================================
# 3. QAT 模型重构 (用于 verify_with_python.py)
# =================================================================
def get_qat_converted_model(model_int8_path):
# ... (existing get_qat_converted_model code) ...
    """
    辅助函数，用于重建 QAT 图并加载 int8 权重。
    供 'verify_with_python.py' 使用。
    """
    from torch.ao.quantization import QConfigMapping, get_default_qat_qconfig
    import torch.ao.quantization.quantize_fx as qfx
    float_model_def = OffsetNetWithMeta(in_c=2, width=4)
    qengine = "fbgemm"; torch.backends.quantized.engine = qengine
    qconfig_mapping = QConfigMapping().set_global(get_default_qat_qconfig(qengine))
    ex_x = torch.zeros(1, 2, 3, 7, dtype=torch.float32)
    ex_meta = torch.zeros(1, 2, dtype=torch.float32)
    example_inputs = (ex_x, ex_meta)
    model = float_model_def.eval()
    prepared_model = qfx.prepare_qat_fx(model, qconfig_mapping, example_inputs)
    converted_model = qfx.convert_fx(prepared_model)
    converted_model.load_state_dict(torch.load(model_int8_path, map_location="cpu", weights_only=False))
    converted_model.eval()
    return converted_model

# =================================================================
# 4. Main
# =================================================================
if __name__ == "__main__":
# ... (existing main code) ...
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-in", type=str, default="best_int8.pt", help="Input INT8 state_dict (best_int8.pt)")
    parser.add_argument("--out-dir", type=str, default="src", help="Output directory for C files (src/)")
    args = parser.parse_args()
    if not os.path.exists(args.model_in): print(f"错误: 模型文件未找到 '{args.model_in}'"); print("请将 'best_int8.pt' 复制到项目根目录。")
    else: model_def = OffsetNetWithMeta(in_c=2, width=4); export_model_data(args.model_in, model_def, args.out_dir)


