# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.ao.quantization as tq
import torch.nn.quantized as nnq
import torch.nn.intrinsic.quantized as nniq
from pathlib import Path
import argparse
import json # Although not directly used in export_weights_for_c, might be needed if qparams are loaded here later
import traceback # Added for better error reporting

# =========================
# 必要配置 (从训练脚本复制，确保一致)
# =========================
K_SPARSE_POINTS = 8
WIDTH           = 8
QENGINE         = "fbgemm" # 确保与训练时使用的引擎一致

# C 代码中的 MLP 定义 (必须与 C 代码和训练脚本一致)
C_MLP_INPUT_SIZE    = K_SPARSE_POINTS * 3 + 2 # 26
C_MLP_HIDDEN1_SIZE  = WIDTH * 2 * 2         # 32 (对应 net.0 输出)
C_MLP_HIDDEN2_SIZE  = WIDTH * 2             # 16 (对应 net.2 输出)
C_MLP_OUTPUT_SIZE   = 2                     # (对应 net.4 输出)

# Y 轴处理的最大值 (从训练脚本复制)
MAX_DY_ABS = 0.25
# 注意: max_dx 在模型定义中硬编码为 4.0，如果训练脚本修改了，这里也要改

# =========================
# 模型定义 (从训练脚本复制)
# =========================
class SparseCompensationMLP(nn.Module):
    """(复制自训练脚本) MLP + QuantStub/DeQuantStub"""
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

        self.max_dx, self.max_dy = max_dx, max_dy # max_dx is currently hardcoded

    def forward(self, x, meta):
        # Forward pass definition is needed for model structure but not for weight export
        h = torch.cat([x, meta], dim=1)
        h = self.quant_in(h)
        h = self.net(h)
        out = self.dequant_out(h)
        out = torch.tanh(out)
        out = torch.stack([out[:, 0] * self.max_dx, out[:, 1] * self.max_dy], dim=1)
        return out

    def fuse_model(self):
        """(复制自训练脚本) 融合 (Linear, ReLU) 对"""
        if self.net:
            torch.ao.quantization.fuse_modules(
                self.net, [('0', '1'), ('2', '3')], inplace=True
            )

# =========================
# 权重导出函数 (修改为生成 .h 和 .c)
# =========================
def export_weights_for_c(model_int8_path: Path, output_h_path: Path, output_c_path: Path):
    """加载 INT8 模型并将其权重和偏置导出为 C 头文件 (.h) 和源文件 (.c)"""
    print(f"[INFO] Exporting weights from {model_int8_path} to {output_h_path} and {output_c_path}...")
    try:
        # 1. 加载 INT8 模型状态字典 (需要模型定义)
        model_def = SparseCompensationMLP(k_points=K_SPARSE_POINTS, width=WIDTH, hidden=16)
        model_def.fuse_model()
        model_load = model_def.cpu()
        model_load.qconfig = tq.get_default_qat_qconfig(QENGINE)
        tq.prepare_qat(model_load, inplace=True)
        model_load.eval()
        quant_model_struct = tq.convert(model_load, inplace=False)
        quant_model_struct.eval()

        state_dict = torch.load(model_int8_path, map_location='cpu')
        quant_model_struct.load_state_dict(state_dict)

        # 2. 准备 C 头文件 (.h) 和源文件 (.c) 内容
        h_code = []
        c_code = []

        # --- .h 文件内容 ---
        h_code.append(f"#ifndef MODEL_WEIGHTS_H_{output_h_path.stem.upper()}") # 使用文件名创建唯一宏
        h_code.append(f"#define MODEL_WEIGHTS_H_{output_h_path.stem.upper()}")
        h_code.append("\n#include <stdint.h>")
        h_code.append("\n// --- Quantized Model Weights and Biases (Extern Declarations) ---")
        h_code.append("// Extracted from: " + str(model_int8_path.name))
        h_code.append("// Model Structure: MLP Input={} -> {} -> {} -> {}".format(
            C_MLP_INPUT_SIZE, C_MLP_HIDDEN1_SIZE, C_MLP_HIDDEN2_SIZE, C_MLP_OUTPUT_SIZE
        ))

        # --- .c 文件内容 ---
        c_code.append(f'#include "{output_h_path.name}"') # 包含对应的头文件
        c_code.append("\n// --- Quantized Model Weights and Biases (Definitions) ---")

        # 3. 遍历 state_dict 并格式化权重/偏置
        found_weights = {}
        # (修改) 显式遍历模型层以查找 _packed_params
        for layer_name, layer_module in quant_model_struct.named_modules():
            # 我们只关心量化线性层或融合后的量化线性层+ReLU
            is_quantized_linear = isinstance(layer_module, (nnq.Linear, nniq.LinearReLU))

            if is_quantized_linear and hasattr(layer_module, '_packed_params'):
                print(f"[INFO] Processing packed params for layer: {layer_name}")
                packed_params = layer_module._packed_params
                # packed_params._weight_bias() 返回 (quantized_weight, bias)
                weight_qtensor, bias_tensor = packed_params._weight_bias()

                # --- 处理权重 ---
                if weight_qtensor is not None:
                    weight_name_c = layer_name.replace('.', '_') + "_weight"
                    # (修改) 添加 .detach()
                    weight_int8 = weight_qtensor.detach().int_repr().numpy().flatten()
                    c_type = "const int8_t"
                    data_list = ",\n  ".join(map(str, weight_int8))
                    size = len(weight_int8)

                    expected_size = 0
                    if layer_name == "net.0": expected_size = C_MLP_HIDDEN1_SIZE * C_MLP_INPUT_SIZE
                    elif layer_name == "net.2": expected_size = C_MLP_HIDDEN2_SIZE * C_MLP_HIDDEN1_SIZE
                    elif layer_name == "net.4": expected_size = C_MLP_OUTPUT_SIZE * C_MLP_HIDDEN2_SIZE
                    if expected_size > 0 and size != expected_size:
                        print(f"[WARN] Size mismatch for {layer_name}.weight: Expected {expected_size}, Got {size}.")

                    h_code.append(f"\n// Weight: {layer_name}.weight (Size: {size})")
                    h_code.append(f"extern {c_type} {weight_name_c}[{size}];")
                    c_code.append(f"\n// Weight: {layer_name}.weight (Shape: {list(weight_qtensor.shape)} -> Flattened Size: {size})")
                    c_code.append(f"{c_type} {weight_name_c}[{size}] = {{\n  {data_list}\n}};")
                    found_weights[f"{layer_name}.weight"] = size
                else:
                    print(f"[WARN] No weight found in _packed_params for layer: {layer_name}")


                # --- 处理偏置 ---
                if bias_tensor is not None:
                    bias_name_c = layer_name.replace('.', '_') + "_bias"
                    # (修改) 添加 .detach()
                    bias_fp32 = bias_tensor.detach().numpy().flatten()
                    c_type = "const float"
                    data_list = ",\n  ".join([f"{x:.8f}f" for x in bias_fp32])
                    size = len(bias_fp32)

                    expected_size = 0
                    if layer_name == "net.0": expected_size = C_MLP_HIDDEN1_SIZE
                    elif layer_name == "net.2": expected_size = C_MLP_HIDDEN2_SIZE
                    elif layer_name == "net.4": expected_size = C_MLP_OUTPUT_SIZE
                    if expected_size > 0 and size != expected_size:
                         print(f"[WARN] Size mismatch for {layer_name}.bias: Expected {expected_size}, Got {size}.")

                    h_code.append(f"\n// Bias: {layer_name}.bias (Size: {size})")
                    h_code.append(f"extern {c_type} {bias_name_c}[{size}];")
                    c_code.append(f"\n// Bias: {layer_name}.bias (Shape: {list(bias_tensor.shape)} -> Flattened Size: {size})")
                    c_code.append(f"{c_type} {bias_name_c}[{size}] = {{\n  {data_list}\n}};")
                    found_weights[f"{layer_name}.bias"] = size
                #else:
                #    print(f"[INFO] No bias found in _packed_params for layer: {layer_name}") # Bias might be optional

            # 打印状态字典中的其他非 Tensor 条目（用于调试）
            elif isinstance(layer_module, torch.Tensor) and not layer_name.startswith("net.") :
                 pass # 通常是 scale/zero_point 等，我们不在这里导出
            elif not isinstance(layer_module, (nn.Sequential, SparseCompensationMLP, tq.QuantStub, tq.DeQuantStub)):
                 # 打印意外的模块类型
                 #print(f"[DEBUG] Skipping module: {layer_name} (type: {type(layer_module)})") # Can be noisy, uncomment if needed
                 pass # Skip modules we don't explicitly handle

        # --- 结束修改 ---


        # --- .h 文件结束 ---
        h_code.append(f"\n#endif // MODEL_WEIGHTS_H_{output_h_path.stem.upper()}")

        # 检查是否找到了所有预期的权重/偏置
        # (修改) 更新预期的键名，不再包含点号
        expected_keys = {"net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias", "net.4.weight", "net.4.bias"}
        if set(found_weights.keys()) != expected_keys:
             print(f"[WARN] Did not find all expected weights/biases. Required: {expected_keys}. Found: {set(found_weights.keys())}")


        # 4. 写入文件
        with open(output_h_path, 'w') as f:
            f.write("\n".join(h_code))
        print(f"[INFO] Successfully exported declarations to {output_h_path}")

        with open(output_c_path, 'w') as f:
            f.write("\n".join(c_code))
        print(f"[INFO] Successfully exported definitions to {output_c_path}")


    except FileNotFoundError:
        print(f"[ERROR] Input model file not found: {model_int8_path}")
    except Exception as e:
        print(f"[ERROR] Failed to export weights: {e}")
        traceback.print_exc()

# =========================
# 命令行接口 (修改以接受两个输出路径)
# =========================
if __name__ == "__main__":
    model_path = '/work/hwc/SPARSE/runs_sparse_v2_3x6_qat_minmax/best_qat_int8.pt'
    output_h_path = '/work/hwc/SPARSE/infer/model_weights.h'
    output_c_path = '/work/hwc/SPARSE/infer/model_weights.c'
    model_file = Path(model_path)
    output_h_file = Path(output_h_path)
    output_c_file = Path(output_c_path)

    # 确保输出目录存在
    output_h_file.parent.mkdir(parents=True, exist_ok=True)
    output_c_file.parent.mkdir(parents=True, exist_ok=True) # 确保 .c 文件目录也存在

    export_weights_for_c(model_file, output_h_file, output_c_file)


