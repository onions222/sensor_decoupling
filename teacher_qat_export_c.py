import os
import torch
from typing import Dict, Any

from teacher_train import TrainingConfig, TeacherModel
from teacher_qat_export_int8 import build_qat_teacher_from_ckpt


def export_tensor_c_array(name: str, tensor: torch.Tensor, indent: int = 4) -> str:
    """
    将 PyTorch Tensor 转成 C 语言数组定义
    """
    flat = tensor.contiguous().view(-1).cpu().numpy()
    indent_space = " " * indent
    values = ", ".join(str(int(v)) for v in flat)
    return f"static const int8_t {name}[] = {{ {values} }};\n"


def export_float_c_array(name: str, tensor: torch.Tensor, indent: int = 4) -> str:
    flat = tensor.contiguous().view(-1).cpu().numpy()
    indent_space = " " * indent
    values = ", ".join(f"{float(v):.9e}f" for v in flat)
    return f"static const float {name}[] = {{ {values} }};\n"


def export_params_to_c(int8_params: Dict[str, Any], calib: Dict[str, Any], out_dir: str):
    """
    将 Python int8 权重 + scale + bias 导出为 C 语言头文件
    """
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    h_weights = []
    h_scales = []
    h_shapes = []

    # 遍历 odd / even 分支
    for branch in ("odd", "even"):
        layers = int8_params[branch]
        calib_branch = calib[branch]

        for layer in layers:
            name = layer["name"].replace(".", "_")  # e.g. odd_net_net_0

            # ---- 导出 INT8 权重 ----
            w = layer["weight_int8"].to(torch.int8)
            c_name = f"{branch}_{name}_w"
            h_weights.append(export_tensor_c_array(c_name, w))

            # ---- bias float ----
            b = layer["bias_float"]
            if b is None:
                b = torch.zeros(layer["out_channels"], dtype=torch.float32)
            c_bias = f"{branch}_{name}_bias"
            h_weights.append(export_float_c_array(c_bias, b))

            # ---- 几何参数 ----
            kh, kw = layer["kernel_size"]
            ic = layer["in_channels"]
            oc = layer["out_channels"]
            stride_h, stride_w = layer["stride"]
            pad_h, pad_w = layer["padding"]

            h_shapes.append(
                f"// {branch} {name}\n"
                f"#define {branch.upper()}_{name.upper()}_IC {ic}\n"
                f"#define {branch.upper()}_{name.upper()}_OC {oc}\n"
                f"#define {branch.upper()}_{name.upper()}_KH {kh}\n"
                f"#define {branch.upper()}_{name.upper()}_KW {kw}\n"
                f"#define {branch.upper()}_{name.upper()}_STR {stride_h}\n"
                f"#define {branch.upper()}_{name.upper()}_PAD {pad_h}\n\n"
            )

            # ---- scale ----
            x_scale = calib_branch[layer["name"]]["x_scale"]
            w_scale = float(layer["weight_scale"].view(-1)[0])
            w_zp = float(layer["weight_zero_point"].view(-1)[0])

            h_scales.append(
                f"// {branch} {name}\n"
                f"static const float {branch}_{name}_x_scale = {x_scale:.9e}f;\n"
                f"static const float {branch}_{name}_w_scale = {w_scale:.9e}f;\n"
                f"static const int   {branch}_{name}_w_zp    = {(int(w_zp))};\n\n"
            )

    # ---- 写入文件 ----
    with open(os.path.join(out_dir, "teacher_int8_params.h"), "w") as f:
        f.write("#pragma once\n#include <stdint.h>\n\n")
        f.writelines(h_weights)

    with open(os.path.join(out_dir, "teacher_int8_shapes.h"), "w") as f:
        f.write("#pragma once\n\n")
        f.writelines(h_shapes)

    with open(os.path.join(out_dir, "teacher_int8_scales.h"), "w") as f:
        f.write("#pragma once\n\n")
        f.writelines(h_scales)

    print("[EXPORT] 已成功导出 C 参数到 infer3/ 文件夹")


def main():
    cfg = TrainingConfig()

    float_teacher_ckpt = cfg.teacher_model_path
    qat_fp32_ckpt = float_teacher_ckpt.replace(".pth", "_qat_fp32.pth")
    int8_param_path = float_teacher_ckpt.replace(".pth", "_qat_int8_params.pt")
    calib_path = float_teacher_ckpt.replace(".pth", "_qat_act_calib.pt")

    print("[EXPORT] loading files...")
    int8_params: Dict[str, Any] = torch.load(int8_param_path, map_location="cpu")
    calib: Dict[str, Any] = torch.load(calib_path, map_location="cpu")

    out_dir = "infer3"
    export_params_to_c(int8_params, calib, out_dir)


if __name__ == "__main__":
    main()