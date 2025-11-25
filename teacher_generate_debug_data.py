import os
from typing import List, Dict, Any
from typing import Dict, Iterable, Optional, Tuple
import torch

class TrainingConfig:
    """Centralized configuration used by ``main``."""

    json_data_dir: str = "/Users/onion/Desktop/code/sensor_decoupling/training_data/diag"
    viz_json_path: str = "/Users/onion/Desktop/code/sensor_decoupling/training_data/aligned_data_for_training_int/aligned_g26.json"
    patch_size: Tuple[int, int] = (3, 5)
    batch_size: int = 64
    train_ratio: float = 0
    num_epochs: int = 100
    learning_rate_teacher: float = 8e-4
    learning_rate_student: float = 1e-3
    alpha: float = 0.3  # hard vs soft target mixing weight
    train_teacher: bool = True
    enable_visualization: bool = True
    teacher_model_path: str = "/Users/onion/Desktop/code/sensor_decoupling/distill/decoupler_model_v16_teacher_best.pth"
    student_model_path: str = "/Users/onion/Desktop/code/sensor_decoupling/distill/pths/decoupler_model_v16_student_best.pth"
    random_seed: int = 42

from teacher_train import (
    prepare_device,
    build_dataloaders,
    TeacherModel,
    CoM_from_Patch_V12,
)
from teacher_qat_int8_deploy_infer import Int8TeacherModelDeploy


def collect_debug_samples(num_samples: int = 64) -> Dict[str, Any]:
    """
    从验证集收集若干样本，并用 Python 部署版 int8 Teacher 生成对应输出与全局坐标。
    返回:
        {
            "in_patches": [N, 15] float,
            "is_odd": [N] int,
            "out_patches": [N, 15] float,
            "peak_r_m": [N] float,
            "peak_c_m_10": [N] float,
            "global_x_18": [N] float,
            "global_y": [N] float,
        }
    """
    cfg = TrainingConfig()
    device = prepare_device()

    # Data
    train_loader, val_loader = build_dataloaders(cfg, device)
    if val_loader is None:
        raise RuntimeError("val_loader is None, 无法生成调试数据。")

    # Float Teacher
    float_teacher = TeacherModel().to(device)
    float_teacher.load_state_dict(
        torch.load(cfg.teacher_model_path, map_location=device)
    )
    float_teacher.eval()

    # Int8 部署 Teacher 所需参数
    int8_param_path = cfg.teacher_model_path.replace(".pth", "_qat_int8_params.pt")
    calib_path = cfg.teacher_model_path.replace(".pth", "_qat_act_calib.pt")

    if not os.path.isfile(int8_param_path):
        raise FileNotFoundError(
            f"找不到 int8 参数文件: {int8_param_path}，请先运行 teacher_qat_export_int8.py 和 teacher_qat_export_c.py。"
        )
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(
            f"找不到激活标定文件: {calib_path}，请先运行 teacher_qat_calib_scales.py。"
        )

    int8_params: Dict[str, Any] = torch.load(int8_param_path, map_location="cpu")
    calib_scales: Dict[str, Any] = torch.load(calib_path, map_location="cpu")

    # 部署版 int8 Teacher（Python 仿真）
    deploy_teacher = Int8TeacherModelDeploy(
        float_teacher, int8_params, calib_scales
    ).to(device)
    deploy_teacher.eval()

    # CoM 模块（与 teacher_train 保持一致）
    com_module = CoM_from_Patch_V12().to(device)
    com_module.eval()

    in_patches: List[torch.Tensor] = []
    out_patches: List[torch.Tensor] = []
    is_odd_list: List[int] = []
    peak_r_m_list: List[torch.Tensor] = []
    peak_c_m_10_list: List[torch.Tensor] = []
    global_x_list: List[torch.Tensor] = []
    global_y_list: List[torch.Tensor] = []

    num_collected = 0

    with torch.no_grad():
        for batch in val_loader:
            (
                raw_patch,
                clean_patch,
                is_odd,
                peak_r_gt,
                peak_c_gt_18,
                peak_r_m,
                peak_c_m_10,
            ) = batch

            raw_patch = raw_patch.to(device)  # [B,1,3,5]
            is_odd = is_odd.to(device)
            peak_r_m = peak_r_m.to(device).float()
            peak_c_m_10 = peak_c_m_10.to(device).float()

            # 部署版 int8 Teacher 输出 patch
            dec_patch_int8 = deploy_teacher(raw_patch, is_odd)  # [B,1,3,5]

            # 用与 teacher_train 一致的 CoM_from_Patch_V12 计算全局坐标
            coords = com_module(dec_patch_int8, is_odd, peak_r_m, peak_c_m_10)
            # coords: [B, 2] -> (x_18, y)

            bsz = raw_patch.size(0)
            for i in range(bsz):
                if num_collected >= num_samples:
                    break

                in_flat = raw_patch[i, 0].reshape(-1).cpu().clone()
                out_flat = dec_patch_int8[i, 0].reshape(-1).cpu().clone()
                flag = int(is_odd[i].item())
                pr = peak_r_m[i].item()
                pc10 = peak_c_m_10[i].item()
                gx = coords[i, 0].item()
                gy = coords[i, 1].item()

                in_patches.append(in_flat)
                out_patches.append(out_flat)
                is_odd_list.append(flag)
                peak_r_m_list.append(torch.tensor(pr))
                peak_c_m_10_list.append(torch.tensor(pc10))
                global_x_list.append(torch.tensor(gx))
                global_y_list.append(torch.tensor(gy))

                num_collected += 1

            if num_collected >= num_samples:
                break

    if num_collected == 0:
        raise RuntimeError("没有从验证集中收集到样本。")

    in_patches_tensor = torch.stack(in_patches, dim=0)        # [N,15]
    out_patches_tensor = torch.stack(out_patches, dim=0)      # [N,15]
    is_odd_tensor = torch.tensor(is_odd_list, dtype=torch.int32)
    peak_r_m_tensor = torch.stack(peak_r_m_list, dim=0).float()
    peak_c_m_10_tensor = torch.stack(peak_c_m_10_list, dim=0).float()
    global_x_tensor = torch.stack(global_x_list, dim=0).float()
    global_y_tensor = torch.stack(global_y_list, dim=0).float()

    return {
        "in_patches": in_patches_tensor,
        "out_patches": out_patches_tensor,
        "is_odd": is_odd_tensor,
        "peak_r_m": peak_r_m_tensor,
        "peak_c_m_10": peak_c_m_10_tensor,
        "global_x_18": global_x_tensor,
        "global_y": global_y_tensor,
    }


def float_to_c_literal(x: float) -> str:
    return f"{float(x):.9e}f"


def write_debug_header(samples: Dict[str, Any], out_path: str, patch_h: int = 3, patch_w: int = 5):
    in_patches = samples["in_patches"]
    out_patches = samples["out_patches"]
    is_odd = samples["is_odd"]
    peak_r_m = samples["peak_r_m"]
    peak_c_m_10 = samples["peak_c_m_10"]
    global_x_18 = samples["global_x_18"]
    global_y = samples["global_y"]

    N = in_patches.size(0)

    lines = []
    lines.append("#pragma once\n")
    lines.append("#include <stdint.h>\n\n")
    lines.append(f"#define DEBUG_NUM_SAMPLES {N}\n")
    lines.append(f"#define DEBUG_PATCH_H {patch_h}\n")
    lines.append(f"#define DEBUG_PATCH_W {patch_w}\n\n")

    # 输入 patch
    lines.append("static const float debug_in_patches[DEBUG_NUM_SAMPLES][DEBUG_PATCH_H*DEBUG_PATCH_W] = {\n")
    for i in range(N):
        vals = [float_to_c_literal(v) for v in in_patches[i].tolist()]
        lines.append(f"    {{ {', '.join(vals)} }},\n")
    lines.append("};\n\n")

    # is_odd
    lines.append("static const int debug_is_odd[DEBUG_NUM_SAMPLES] = {\n")
    lines.append("    " + ", ".join(str(int(v)) for v in is_odd.tolist()) + "\n")
    lines.append("};\n\n")

    # 输出 patch
    lines.append("static const float debug_out_patches[DEBUG_NUM_SAMPLES][DEBUG_PATCH_H*DEBUG_PATCH_W] = {\n")
    for i in range(N):
        vals = [float_to_c_literal(v) for v in out_patches[i].tolist()]
        lines.append(f"    {{ {', '.join(vals)} }},\n")
    lines.append("};\n\n")

    # peak_r_m
    lines.append("static const float debug_peak_r_m[DEBUG_NUM_SAMPLES] = {\n")
    lines.append("    " + ", ".join(float_to_c_literal(v) for v in peak_r_m.tolist()) + "\n")
    lines.append("};\n\n")

    # peak_c_m_10
    lines.append("static const float debug_peak_c_m_10[DEBUG_NUM_SAMPLES] = {\n")
    lines.append("    " + ", ".join(float_to_c_literal(v) for v in peak_c_m_10.tolist()) + "\n")
    lines.append("};\n\n")

    # Python 计算的全局坐标 (18 列)
    lines.append("static const float debug_global_x_18[DEBUG_NUM_SAMPLES] = {\n")
    lines.append("    " + ", ".join(float_to_c_literal(v) for v in global_x_18.tolist()) + "\n")
    lines.append("};\n\n")

    lines.append("static const float debug_global_y[DEBUG_NUM_SAMPLES] = {\n")
    lines.append("    " + ", ".join(float_to_c_literal(v) for v in global_y.tolist()) + "\n")
    lines.append("};\n\n")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.writelines(lines)

    print(f"[DEBUG EXPORT] 已生成头文件: {out_path}")
    print(f"[DEBUG EXPORT] 样本数量: {N}")


def main():
    num_samples = 11
    print(f"[DEBUG EXPORT] 准备从验证集中收集 {num_samples} 个样本...")
    samples = collect_debug_samples(num_samples=num_samples)
    out_path = os.path.join("infer3", "debug_data.h")
    write_debug_header(samples, out_path)


if __name__ == "__main__":
    main()