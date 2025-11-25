import os
from typing import Dict, Any

import torch
import torch.nn as nn

from teacher_train import (
    TrainingConfig,
    prepare_device,
    build_dataloaders,
)
from teacher_qat_export_int8 import build_qat_teacher_from_ckpt


device = prepare_device()


def collect_activation_stats(
    qat_teacher: nn.Module,
    val_loader,
    max_eval_samples: int = 2000,
) -> Dict[str, Dict[str, float]]:
    """
    在 QAT Teacher 上收集每个 conv 层的输入 / 输出最大绝对值。
    只关心 odd_net.net.* 和 even_net.net.* 四层卷积。
    """
    # 只记录这几层
    target_layers = []
    for name, m in qat_teacher.named_modules():
        if name.startswith("odd_net.net.") or name.startswith("even_net.net."):
            if isinstance(m, nn.Conv2d):
                target_layers.append(name)

    stats: Dict[str, Dict[str, float]] = {
        name: {"max_in": 0.0, "max_out": 0.0} for name in target_layers
    }

    # 注册 forward hook
    handles = []

    def make_hook(name):
        def hook(module, inputs, output):
            x = inputs[0].detach()
            y = output.detach()
            max_in = float(x.abs().max().item())
            max_out = float(y.abs().max().item())
            if max_in > stats[name]["max_in"]:
                stats[name]["max_in"] = max_in
            if max_out > stats[name]["max_out"]:
                stats[name]["max_out"] = max_out
        return hook

    for name, m in qat_teacher.named_modules():
        if name in stats:
            h = m.register_forward_hook(make_hook(name))
            handles.append(h)

    # 跑一遍 val 集（可限制样本数）
    num_seen = 0
    qat_teacher.eval()
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

            raw_patch = raw_patch.to(device)
            is_odd = is_odd.to(device)

            _ = qat_teacher(raw_patch, is_odd)

            num_seen += raw_patch.size(0)
            if num_seen >= max_eval_samples:
                break

    # 清理 hook
    for h in handles:
        h.remove()

    return stats


def build_calib_scales_from_stats(stats: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """
    根据每层的 max_in / max_out 构建：
        calib = {
            "odd": {
                "odd_net.net.0": {"x_scale": ..., "y_scale": ...},
                ...
            },
            "even": {
                "even_net.net.0": {...},
                ...
            }
        }
    """
    calib = {"odd": {}, "even": {}}
    eps = 1e-8

    for name, v in stats.items():
        max_in = v["max_in"]
        max_out = v["max_out"]

        x_scale = max_in / 127.0 if max_in > eps else 1.0
        y_scale = max_out / 127.0 if max_out > eps else 1.0

        if name.startswith("odd_net"):
            branch = "odd"
        elif name.startswith("even_net"):
            branch = "even"
        else:
            continue

        calib[branch][name] = {
            "x_scale": x_scale,
            "y_scale": y_scale,
            "x_zero_point": 0,
            "y_zero_point": 0,
        }

    return calib


def main():
    cfg = TrainingConfig()
    cfg.batch_size = 64

    # Data
    train_loader, val_loader = build_dataloaders(cfg, device)
    if val_loader is None:
        raise RuntimeError("校准失败：val_loader 为空。")

    float_teacher_ckpt = cfg.teacher_model_path
    qat_fp32_ckpt = float_teacher_ckpt.replace(".pth", "_qat_fp32.pth")

    if not os.path.isfile(qat_fp32_ckpt):
        raise FileNotFoundError(
            f"找不到 QAT FP32 checkpoint: {qat_fp32_ckpt}\n"
            f"请先运行 teacher_qat_train.py 生成该文件。"
        )

    print(f"[CALIB] 使用 QAT FP32 ckpt: {qat_fp32_ckpt}")
    qat_teacher = build_qat_teacher_from_ckpt(qat_fp32_ckpt, device=device).to(device)

    print("[CALIB] 开始在 val 集上收集激活统计量...")
    stats = collect_activation_stats(qat_teacher, val_loader, max_eval_samples=2000)

    print("[CALIB] 统计结果（每层 max_in / max_out）：")
    for name, v in stats.items():
        print(
            f"  {name}: max_in={v['max_in']:.6e}, max_out={v['max_out']:.6e}"
        )

    calib = build_calib_scales_from_stats(stats)

    # 保存
    calib_path = float_teacher_ckpt.replace(".pth", "_qat_act_calib.pt")
    torch.save(calib, calib_path)
    print(f"[CALIB] 标定结果已保存到: {calib_path}")

    # 打印简要 summary
    for branch in ["odd", "even"]:
        print(f"\n[CALIB] 分支: {branch}")
        for name, d in calib[branch].items():
            print(
                f"  {name}: x_scale={d['x_scale']:.6e}, y_scale={d['y_scale']:.6e}"
            )


if __name__ == "__main__":
    main()