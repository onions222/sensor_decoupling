import os
from typing import Dict, Any, List

import torch
import torch.ao.quantization as tq
import torch.backends.quantized as backend

from teacher_train import (
    TrainingConfig,
    TeacherModel,
)
from teacher_qat_train import fuse_teacher_model


def build_qat_teacher_from_ckpt(qat_fp32_ckpt: str, device: torch.device) -> torch.nn.Module:
    """
    根据 QAT FP32 state_dict 重建一个带 fake-quant 的 QAT Teacher 模型。
    注意：这里不做 convert，所以所有 conv 仍是 float，但有 weight_fake_quant 和 activation_post_process。
    """
    # 1) 构建基础 Teacher 并做 fuse（与训练时一致）
    model = TeacherModel().to(device)
    model = fuse_teacher_model(model)

    # 2) 设置量化后端和 qconfig（与训练/评估时保持一致）
    supported = backend.supported_engines
    if "fbgemm" in supported:
        backend.engine = "fbgemm"
    else:
        backend.engine = supported[0]
    model.qconfig = tq.get_default_qat_qconfig(backend.engine)

    # 3) prepare_qat，插入 fake-quant 模块
    model.train()
    model = tq.prepare_qat(model, inplace=True)

    # 4) 加载 QAT 训练好的 state_dict
    state_qat = torch.load(qat_fp32_ckpt, map_location=device)
    model.load_state_dict(state_qat)

    # 5) 评估模式（仍然是 QAT 模型，conv 为 float + fake-quant）
    model.eval()
    return model


def quantize_weight_per_channel(
    w_fp32: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
) -> torch.Tensor:
    """
    手动按 per-channel 量化权重为 int8：
        w_int8[c, ...] = clamp(round(w_fp32[c, ...] / scale[c]) + zp[c], -128, 127)
    """
    # scale / zp 形状通常为 [out_channels]
    s = scale.view(-1, 1, 1, 1)
    zp = zero_point.view(-1, 1, 1, 1)
    w_int = torch.round(w_fp32 / s) + zp
    w_int = torch.clamp(w_int, -128, 127).to(torch.int8)
    return w_int


def export_int8_params_from_qat_teacher(
    qat_teacher: torch.nn.Module,
    save_path: str,
) -> Dict[str, Any]:
    """
    遍历 QAT Teacher（带 fake-quant）中 odd_net / even_net 的卷积层，
    导出 int8 权重、量化参数和卷积超参数，并保存到 save_path。
    返回导出的参数字典。
    """

    export: Dict[str, List[Dict[str, Any]]] = {
        "odd": [],
        "even": [],
        "other": [],
    }

    # 遍历所有子模块，筛选出带 weight_fake_quant 的 conv/conv+relu 层
    for name, m in qat_teacher.named_modules():
        # 只处理真正的卷积层（QAT 模块都有 weight 和 weight_fake_quant）
        if not (hasattr(m, "weight") and hasattr(m, "weight_fake_quant")):
            continue

        # 只关心 odd_net / even_net 分支
        if name.startswith("odd_net"):
            branch = "odd"
        elif name.startswith("even_net"):
            branch = "even"
        else:
            branch = "other"

        # 提取权重与权重量化参数
        w_fp32 = m.weight.detach().cpu()
        w_fq = m.weight_fake_quant

        if not (hasattr(w_fq, "scale") and hasattr(w_fq, "zero_point")):
            print(f"[WARN] Module {name} 的 weight_fake_quant 没有 scale/zero_point，跳过。")
            continue

        w_scale = w_fq.scale.detach().cpu().clone()
        w_zero_point = w_fq.zero_point.detach().cpu().clone()
        w_int8 = quantize_weight_per_channel(w_fp32, w_scale, w_zero_point)

        # bias 暂时保持为 float32（后续可根据实际 int8 推理公式再量化到 int32）
        b = m.bias.detach().cpu().clone() if m.bias is not None else None

        # 激活量化参数（输出激活）
        act_scale = None
        act_zero_point = None
        if hasattr(m, "activation_post_process"):
            act_observer = m.activation_post_process
            if hasattr(act_observer, "scale") and hasattr(act_observer, "zero_point"):
                act_scale = act_observer.scale.detach().cpu().clone()
                act_zero_point = act_observer.zero_point.detach().cpu().clone()

        # 卷积超参数
        kernel_size = getattr(m, "kernel_size", None)
        stride = getattr(m, "stride", None)
        padding = getattr(m, "padding", None)
        in_channels = getattr(m, "in_channels", None)
        out_channels = getattr(m, "out_channels", None)

        layer_info: Dict[str, Any] = {
            "name": name,
            "weight_int8": w_int8,           # int8 权重 [C_out, C_in, kH, kW]
            "weight_scale": w_scale,         # per-channel scale [C_out]
            "weight_zero_point": w_zero_point,  # per-channel zp [C_out]
            "bias_float": b,                 # 暂时保持 float32
            "act_scale": act_scale,          # 输出激活 scale（可能为标量 tensor）
            "act_zero_point": act_zero_point,# 输出激活 zp
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "in_channels": in_channels,
            "out_channels": out_channels,
        }

        export[branch].append(layer_info)

    # 保存到文件
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(export, save_path)

    # 打印一个简要 summary，方便人工检查
    print(f"[EXPORT] 已导出 {len(export['odd'])} 个 odd_net 卷积层，"
          f"{len(export['even'])} 个 even_net 卷积层，"
          f"{len(export['other'])} 个其他卷积层。")

    for branch in ["odd", "even", "other"]:
        print(f"\n[EXPORT] 分支: {branch}")
        for i, info in enumerate(export[branch]):
            w = info["weight_int8"]
            print(
                f"  [{i}] {info['name']}: "
                f"w_int8.shape={tuple(w.shape)}, "
                f"w_scale.shape={tuple(info['weight_scale'].shape)}, "
                f"act_scale={None if info['act_scale'] is None else info['act_scale'].item() if info['act_scale'].numel()==1 else list(info['act_scale'].shape)}"
            )

    print(f"\n[EXPORT] 参数已保存到: {save_path}")
    return export


def main():
    device = torch.device("cpu")  # 导出参数不需要 GPU

    cfg = TrainingConfig()
    float_teacher_ckpt = cfg.teacher_model_path
    qat_fp32_ckpt = float_teacher_ckpt.replace(".pth", "_qat_fp32.pth")

    if not os.path.isfile(qat_fp32_ckpt):
        raise FileNotFoundError(
            f"找不到 QAT FP32 checkpoint: {qat_fp32_ckpt}\n"
            f"请先运行 teacher_qat_train.py 生成该文件。"
        )

    print(f"[INFO] 使用 QAT FP32 ckpt: {qat_fp32_ckpt}")
    qat_teacher = build_qat_teacher_from_ckpt(qat_fp32_ckpt, device=device)

    # 导出的参数文件路径
    int8_param_path = float_teacher_ckpt.replace(".pth", "_qat_int8_params.pt")

    export_int8_params_from_qat_teacher(qat_teacher, int8_param_path)


if __name__ == "__main__":
    main()