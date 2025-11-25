import os
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.quantization as tq
import torch.backends.quantized as backend

from teacher_train import (
    TrainingConfig,
    prepare_device,
    build_dataloaders,
    TeacherModel,
    CoM_from_Patch_V12,
    CoM_from_18col_Patch_V15,
)

from teacher_qat_export_int8 import build_qat_teacher_from_ckpt


device = prepare_device()


# ---------------------------------------------------------
# 1. 用导出的 int8 参数构造 Python 侧的“int8 Teacher 分支”
# ---------------------------------------------------------

class Int8ConvLayerPython(nn.Module):
    """
    保存一层卷积的 int8 权重 + 量化参数 + 卷积超参数。
    这里的 forward 使用“量化 → 反量化 + float conv”的方式模拟 int8 卷积。
    """

    def __init__(self, layer_info: Dict[str, Any]):
        super().__init__()
        self.name = layer_info["name"]

        # int8 权重和量化参数
        self.weight_int8: torch.Tensor = layer_info["weight_int8"].to(torch.int8)
        self.weight_scale: torch.Tensor = layer_info["weight_scale"].float()
        self.weight_zero_point: torch.Tensor = layer_info["weight_zero_point"].float()

        # bias 先保持 float32
        self.bias: torch.Tensor = (
            layer_info["bias_float"].float() if layer_info["bias_float"] is not None else None
        )

        # 激活量化参数（conv 输出）
        self.act_scale = (
            float(layer_info["act_scale"].item())
            if (layer_info["act_scale"] is not None and layer_info["act_scale"].numel() == 1)
            else None
        )
        self.act_zero_point = (
            int(layer_info["act_zero_point"].item())
            if (layer_info["act_zero_point"] is not None and layer_info["act_zero_point"].numel() == 1)
            else 0
        )

        # 卷积几何参数
        self.kernel_size = layer_info["kernel_size"]
        self.stride = layer_info["stride"]
        self.padding = layer_info["padding"]
        self.in_channels = layer_info["in_channels"]
        self.out_channels = layer_info["out_channels"]

        # 把权重量化参数展平成标量（per-tensor），方便使用
        # 你的导出里 w_scale.shape = (1,), w_zero_point.shape = (1,)
        self.w_scale = float(self.weight_scale.view(-1)[0])
        self.w_zp = float(self.weight_zero_point.view(-1)[0])

    def conv2d_int8_sim(
        self,
        x_q: torch.Tensor,
        x_scale: float,
        x_zero_point: int,
    ) -> torch.Tensor:
        """
        用 int8 权重 + 输入量化参数模拟一层卷积（在 float 上实现）：
          1) x_q -> x_dequant
          2) w_int8 -> w_dequant
          3) y = Conv2d(x_dequant, w_dequant) + bias
          4) ReLU
        注意：这里不再做输出量化，输出为 float32，对应 QAT conv 中的输出。
        """
        # 1) 反量化输入 x
        x_deq = (x_q.to(torch.float32) - x_zero_point) * x_scale

        # 2) 反量化权重
        w_q = self.weight_int8.to(torch.float32)
        w_deq = (w_q - self.w_zp) * self.w_scale

        # 3) 卷积 (float) + bias
        y = F.conv2d(
            x_deq,
            w_deq,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        # 4) ReLU
        y = F.relu(y)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: 上一层的浮点激活 x。
        行为: 模拟 QAT Conv2d 的行为——对输入做一次 fake-quant，再用量化权重做卷积。
        具体步骤:
          1) 用本层的 act_scale/act_zero_point 将 x 量化为 int8；
          2) 将量化后的输入和 int8 权重反量化到 float；
          3) 进行 Conv2d + ReLU，输出 float32。
        """
        if self.act_scale is None:
            x_scale = 1.0
            x_zero_point = 0
        else:
            x_scale = self.act_scale
            x_zero_point = self.act_zero_point

        # x -> int8
        x_q = torch.round(x / x_scale + x_zero_point)
        x_q = torch.clamp(x_q, -128, 127).to(torch.int8)

        # 使用量化后的输入和权重做卷积仿真
        y = self.conv2d_int8_sim(x_q, x_scale, x_zero_point)
        return y


class Int8PatchNetBranch(nn.Module):
    """
    替代原来的 _PatchNet_V13_Large_Teacher 的 net，用 Python int8 仿真版 conv 堆叠。
    """

    def __init__(self, layers_info: List[Dict[str, Any]]):
        super().__init__()

        # 按 name 排序，确保顺序为 net.0, net.3, net.6, net.9
        layers_info_sorted = sorted(layers_info, key=lambda d: d["name"])
        self.layers = nn.ModuleList(
            [Int8ConvLayerPython(info) for info in layers_info_sorted]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        模拟 QAT 下的 PatchNet：
          - 每一层在前向时，对输入做一次 fake-quant（按本层的 act_scale/act_zero_point）；
          - 用 int8 权重 + 输入量化参数在 float 域中完成 Conv2d + ReLU；
          - 层间传递的始终是浮点激活（已经包含量化噪声），与 QAT 训练时的行为一致。
        """
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Int8TeacherModelPython(nn.Module):
    """
    完整的 TeacherModel 仿真 int8 版本：
      - 使用原 TeacherModel 的 forward 流程（patch 提取、odd/even 分支等）
      - 但 odd_net / even_net 替换为 Int8PatchNetBranch
    """

    def __init__(self, float_teacher: TeacherModel, int8_params: Dict[str, Any]):
        super().__init__()

        # 复制一份 float Teacher 结构与参数（主要是 patch 处理和 CoM 前的部分）
        self.teacher = TeacherModel()
        self.teacher.load_state_dict(float_teacher.state_dict())

        # 替换 odd / even 分支
        odd_layers = int8_params["odd"]
        even_layers = int8_params["even"]
        self.teacher.odd_net = Int8PatchNetBranch(odd_layers)
        self.teacher.even_net = Int8PatchNetBranch(even_layers)

    def forward(self, raw_patch: torch.Tensor, is_odd: torch.Tensor) -> torch.Tensor:
        """
        跟 TeacherModel 一致：输入 raw patches + is_odd mask，输出 decoupled patch。
        具体细节依赖 TeacherModel 的 forward 实现，这里直接调用内部 self.teacher。
        """
        return self.teacher(raw_patch, is_odd)


# ---------------------------------------------------------
# 2. 评估：float Teacher vs QAT Teacher vs Python-int8 Teacher
# ---------------------------------------------------------

def evaluate_int8_teacher(
    float_teacher_ckpt: str,
    qat_fp32_ckpt: str,
    int8_param_path: str,
    max_eval_samples: int = 2000,
):
    cfg = TrainingConfig()
    cfg.batch_size = 64

    # Data
    train_loader, val_loader = build_dataloaders(cfg, device)
    if val_loader is None:
        raise RuntimeError("评估失败：val_loader 为空。")

    # 1) 浮点 Teacher
    print(f"[EVAL] 加载浮点 Teacher: {float_teacher_ckpt}")
    float_teacher = TeacherModel().to(device)
    float_teacher.load_state_dict(torch.load(float_teacher_ckpt, map_location=device))
    float_teacher.eval()

    # 2) QAT Teacher（带 fake-quant）
    print(f"[EVAL] 从 QAT FP32 ckpt 构建 QAT Teacher: {qat_fp32_ckpt}")
    qat_teacher = build_qat_teacher_from_ckpt(qat_fp32_ckpt, device=device)
    qat_teacher.to(device)
    qat_teacher.eval()

    # 3) Python int8 Teacher
    print(f"[EVAL] 加载导出的 int8 参数: {int8_param_path}")
    int8_params: Dict[str, Any] = torch.load(int8_param_path, map_location="cpu")
    int8_teacher = Int8TeacherModelPython(float_teacher, int8_params).to(device)
    int8_teacher.eval()

    # CoM 模块
    com_pred = CoM_from_Patch_V12(*cfg.patch_size).to(device)
    com_gt = CoM_from_18col_Patch_V15(*cfg.patch_size).to(device)

    # 统计量
    diff_float_qat_x = []
    diff_float_qat_y = []
    diff_float_int8_x = []
    diff_float_int8_y = []

    dist_float_gt = []
    dist_qat_gt = []
    dist_int8_gt = []

    num_eval = 0

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
            clean_patch = clean_patch.to(device)
            is_odd = is_odd.to(device)
            peak_r_gt = peak_r_gt.to(device)
            peak_c_gt_18 = peak_c_gt_18.to(device)
            peak_r_m = peak_r_m.to(device)
            peak_c_m_10 = peak_c_m_10.to(device)

            # 浮点 Teacher
            dec_patch_float = float_teacher(raw_patch, is_odd)
            coords_float = com_pred(dec_patch_float, is_odd, peak_r_m, peak_c_m_10)

            # QAT Teacher (fake-quant)
            dec_patch_qat = qat_teacher(raw_patch, is_odd)
            coords_qat = com_pred(dec_patch_qat, is_odd, peak_r_m, peak_c_m_10)

            # Python int8 Teacher
            dec_patch_int8 = int8_teacher(raw_patch, is_odd)
            coords_int8 = com_pred(dec_patch_int8, is_odd, peak_r_m, peak_c_m_10)

            # GT
            coords_gt = com_gt(clean_patch, peak_r_gt, peak_c_gt_18)

            # 差值统计
            diff_fq = coords_float - coords_qat
            diff_fi = coords_float - coords_int8

            diff_float_qat_x.append(diff_fq[:, 0].abs().cpu())
            diff_float_qat_y.append(diff_fq[:, 1].abs().cpu())
            diff_float_int8_x.append(diff_fi[:, 0].abs().cpu())
            diff_float_int8_y.append(diff_fi[:, 1].abs().cpu())

            dist_float_gt.append((coords_float - coords_gt).norm(dim=1).cpu())
            dist_qat_gt.append((coords_qat - coords_gt).norm(dim=1).cpu())
            dist_int8_gt.append((coords_int8 - coords_gt).norm(dim=1).cpu())

            num_eval += raw_patch.size(0)
            if num_eval >= max_eval_samples:
                break

    # 汇总
    diff_float_qat_x = torch.cat(diff_float_qat_x)
    diff_float_qat_y = torch.cat(diff_float_qat_y)
    diff_float_int8_x = torch.cat(diff_float_int8_x)
    diff_float_int8_y = torch.cat(diff_float_int8_y)

    dist_float_gt = torch.cat(dist_float_gt)
    dist_qat_gt = torch.cat(dist_qat_gt)
    dist_int8_gt = torch.cat(dist_int8_gt)

    print("\n[EVAL] === 浮点 Teacher vs QAT Teacher 坐标差 (val 集) ===")
    print(f"|Δx| mean = {diff_float_qat_x.mean().item():.6e}, median = {diff_float_qat_x.median().item():.6e}")
    print(f"|Δy| mean = {diff_float_qat_y.mean().item():.6e}, median = {diff_float_qat_y.median().item():.6e}")

    print("\n[EVAL] === 浮点 Teacher vs Python-int8 Teacher 坐标差 (val 集) ===")
    print(f"|Δx| mean = {diff_float_int8_x.mean().item():.6e}, median = {diff_float_int8_x.median().item():.6e}")
    print(f"|Δy| mean = {diff_float_int8_y.mean().item():.6e}, median = {diff_float_int8_y.median().item():.6e}")

    print("\n[EVAL] === 浮点 / QAT / Python-int8 Teacher vs GT 的距离 (val 集) ===")
    print(f"Dist(float, GT) mean   = {dist_float_gt.mean().item():.6e}")
    print(f"Dist(float, GT) median = {dist_float_gt.median().item():.6e}")
    print(f"Dist(QAT,   GT) mean   = {dist_qat_gt.mean().item():.6e}")
    print(f"Dist(QAT,   GT) median = {dist_qat_gt.median().item():.6e}")
    print(f"Dist(int8,  GT) mean   = {dist_int8_gt.mean().item():.6e}")
    print(f"Dist(int8,  GT) median = {dist_int8_gt.median().item():.6e}")


# ---------------------------------------------------------
# 3. main
# ---------------------------------------------------------

def main():
    cfg = TrainingConfig()
    float_teacher_ckpt = cfg.teacher_model_path
    qat_fp32_ckpt = float_teacher_ckpt.replace(".pth", "_qat_fp32.pth")
    int8_param_path = float_teacher_ckpt.replace(".pth", "_qat_int8_params.pt")

    if not os.path.isfile(qat_fp32_ckpt):
        raise FileNotFoundError(
            f"找不到 QAT FP32 checkpoint: {qat_fp32_ckpt}\n"
            f"请先运行 teacher_qat_train.py 生成该文件。"
        )
    if not os.path.isfile(int8_param_path):
        raise FileNotFoundError(
            f"找不到导出的 int8 参数文件: {int8_param_path}\n"
            f"请先运行 teacher_qat_export_int8.py 生成该文件。"
        )

    print("Data (V16 Pipeline) will be loaded by build_dataloaders in evaluate_int8_teacher...")
    evaluate_int8_teacher(
        float_teacher_ckpt=float_teacher_ckpt,
        qat_fp32_ckpt=qat_fp32_ckpt,
        int8_param_path=int8_param_path,
        max_eval_samples=2000,
    )


if __name__ == "__main__":
    main()