import os
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from teacher_train import (
    TrainingConfig,
    prepare_device,
    build_dataloaders,
    TeacherModel,
    CoM_from_Patch_V12,
    CoM_from_18col_Patch_V15,
)

device = prepare_device()


class Int8ConvLayerDeploy(nn.Module):
    """
    部署规则下的一层卷积：
      - 输入: float32 激活 x_real
      - 步骤:
          1) x_real -> x_int8 (对称量化, x_zp=0, x_scale 来自校准)
          2) w_int8, w_scale, w_zp 来自 QAT 导出
          3) acc = Conv2d( x_int8, w_int8 - w_zp )
          4) y_real = acc * (x_scale * w_scale) + bias
          5) ReLU
      - 输出: float32 y_real
    """

    def __init__(
        self,
        layer_info: Dict[str, Any],
        x_scale: float,
    ):
        super().__init__()
        self.name = layer_info["name"]

        # int8 权重 & 量化参数
        self.weight_int8: torch.Tensor = layer_info["weight_int8"].to(torch.int8)
        self.weight_scale: float = float(layer_info["weight_scale"].view(-1)[0])
        # PyTorch 导出的 w_zp 是 float，但本质上是整数 zero_point
        w_zp_tensor = layer_info["weight_zero_point"]
        self.weight_zero_point: int = int(round(float(w_zp_tensor.view(-1)[0])))

        # bias 使用浮点
        self.bias: torch.Tensor = (
            layer_info["bias_float"].float() if layer_info["bias_float"] is not None else None
        )

        # 卷积几何参数
        self.stride = layer_info["stride"]
        self.padding = layer_info["padding"]

        # 输入量化 scale（对称量化，零点固定为 0）
        self.x_scale: float = float(x_scale)
        self.x_zero_point: int = 0

    def forward(self, x_real: torch.Tensor) -> torch.Tensor:
        """
        x_real: float32, shape [N, C_in, H, W]
        返回: y_real float32, shape [N, C_out, H_out, W_out]
        """

        # 1) x_real -> x_int8 （对称量化）
        x_q = torch.round(x_real / self.x_scale)
        x_q = torch.clamp(x_q, -128, 127).to(torch.int8)

        # 2) 准备用于 conv 的“中心化”权重 & 输入
        #    在 C 里可以用 int32 做乘加，这里用 float 模拟
        x_c = x_q.to(torch.float32)  # 代表 (x_int8 - 0)
        w_q = self.weight_int8.to(torch.float32)
        w_c = w_q - float(self.weight_zero_point)

        # 3) conv 累加 (此处用 float 实现，逻辑上对应 int32 累加)
        acc = F.conv2d(
            x_c,
            w_c,
            bias=None,
            stride=self.stride,
            padding=self.padding,
        )  # acc 相当于 Σ (x_int8 * (w_int8 - w_zp))

        # 4) 还原到实际数值域：acc * (x_scale * w_scale) + bias
        y_real = acc * (self.x_scale * self.weight_scale)
        if self.bias is not None:
            # broadcast add
            y_real = y_real + self.bias.view(1, -1, 1, 1)

        # 5) ReLU
        y_real = F.relu(y_real)

        return y_real


class Int8PatchNetBranchDeploy(nn.Module):
    """
    使用 Int8ConvLayerDeploy 堆叠成 odd_net / even_net 的分支。
    """

    def __init__(
        self,
        layers_info: List[Dict[str, Any]],
        calib_branch: Dict[str, Dict[str, float]],
    ):
        super().__init__()

        # 按 name 排序，确保顺序为 net.0, net.3, net.6, net.9
        layers_info_sorted = sorted(layers_info, key=lambda d: d["name"])

        conv_layers = []
        for info in layers_info_sorted:
            name = info["name"]
            if name not in calib_branch:
                raise KeyError(
                    f"在 calib 中找不到层 {name} 的标定信息，请检查 calib 文件。"
                )
            x_scale = calib_branch[name]["x_scale"]
            conv_layer = Int8ConvLayerDeploy(info, x_scale=x_scale)
            conv_layers.append(conv_layer)

        self.layers = nn.ModuleList(conv_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = layer(out)
        return out


class Int8TeacherModelDeploy(nn.Module):
    """
    完整的部署版 Teacher：
      - 保留原 TeacherModel 的结构和所有非卷积部分（patch 处理、CoM 等）
      - 将 odd_net / even_net 分支替换为 Int8PatchNetBranchDeploy（按部署规则做 int8 卷积）
    """

    def __init__(
        self,
        float_teacher: TeacherModel,
        int8_params: Dict[str, Any],
        calib_scales: Dict[str, Any],
    ):
        super().__init__()

        # 拷贝 float Teacher 的结构和参数（主要是 patch pipeline 与 CoM 前后处理）
        self.teacher = TeacherModel()
        self.teacher.load_state_dict(float_teacher.state_dict())

        # odd / even 分支
        odd_layers = int8_params["odd"]
        even_layers = int8_params["even"]

        odd_calib = calib_scales["odd"]
        even_calib = calib_scales["even"]

        self.teacher.odd_net = Int8PatchNetBranchDeploy(odd_layers, odd_calib)
        self.teacher.even_net = Int8PatchNetBranchDeploy(even_layers, even_calib)

    def forward(self, raw_patch: torch.Tensor, is_odd: torch.Tensor) -> torch.Tensor:
        return self.teacher(raw_patch, is_odd)


# ---------------------------------------------------------
# 评估：float Teacher vs 部署规则 int8 Teacher
# ---------------------------------------------------------

def evaluate_int8_deploy_teacher(
    float_teacher_ckpt: str,
    int8_param_path: str,
    calib_path: str,
    max_eval_samples: int = 2000,
):
    cfg = TrainingConfig()
    cfg.batch_size = 64

    train_loader, val_loader = build_dataloaders(cfg, device)
    if val_loader is None:
        raise RuntimeError("评估失败：val_loader 为空。")

    # 1) 浮点 Teacher
    print(f"[EVAL-DEPLOY] 加载浮点 Teacher: {float_teacher_ckpt}")
    float_teacher = TeacherModel().to(device)
    float_teacher.load_state_dict(torch.load(float_teacher_ckpt, map_location=device))
    float_teacher.eval()

    # 2) 部署规则 int8 Teacher (Python 仿真)
    print(f"[EVAL-DEPLOY] 加载导出的 int8 参数: {int8_param_path}")
    int8_params: Dict[str, Any] = torch.load(int8_param_path, map_location="cpu")

    print(f"[EVAL-DEPLOY] 加载激活标定参数: {calib_path}")
    calib_scales: Dict[str, Any] = torch.load(calib_path, map_location="cpu")

    deploy_teacher = Int8TeacherModelDeploy(float_teacher, int8_params, calib_scales).to(device)
    deploy_teacher.eval()

    # CoM 模块
    com_pred = CoM_from_Patch_V12(*cfg.patch_size).to(device)
    com_gt = CoM_from_18col_Patch_V15(*cfg.patch_size).to(device)

    diff_float_int8_x = []
    diff_float_int8_y = []

    dist_float_gt = []
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

            # 部署规则 int8 Teacher
            dec_patch_int8 = deploy_teacher(raw_patch, is_odd)
            coords_int8 = com_pred(dec_patch_int8, is_odd, peak_r_m, peak_c_m_10)

            # GT
            coords_gt = com_gt(clean_patch, peak_r_gt, peak_c_gt_18)

            # 差值统计
            diff_fi = coords_float - coords_int8
            diff_float_int8_x.append(diff_fi[:, 0].abs().cpu())
            diff_float_int8_y.append(diff_fi[:, 1].abs().cpu())

            dist_float_gt.append((coords_float - coords_gt).norm(dim=1).cpu())
            dist_int8_gt.append((coords_int8 - coords_gt).norm(dim=1).cpu())

            num_eval += raw_patch.size(0)
            if num_eval >= max_eval_samples:
                break

    diff_float_int8_x = torch.cat(diff_float_int8_x)
    diff_float_int8_y = torch.cat(diff_float_int8_y)

    dist_float_gt = torch.cat(dist_float_gt)
    dist_int8_gt = torch.cat(dist_int8_gt)

    print("\n[EVAL-DEPLOY] === 浮点 Teacher vs 部署 int8 Teacher 坐标差 (val 集) ===")
    print(f"|Δx| mean = {diff_float_int8_x.mean().item():.6e}, median = {diff_float_int8_x.median().item():.6e}")
    print(f"|Δy| mean = {diff_float_int8_y.mean().item():.6e}, median = {diff_float_int8_y.median().item():.6e}")

    print("\n[EVAL-DEPLOY] === 浮点 / 部署 int8 Teacher vs GT 的距离 (val 集) ===")
    print(f"Dist(float, GT) mean   = {dist_float_gt.mean().item():.6e}")
    print(f"Dist(float, GT) median = {dist_float_gt.median().item():.6e}")
    print(f"Dist(int8,  GT) mean   = {dist_int8_gt.mean().item():.6e}")
    print(f"Dist(int8,  GT) median = {dist_int8_gt.median().item():.6e}")


def main():
    cfg = TrainingConfig()
    float_teacher_ckpt = cfg.teacher_model_path

    int8_param_path = float_teacher_ckpt.replace(".pth", "_qat_int8_params.pt")
    calib_path = float_teacher_ckpt.replace(".pth", "_qat_act_calib.pt")

    if not os.path.isfile(int8_param_path):
        raise FileNotFoundError(
            f"找不到导出的 int8 参数文件: {int8_param_path}\n"
            f"请先运行 teacher_qat_export_int8.py 生成该文件。"
        )
    if not os.path.isfile(calib_path):
        raise FileNotFoundError(
            f"找不到激活标定文件: {calib_path}\n"
            f"请先运行 teacher_qat_calib_scales.py 生成该文件。"
        )

    print("Data (V16 Pipeline) will be loaded by build_dataloaders in evaluate_int8_deploy_teacher...")
    evaluate_int8_deploy_teacher(
        float_teacher_ckpt=float_teacher_ckpt,
        int8_param_path=int8_param_path,
        calib_path=calib_path,
        max_eval_samples=2000,
    )


if __name__ == "__main__":
    main()