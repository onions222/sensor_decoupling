import os
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torch.ao.quantization as tq
import torch.backends.quantized as backend

from teacher_train import (
    TrainingConfig,
    prepare_device,
    build_dataloaders,
    TeacherModel,
    CoM_from_Patch_V12,
    CoM_from_18col_Patch_V15,
    ensure_parent_dir,
)

device = prepare_device()


# ---------------------------------------------------------
# 1. Teacher QAT 的训练 & 验证（和原 Teacher 训练类似）
# ---------------------------------------------------------

def train_qat_epoch(
    model: nn.Module,
    loader: DataLoader,
    com_pred: CoM_from_Patch_V12,
    com_gt: CoM_from_18col_Patch_V15,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
):
    model.train()
    total_loss, total_coords_loss = 0.0, 0.0
    for batch in loader:
        raw_patch, clean_patch, is_odd, peak_r_gt, peak_c_gt_18, peak_r_m, peak_c_m_10 = batch

        # QAT 模型直接输出解耦 patch
        pred_decoupled_patch = model(raw_patch, is_odd)
        pred_global_coords = com_pred(pred_decoupled_patch, is_odd, peak_r_m, peak_c_m_10)

        with torch.no_grad():
            target_global_coords = com_gt(clean_patch, peak_r_gt, peak_c_gt_18)

        loss_coords = loss_fn(pred_global_coords, target_global_coords)

        optimizer.zero_grad()
        loss_coords.backward()
        optimizer.step()

        total_loss += loss_coords.item()
        total_coords_loss += loss_coords.item()

    num_batches = max(1, len(loader))
    return total_loss / num_batches, total_coords_loss / num_batches


def validate_qat_epoch(
    model: nn.Module,
    loader: DataLoader,
    com_pred: CoM_from_Patch_V12,
    com_gt: CoM_from_18col_Patch_V15,
    loss_fn: nn.Module,
):
    model.eval()
    total_loss, total_coords_loss = 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            raw_patch, clean_patch, is_odd, peak_r_gt, peak_c_gt_18, peak_r_m, peak_c_m_10 = batch

            pred_decoupled_patch = model(raw_patch, is_odd)
            pred_global_coords = com_pred(pred_decoupled_patch, is_odd, peak_r_m, peak_c_m_10)
            target_global_coords = com_gt(clean_patch, peak_r_gt, peak_c_gt_18)

            loss_coords = loss_fn(pred_global_coords, target_global_coords)
            total_loss += loss_coords.item()
            total_coords_loss += loss_coords.item()

    num_batches = max(1, len(loader))
    return total_loss / num_batches, total_coords_loss / num_batches


# ---------------------------------------------------------
# 2. 对 TeacherModel 的 odd_net / even_net 做 conv-bn-relu fuse
# ---------------------------------------------------------

def fuse_teacher_model(model: TeacherModel):
    """
    对 _PatchNet_V13_Large_Teacher 结构做 conv-bn-relu fuse:
        [Conv, BN, ReLU] x 3
        最后一层 Conv 不 fuse（没有后续 BN/ReLU）
    注意：conv / bn 必须在 eval 模式下才能进行融合。
    """
    # 先切换到 eval 模式，确保 BatchNorm 不在 training 状态
    model.eval()
    # odd 分支
    tq.fuse_modules(
        model.odd_net.net,
        [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']],
        inplace=True,
    )
    # even 分支
    tq.fuse_modules(
        model.even_net.net,
        [['0', '1', '2'], ['3', '4', '5'], ['6', '7', '8']],
        inplace=True,
    )
    return model


# ---------------------------------------------------------
# 3. 使用官方 QAT 训练 Teacher
# ---------------------------------------------------------

def run_teacher_qat_training(
    float_teacher_ckpt: str,
    num_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> Tuple[str, str]:
    """
    对 TeacherModel 执行官方 QAT:
      - 从已有的浮点 Teacher ckpt 加载
      - fuse conv-bn-relu
      - 设置 qconfig & prepare_qat
      - 训练若干 epoch
      - convert 成真正的量化模型

    返回:
      (qat_fp32_ckpt_path, qat_int8_model_path)
    """
    cfg = TrainingConfig()
    cfg.batch_size = batch_size

    train_loader, val_loader = build_dataloaders(cfg, device)
    if train_loader is None or val_loader is None:
        raise RuntimeError("数据加载失败，无法进行 QAT 训练。")

    # 1) 构造并加载原始浮点 Teacher
    print(f"[QAT] 加载浮点 Teacher: {float_teacher_ckpt}")
    float_teacher = TeacherModel().to(device)
    state = torch.load(float_teacher_ckpt, map_location=device)
    float_teacher.load_state_dict(state)
    float_teacher.eval()

    # 2) 拷贝一份用于 QAT 的 Teacher
    teacher_qat = TeacherModel().to(device)
    teacher_qat.load_state_dict(state)

    # 3) fuse conv-bn-relu
    print("[QAT] 对 TeacherModel 进行 conv-bn-relu fuse ...")
    teacher_qat = fuse_teacher_model(teacher_qat)

    # prepare_qat 要求模型处于 training 模式
    teacher_qat.train()

    # 4) 设置 QAT qconfig（选择合适的 engine）
    supported = backend.supported_engines
    if "fbgemm" in supported:
        backend.engine = "fbgemm"
    else:
        backend.engine = supported[0]
    print(f"[QAT] 使用量化后端 engine = {backend.engine}")

    teacher_qat.qconfig = tq.get_default_qat_qconfig(backend.engine)
    print("[QAT] 使用默认 QAT qconfig:", teacher_qat.qconfig)

    # 5) prepare_qat
    print("[QAT] 准备进入 QAT（prepare_qat）...")
    teacher_qat = tq.prepare_qat(teacher_qat, inplace=True)
    teacher_qat.to(device)

    # 6) 训练循环（和原 Teacher 训练类似，loss 是 CoM MSE）
    com_pred = CoM_from_Patch_V12(*cfg.patch_size).to(device)
    com_gt = CoM_from_18col_Patch_V15(*cfg.patch_size).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(teacher_qat.parameters(), lr=lr)

    best_val_coords = float("inf")
    qat_fp32_ckpt = float_teacher_ckpt.replace(".pth", "_qat_fp32.pth")

    print(f"[QAT] 开始 QAT 训练，共 {num_epochs} 个 epoch ...")
    for epoch in range(num_epochs):
        train_loss, train_coords = train_qat_epoch(
            teacher_qat, train_loader, com_pred, com_gt, loss_fn, optimizer
        )
        val_loss, val_coords = validate_qat_epoch(
            teacher_qat, val_loader, com_pred, com_gt, loss_fn
        )
        print(
            f"[QAT][Epoch {epoch+1:03d}/{num_epochs:03d}] "
            f"Train Coords: {train_coords:.6e} | Val Coords: {val_coords:.6e}"
        )

        if val_coords < best_val_coords:
            best_val_coords = val_coords
            ensure_parent_dir(qat_fp32_ckpt)
            torch.save(teacher_qat.state_dict(), qat_fp32_ckpt)
            print(f"    -> 保存当前最佳 QAT FP32 Teacher: {qat_fp32_ckpt}")

    print(f"[QAT] QAT 训练完成，最佳 Val Coords = {best_val_coords:.6e}")
    print(f"[QAT] 最佳 QAT FP32 state_dict 已保存到: {qat_fp32_ckpt}")
    return qat_fp32_ckpt


# ---------------------------------------------------------
# 4. 使用量化 Teacher 做完整 pipeline 评估
# ---------------------------------------------------------

def evaluate_quantized_teacher(
    float_teacher_ckpt: str,
    qat_fp32_ckpt: str,
    batch_size: int = 64,
    max_eval_samples: int = 2000,
):
    """
    使用原浮点 Teacher 与基于 QAT FP32 state_dict 构造的量化 Teacher 在同一验证集上进行对比：
      - 浮点 Teacher vs GT 的坐标误差
      - 量化 Teacher vs GT 的坐标误差
      - 浮点 Teacher vs 量化 Teacher 的坐标差
    """
    cfg = TrainingConfig()
    cfg.batch_size = batch_size
    train_loader, val_loader = build_dataloaders(cfg, device)
    if val_loader is None:
        raise RuntimeError("评估失败：val_loader 为空。")

    # 加载原始浮点 Teacher（作为 reference）
    print(f"[EVAL] 加载原始浮点 Teacher: {float_teacher_ckpt}")
    float_teacher = TeacherModel().to(device)
    float_teacher.load_state_dict(torch.load(float_teacher_ckpt, map_location=device))
    float_teacher.eval()

    # 基于 QAT FP32 state_dict 构造量化 Teacher 模型
    print(f"[EVAL] 从 QAT FP32 checkpoint 构建量化 Teacher: {qat_fp32_ckpt}")
    # 1) 构造与 QAT 训练相同结构的模型，先 fuse, 再 prepare_qat
    quant_teacher = TeacherModel().to(device)
    quant_teacher = fuse_teacher_model(quant_teacher)
    # 设置与训练时相同的量化后端和 qconfig
    supported = backend.supported_engines
    if "fbgemm" in supported:
        backend.engine = "fbgemm"
    else:
        backend.engine = supported[0]
    quant_teacher.qconfig = tq.get_default_qat_qconfig(backend.engine)

    # prepare_qat 要求模型处于 training 模式
    quant_teacher.train()
    quant_teacher = tq.prepare_qat(quant_teacher, inplace=True)
    # 2) 加载训练好的 QAT FP32 state_dict
    state_qat = torch.load(qat_fp32_ckpt, map_location=device)
    quant_teacher.load_state_dict(state_qat)
    # 3) 直接作为带 fake-quant 的 QAT 模型使用（不再 convert 成真正的量化算子）
    #    由于当前 PyTorch 构建在 CPU 上不支持 quantized::conv2d_relu，我们保留 fake-quant 版本用于误差评估。
    quant_teacher.eval()

    com_pred = CoM_from_Patch_V12(*cfg.patch_size).to(device)
    com_gt = CoM_from_18col_Patch_V15(*cfg.patch_size).to(device)

    # 统计指标
    coords_diff_x = []
    coords_diff_y = []
    dist_float_gt = []
    dist_quant_gt = []

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

            # 浮点 Teacher pipeline
            pred_patch_float = float_teacher(raw_patch, is_odd)
            coords_float = com_pred(pred_patch_float, is_odd, peak_r_m, peak_c_m_10)

            # 量化 Teacher pipeline
            pred_patch_quant = quant_teacher(raw_patch, is_odd)
            coords_quant = com_pred(pred_patch_quant, is_odd, peak_r_m, peak_c_m_10)

            # GT coords
            coords_gt = com_gt(clean_patch, peak_r_gt, peak_c_gt_18)

            # 差异统计
            diff = coords_float - coords_quant
            coords_diff_x.append(diff[:, 0].abs().cpu())
            coords_diff_y.append(diff[:, 1].abs().cpu())

            dist_float = (coords_float - coords_gt).norm(dim=1)
            dist_quant = (coords_quant - coords_gt).norm(dim=1)
            dist_float_gt.append(dist_float.cpu())
            dist_quant_gt.append(dist_quant.cpu())

            num_eval += raw_patch.size(0)
            if num_eval >= max_eval_samples:
                break

    # 汇总
    coords_diff_x_all = torch.cat(coords_diff_x)
    coords_diff_y_all = torch.cat(coords_diff_y)
    dist_float_gt_all = torch.cat(dist_float_gt)
    dist_quant_gt_all = torch.cat(dist_quant_gt)

    print("\n[EVAL] === 浮点 Teacher vs 量化 Teacher 坐标差 (val 集) ===")
    print(f"|Δx| mean = {coords_diff_x_all.mean().item():.6e}, median = {coords_diff_x_all.median().item():.6e}")
    print(f"|Δy| mean = {coords_diff_y_all.mean().item():.6e}, median = {coords_diff_y_all.median().item():.6e}")

    print("\n[EVAL] === 浮点 Teacher vs GT 的距离 (val 集) ===")
    print(f"Dist(float, GT) mean   = {dist_float_gt_all.mean().item():.6e}")
    print(f"Dist(float, GT) median = {dist_float_gt_all.median().item():.6e}")

    print("\n[EVAL] === 量化 Teacher vs GT 的距离 (val 集) ===")
    print(f"Dist(quant, GT) mean   = {dist_quant_gt_all.mean().item():.6e}")
    print(f"Dist(quant, GT) median = {dist_quant_gt_all.median().item():.6e}")


# ---------------------------------------------------------
# 5. main：一键跑 QAT + 评估
# ---------------------------------------------------------

def main():
    cfg = TrainingConfig()
    float_teacher_ckpt = cfg.teacher_model_path  # 你原来的浮点 Teacher ckpt

    if not os.path.isfile(float_teacher_ckpt):
        raise FileNotFoundError(
            f"找不到浮点 Teacher 模型文件: {float_teacher_ckpt}\n"
            f"请先用 teacher_train.py 训练并保存 Teacher。"
        )

    # 1) 跑 QAT 训练
    qat_fp32_ckpt = run_teacher_qat_training(
        float_teacher_ckpt=float_teacher_ckpt,
        num_epochs=30,      # 可以先用 5 个 epoch 试试，之后再调大
        lr=cfg.learning_rate_teacher,
        batch_size=cfg.batch_size,
    )

    print("\n[MAIN] QAT 训练完成，开始评估量化 Teacher 精度 ...")

    # 2) 使用量化 Teacher 在 val 集上评估完整 pipeline
    evaluate_quantized_teacher(
        float_teacher_ckpt=float_teacher_ckpt,
        qat_fp32_ckpt=qat_fp32_ckpt,
        batch_size=cfg.batch_size,
        max_eval_samples=2000,
    )


if __name__ == "__main__":
    main()