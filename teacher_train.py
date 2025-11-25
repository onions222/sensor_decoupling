"""V16 teacher/student training pipeline with explicit configuration."""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------
ODD_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (10, 11), (12, 13), (14, 15), (16, 17)]
ODD_SINGLE = [(8, 4), (9, 5)]
ODD_MAP_18_TO_10 = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 5, 10: 6, 11: 6, 12: 7, 13: 7, 14: 8, 15: 8, 16: 9, 17: 9}

EVEN_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
EVEN_SINGLE = [(0, 0), (17, 9)]
EVEN_MAP_18_TO_10 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6, 12: 6, 13: 7, 14: 7, 15: 8, 16: 8, 17: 9}


@dataclass
class TrainingConfig:
    """Centralized configuration used by ``main``."""

    json_data_dir: str = "/Users/onion/Desktop/code/sensor_decoupling/training_data/aligned_data_for_training_int"
    viz_json_path: str = "/Users/onion/Desktop/code/sensor_decoupling/training_data/aligned_data_for_training_int/aligned_g26.json"
    patch_size: Tuple[int, int] = (3, 5)
    batch_size: int = 64
    train_ratio: float = 0.8
    num_epochs: int = 100
    learning_rate_teacher: float = 8e-4
    learning_rate_student: float = 1e-3
    alpha: float = 0.3  # hard vs soft target mixing weight
    train_teacher: bool = True
    enable_visualization: bool = True
    teacher_model_path: str = "/Users/onion/Desktop/code/sensor_decoupling/distill/decoupler_model_v16_teacher_best.pth"
    student_model_path: str = "/Users/onion/Desktop/code/sensor_decoupling/distill/pths/decoupler_model_v16_student_best.pth"
    random_seed: int = 42


def prepare_device() -> torch.device:
    """Return the best available device and log it once."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def ensure_parent_dir(path: str) -> None:
    """Create parent directory for ``path`` when needed."""

    os.makedirs(os.path.dirname(path), exist_ok=True)


def compress_merging_to_10_col(matrix_18_col: np.ndarray) -> np.ndarray:
    """Convert a 32x18 matrix into its 32x10 effective merging form."""

    matrix_10_col = np.zeros((32, 10), dtype=np.float32)
    for row_idx in range(32):
        is_odd = (row_idx % 2 == 1)
        pairs = ODD_PAIRS if is_odd else EVEN_PAIRS
        singles = ODD_SINGLE if is_odd else EVEN_SINGLE
        mapping = ODD_MAP_18_TO_10 if is_odd else EVEN_MAP_18_TO_10
        for col_left, col_right in pairs:
            col_10 = mapping[col_left]
            avg_val = (matrix_18_col[row_idx, col_left] + matrix_18_col[row_idx, col_right]) / 2.0
            matrix_10_col[row_idx, col_10] = avg_val
        for col_18, col_10 in singles:
            matrix_10_col[row_idx, col_10] = matrix_18_col[row_idx, col_18]
    return matrix_10_col


# ---------------------------------------------------------------------------
# CoM 模块 (保持与历史版本一致)
# ---------------------------------------------------------------------------

class DifferentiableCoM_Patch_3x5(nn.Module):
    """(子模块) 在 3x5 色块局部坐标系中计算 (x, y) [0-4, 0-2]"""
    def __init__(self, height=3, width=5):
        super(DifferentiableCoM_Patch_3x5, self).__init__()
        self.height, self.width = height, width
        y, x = torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)
        x_grid, y_grid = x.view(1, -1).repeat(height, 1), y.view(-1, 1).repeat(1, width)
        self.register_buffer('x_grid_buf', x_grid)
        self.register_buffer('y_grid_buf', y_grid)
    def forward(self, image):
        img = image.squeeze(1); eps = 1e-8
        mass = torch.sum(img, dim=(1, 2)) + eps
        x_weighted = torch.sum(img * self.x_grid_buf, dim=(1, 2))
        y_weighted = torch.sum(img * self.y_grid_buf, dim=(1, 2))
        center_x, center_y = x_weighted / mass, y_weighted / mass
        return torch.stack([center_x, center_y], dim=1)

class CoM_from_Patch_V12(nn.Module):
    """V16 Pred pipeline: (3x5 patch, 10-col peaks) -> 18-col global coords."""
    def __init__(self, patch_h=3, patch_w=5):
        super(CoM_from_Patch_V12, self).__init__()
        self.local_com_calc = DifferentiableCoM_Patch_3x5(patch_h, patch_w)
        self.ph_offset = patch_h // 2
        self.pw_offset_10col = patch_w // 2
        odd_grid_10_np = np.array([0.5, 2.5, 4.5, 6.5, 8.0, 9.0, 10.5, 12.5, 14.5, 16.5], dtype=np.float32)
        even_grid_10_np = np.array([0.0, 1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.0], dtype=np.float32)
        self.register_buffer('odd_grid_10', torch.from_numpy(odd_grid_10_np))
        self.register_buffer('even_grid_10', torch.from_numpy(even_grid_10_np))
    def forward(self, patches, is_odd_flags, peak_rs, peak_cs_10_col):
        local_coords_3x5 = self.local_com_calc(patches)
        local_x_10col_scalar = local_coords_3x5[:, 0]
        local_y_scalar = local_coords_3x5[:, 1]
        global_x_10col_scalar = local_x_10col_scalar + peak_cs_10_col - self.pw_offset_10col
        global_y_scalar = local_y_scalar + peak_rs - self.ph_offset
        x_clamped = torch.clamp(global_x_10col_scalar, 0, 9)
        x_floor, x_ceil = torch.floor(x_clamped).long(), torch.ceil(x_clamped).long()
        x_floor, x_ceil = torch.clamp(x_floor, 0, 9), torch.clamp(x_ceil, 0, 9)
        frac = x_clamped - x_floor.float()
        grids = torch.stack([self.even_grid_10, self.odd_grid_10], dim=0)
        selected_grids = grids[is_odd_flags.long()]
        val_floor = torch.gather(selected_grids, 1, x_floor.unsqueeze(-1)).squeeze(-1)
        val_ceil = torch.gather(selected_grids, 1, x_ceil.unsqueeze(-1)).squeeze(-1)
        global_x_18col_scalar = val_floor + (val_ceil - val_floor) * frac
        return torch.stack([global_x_18col_scalar, global_y_scalar], dim=1)

class CoM_from_18col_Patch_V15(nn.Module):
    """V16 GT pipeline: (3x5 patch, 18-col peaks) -> 18-col global coords."""
    def __init__(self, patch_h=3, patch_w=5):
        super(CoM_from_18col_Patch_V15, self).__init__()
        self.local_com_calc = DifferentiableCoM_Patch_3x5(patch_h, patch_w)
        self.ph_offset = patch_h // 2
        self.pw_offset_18col = patch_w // 2
    def forward(self, patches, peak_rs, peak_cs_18_col):
        local_coords_3x5 = self.local_com_calc(patches)
        local_x_scalar = local_coords_3x5[:, 0]
        local_y_scalar = local_coords_3x5[:, 1]
        global_x_18col_scalar = local_x_scalar + peak_cs_18_col - self.pw_offset_18col
        global_y_scalar = local_y_scalar + peak_rs - self.ph_offset
        return torch.stack([global_x_18col_scalar, global_y_scalar], dim=1)


class _PatchNet_V13_Large_Teacher(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, out_channels, 1)
        )
    def forward(self, x): return self.net(x)

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.odd_net = _PatchNet_V13_Large_Teacher()
        self.even_net = _PatchNet_V13_Large_Teacher()

    def forward(self, x_patches, is_odd_flags):
        out_odd = self.odd_net(x_patches)
        out_even = self.even_net(x_patches)
        is_odd_mask = is_odd_flags.view(-1, 1, 1, 1) > 0.5
        out = torch.where(is_odd_mask, out_odd, out_even)
        return torch.relu(out)

# --- V16 "学生" 模型 (V14 架构) ---
class _PatchNet_V14_Student(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 2, 3, padding=1, bias=False), nn.BatchNorm2d(2), nn.ReLU(inplace=True),
            nn.Conv2d(2, 4, 3, padding=1, bias=False), nn.BatchNorm2d(4), nn.ReLU(inplace=True),
            nn.Conv2d(4, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(inplace=True),
            nn.Conv2d(2, out_channels, 1)
        )
    def forward(self, x): return self.net(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.odd_net = _PatchNet_V14_Student()
        self.even_net = _PatchNet_V14_Student()

    def forward(self, x_patches, is_odd_flags):
        out_odd = self.odd_net(x_patches)
        out_even = self.even_net(x_patches)
        is_odd_mask = is_odd_flags.view(-1, 1, 1, 1) > 0.5
        out = torch.where(is_odd_mask, out_odd, out_even)
        return torch.relu(out)


class SensorDataset_V16_Final(Dataset):
    """Pre-loads all normalized patches onto ``device`` for fast training."""

    def __init__(self, data_dir: str, patch_size: Tuple[int, int], device: torch.device):
        self.device = device
        self.patch_h, self.patch_w = patch_size
        print(f"Pre-processing JSON data from: {data_dir} (V16 Pipeline)")

        samples_cpu = self._load_and_process_patches(data_dir)
        if not samples_cpu:
            print("错误: 未加载任何样本")
            self.n_samples = 0
            return

        print("Stacking and moving all patches to device memory...")
        self.raw_patches_gpu = torch.stack([sample["raw_patch"] for sample in samples_cpu]).unsqueeze(1).to(device)
        self.clean_patches_gpu = torch.stack([sample["clean_patch"] for sample in samples_cpu]).unsqueeze(1).to(device)
        self.is_odd_gpu = torch.tensor([sample["is_odd"] for sample in samples_cpu], dtype=torch.float32).to(device)

        peak_coords_18_gt = np.stack(
            [[sample["peak_r_gt"], sample["peak_c_gt_18"]] for sample in samples_cpu], axis=0
        ).astype(np.float32)
        peak_coords_10_m = np.stack(
            [[sample["peak_r_m"], sample["peak_c_m_10"]] for sample in samples_cpu], axis=0
        ).astype(np.float32)

        self.peak_coords_18_gt_gpu = torch.from_numpy(peak_coords_18_gt).to(device)
        self.peak_coords_10_m_gpu = torch.from_numpy(peak_coords_10_m).to(device)
        self.n_samples = len(self.raw_patches_gpu)
        print(f"Dataset pre-processing complete. Found {self.n_samples} samples (on device).")

    def _load_and_process_patches(self, data_dir: str):
        samples_in_memory = []
        all_files = sorted([file for file in os.listdir(data_dir) if file.lower().endswith(".json")])
        print(f"找到 {len(all_files)} 个 JSON 文件。正在预处理...")
        patch_half_h, patch_half_w = self.patch_h // 2, self.patch_w // 2
        pad2d_18_col = (patch_half_w, patch_half_w, patch_half_h, patch_half_h)
        pad2d_10_col = pad2d_18_col
        eps = 1e-8

        for filename in tqdm(all_files, desc="Pre-processing JSONs"):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    aligned_data = json.load(file)
                for pair_data in aligned_data.values():
                    merging_18 = np.array(pair_data["merging"]["normalized_matrix"], dtype=np.float32)
                    target_18 = np.array(pair_data["nonmerging"]["normalized_matrix"], dtype=np.float32)
                    if merging_18.shape != (32, 18):
                        continue

                    peak_r_gt, peak_c_gt_18 = np.unravel_index(np.argmax(target_18), (32, 18))
                    target_t_18 = F.pad(torch.from_numpy(target_18), pad2d_18_col)
                    clean_patch = target_t_18[peak_r_gt:peak_r_gt + self.patch_h, peak_c_gt_18:peak_c_gt_18 + self.patch_w]

                    effective_merging_10 = compress_merging_to_10_col(merging_18)
                    peak_r_m, peak_c_m_10 = np.unravel_index(np.argmax(effective_merging_10), (32, 10))
                    is_odd = float(peak_r_m % 2 == 1)
                    merging_t_10 = F.pad(torch.from_numpy(effective_merging_10), pad2d_10_col)
                    merging_patch = merging_t_10[peak_r_m:peak_r_m + self.patch_h, peak_c_m_10:peak_c_m_10 + self.patch_w]

                    merging_patch = merging_patch.clamp_min(0)
                    merging_patch = merging_patch / (merging_patch.sum() + eps)
                    clean_patch = clean_patch.clamp_min(0)
                    clean_patch = clean_patch / (clean_patch.sum() + eps)

                    samples_in_memory.append(
                        {
                            "raw_patch": merging_patch,
                            "clean_patch": clean_patch,
                            "is_odd": is_odd,
                            "peak_r_gt": peak_r_gt,
                            "peak_c_gt_18": peak_c_gt_18,
                            "peak_r_m": peak_r_m,
                            "peak_c_m_10": peak_c_m_10,
                        }
                    )
            except Exception as exc:  # pylint: disable=broad-except
                print(f"警告: 跳过文件 {filename} (错误: {exc})")
        return samples_in_memory

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            self.raw_patches_gpu[idx],
            self.clean_patches_gpu[idx],
            self.is_odd_gpu[idx],
            self.peak_coords_18_gt_gpu[idx][0],
            self.peak_coords_18_gt_gpu[idx][1],
            self.peak_coords_10_m_gpu[idx][0],
            self.peak_coords_10_m_gpu[idx][1],
        )

def build_dataloaders(config: TrainingConfig, device: torch.device):
    """Create dataset plus train/val loaders."""

    try:
        full_dataset = SensorDataset_V16_Final(config.json_data_dir, config.patch_size, device)
    except Exception as exc:  # pylint: disable=broad-except
        print("--- ⚠️ V16 数据加载失败! ⚠️ ---")
        print(f"错误信息: {exc}")
        return None, None

    if len(full_dataset) < 2:
        print("数据不足，无法切分训练/验证集。")
        return None, None

    train_size = max(1, int(config.train_ratio * len(full_dataset)))
    train_size = min(train_size, len(full_dataset) - 1)
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(config.random_seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    print(f"\nData (V16 Pipeline) loaded successfully! Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader


def train_teacher_epoch(
    model: nn.Module,
    loader: DataLoader,
    com_pred: CoM_from_Patch_V12,
    com_gt: CoM_from_18col_Patch_V15,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
):
    model.train()
    total_loss, total_coords_loss = 0.0, 0.0
    for raw_patch, clean_patch, is_odd, peak_r_gt, peak_c_gt_18, peak_r_m, peak_c_m_10 in loader:
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


def validate_teacher_epoch(
    model: nn.Module,
    loader: DataLoader,
    com_pred: CoM_from_Patch_V12,
    com_gt: CoM_from_18col_Patch_V15,
    loss_fn: nn.Module,
):
    model.eval()
    total_loss, total_coords_loss = 0.0, 0.0
    with torch.no_grad():
        for raw_patch, clean_patch, is_odd, peak_r_gt, peak_c_gt_18, peak_r_m, peak_c_m_10 in loader:
            pred_decoupled_patch = model(raw_patch, is_odd)
            pred_global_coords = com_pred(pred_decoupled_patch, is_odd, peak_r_m, peak_c_m_10)
            target_global_coords = com_gt(clean_patch, peak_r_gt, peak_c_gt_18)
            loss_coords = loss_fn(pred_global_coords, target_global_coords)
            total_loss += loss_coords.item()
            total_coords_loss += loss_coords.item()
    num_batches = max(1, len(loader))
    return total_loss / num_batches, total_coords_loss / num_batches


def train_student_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    loader: DataLoader,
    com_pred: CoM_from_Patch_V12,
    com_gt: CoM_from_18col_Patch_V15,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    alpha: float,
):
    student_model.train()
    total_loss, total_coords_loss, total_distill_loss = 0.0, 0.0, 0.0
    for raw_patch, clean_patch, is_odd, peak_r_gt, peak_c_gt_18, peak_r_m, peak_c_m_10 in loader:
        with torch.no_grad():
            pred_patch_teacher = teacher_model(raw_patch, is_odd)
        pred_patch_student = student_model(raw_patch, is_odd)
        pred_global_coords_student = com_pred(pred_patch_student, is_odd, peak_r_m, peak_c_m_10)
        with torch.no_grad():
            target_global_coords = com_gt(clean_patch, peak_r_gt, peak_c_gt_18)
        loss_coords = loss_fn(pred_global_coords_student, target_global_coords)
        loss_distill = loss_fn(pred_patch_student, pred_patch_teacher)
        total_loss_batch = (alpha * loss_coords) + ((1.0 - alpha) * loss_distill)
        optimizer.zero_grad()
        total_loss_batch.backward()
        optimizer.step()
        total_loss += total_loss_batch.item()
        total_coords_loss += loss_coords.item()
        total_distill_loss += loss_distill.item()
    num_batches = max(1, len(loader))
    return (
        total_loss / num_batches,
        total_coords_loss / num_batches,
        total_distill_loss / num_batches,
    )


def validate_student_epoch(
    student_model: nn.Module,
    teacher_model: nn.Module,
    loader: DataLoader,
    com_pred: CoM_from_Patch_V12,
    com_gt: CoM_from_18col_Patch_V15,
    loss_fn: nn.Module,
    alpha: float,
):
    student_model.eval()
    total_loss, total_coords_loss, total_distill_loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for raw_patch, clean_patch, is_odd, peak_r_gt, peak_c_gt_18, peak_r_m, peak_c_m_10 in loader:
            pred_patch_teacher = teacher_model(raw_patch, is_odd)
            pred_patch_student = student_model(raw_patch, is_odd)
            pred_global_coords_student = com_pred(pred_patch_student, is_odd, peak_r_m, peak_c_m_10)
            target_global_coords = com_gt(clean_patch, peak_r_gt, peak_c_gt_18)
            loss_coords = loss_fn(pred_global_coords_student, target_global_coords)
            loss_distill = loss_fn(pred_patch_student, pred_patch_teacher)
            total_loss_batch = (alpha * loss_coords) + ((1.0 - alpha) * loss_distill)
            total_loss += total_loss_batch.item()
            total_coords_loss += loss_coords.item()
            total_distill_loss += loss_distill.item()
    num_batches = max(1, len(loader))
    return (
        total_loss / num_batches,
        total_coords_loss / num_batches,
        total_distill_loss / num_batches,
    )


def run_teacher_training(
    config: TrainingConfig,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    print("--- 开始训练 (V16 - 教师模型) ---")
    model = TeacherModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate_teacher)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    com_pred = CoM_from_Patch_V12(*config.patch_size).to(device)
    com_gt = CoM_from_18col_Patch_V15(*config.patch_size).to(device)
    loss_fn = nn.MSELoss()
    best_val_coords_loss = float("inf")

    for epoch in range(config.num_epochs):
        train_loss, train_coords = train_teacher_epoch(model, train_loader, com_pred, com_gt, loss_fn, optimizer)
        val_loss, val_coords = validate_teacher_epoch(model, val_loader, com_pred, com_gt, loss_fn)
        scheduler.step(val_coords)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1:03d}/{config.num_epochs:03d}] | (Teacher) | LR: {current_lr:.1e} | "
            f"Val Loss: {val_loss:.4f} | Val Coords: {val_coords:.4f}"
        )
        if val_coords < best_val_coords_loss:
            best_val_coords_loss = val_coords
            ensure_parent_dir(config.teacher_model_path)
            torch.save(model.state_dict(), config.teacher_model_path)
            print(f"    -> 新的最佳教师模型已保存 (Val Coords: {val_coords:.4f})")

    print("--- 教师训练完成 ---")
    print(f"最佳教师模型 (Val Coords: {best_val_coords_loss:.4f}) 已保存至 {config.teacher_model_path}")
    return config.teacher_model_path, "Teacher"


def run_student_training(
    config: TrainingConfig,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    print("--- 开始训练 (V16 - 学生模型/蒸馏) ---")
    teacher_model = TeacherModel().to(device)
    try:
        teacher_model.load_state_dict(torch.load(config.teacher_model_path, map_location=device))
        print("--- 成功加载 V16 教师模型 ---")
    except Exception as exc:  # pylint: disable=broad-except
        print("--- ⚠️ V16 教师模型加载失败! ⚠️ ---")
        print(f"错误: {exc}")
        raise
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model = StudentModel().to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=config.learning_rate_student)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    com_pred = CoM_from_Patch_V12(*config.patch_size).to(device)
    com_gt = CoM_from_18col_Patch_V15(*config.patch_size).to(device)
    loss_fn = nn.MSELoss()
    best_val_coords_loss = float("inf")

    for epoch in range(config.num_epochs):
        train_loss, train_coords, train_distill = train_student_epoch(
            student_model, teacher_model, train_loader, com_pred, com_gt, loss_fn, optimizer, config.alpha
        )
        val_loss, val_coords, val_distill = validate_student_epoch(
            student_model, teacher_model, val_loader, com_pred, com_gt, loss_fn, config.alpha
        )
        scheduler.step(val_coords)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch + 1:03d}/{config.num_epochs:03d}] | (Student) | LR: {current_lr:.1e} | "
            f"Val Loss: {val_loss:.4f} | Val Coords: {val_coords:.4f} | Val Distill: {val_distill:.4f}"
        )
        if val_coords < best_val_coords_loss:
            best_val_coords_loss = val_coords
            ensure_parent_dir(config.student_model_path)
            torch.save(student_model.state_dict(), config.student_model_path)
            print(f"    -> 新的最佳学生模型已保存 (Val Coords: {val_coords:.4f})")

    print("--- 学生训练完成 ---")
    print(f"最佳学生模型 (Val Coords: {best_val_coords_loss:.4f}) 已保存至 {config.student_model_path}")
    return config.student_model_path, "Student"


def load_data_for_viz_v16(json_path: str, patch_size: Tuple[int, int], device: torch.device):
    """Load visualization samples compatible with the V16 pipeline."""

    raw_patch_list, clean_patch_list, is_odd_list = [], [], []
    peak_coords_18_gt_list, peak_coords_10_merging_list = [], []

    if not os.path.isfile(json_path):
        print(f"错误: 可视化文件未找到: {json_path}")
        return [None] * 5

    print(f"正在加载并处理可视化文件: {json_path}")
    patch_h, patch_w = patch_size
    pad2d_18_col = (patch_w // 2, patch_w // 2, patch_h // 2, patch_h // 2)
    pad2d_10_col = pad2d_18_col
    eps = 1e-8

    try:
        with open(json_path, "r", encoding="utf-8") as file:
            aligned_data = json.load(file)
        for pair_data in aligned_data.values():
            merging_18 = np.array(pair_data["merging"]["normalized_matrix"], dtype=np.float32)
            target_18 = np.array(pair_data["nonmerging"]["normalized_matrix"], dtype=np.float32)
            if merging_18.shape != (32, 18):
                continue

            peak_r_gt, peak_c_gt_18 = np.unravel_index(np.argmax(target_18), (32, 18))
            target_t_18 = F.pad(torch.from_numpy(target_18), pad2d_18_col)
            clean_patch = target_t_18[peak_r_gt:peak_r_gt + patch_h, peak_c_gt_18:peak_c_gt_18 + patch_w]

            effective_merging_10 = compress_merging_to_10_col(merging_18)
            peak_r_m, peak_c_m_10 = np.unravel_index(np.argmax(effective_merging_10), (32, 10))
            merging_t_10 = F.pad(torch.from_numpy(effective_merging_10), pad2d_10_col)
            merging_patch = merging_t_10[peak_r_m:peak_r_m + patch_h, peak_c_m_10:peak_c_m_10 + patch_w]

            merging_patch = merging_patch.clamp_min(0)
            merging_patch = merging_patch / (merging_patch.sum() + eps)
            clean_patch = clean_patch.clamp_min(0)
            clean_patch = clean_patch / (clean_patch.sum() + eps)

            raw_patch_list.append(merging_patch)
            clean_patch_list.append(clean_patch)
            is_odd_list.append(float(peak_r_m % 2 == 1))
            peak_coords_18_gt_list.append([peak_r_gt, peak_c_gt_18])
            peak_coords_10_merging_list.append([peak_r_m, peak_c_m_10])
    except Exception as exc:  # pylint: disable=broad-except
        print(f"加载 JSON 时出错: {exc}")
        return [None] * 5

    print(f"成功加载 {len(raw_patch_list)} 个可视化样本。")
    raw_tensor = torch.stack(raw_patch_list).unsqueeze(1).to(device)
    clean_tensor = torch.stack(clean_patch_list).unsqueeze(1).to(device)
    is_odd_tensor = torch.tensor(is_odd_list, dtype=torch.float32).to(device)
    peak_18_gt_tensor = torch.tensor(peak_coords_18_gt_list, dtype=torch.float32).to(device)
    peak_10_merging_tensor = torch.tensor(peak_coords_10_merging_list, dtype=torch.float32).to(device)
    return raw_tensor, clean_tensor, is_odd_tensor, peak_18_gt_tensor, peak_10_merging_tensor


def visualize_model_predictions(
    model_path: str,
    model_type: str,
    config: TrainingConfig,
    device: torch.device,
):
    print("\n--- 开始可视化 (V16) ---")
    print(f"将加载最佳 {model_type} 模型: {model_path}")

    model = StudentModel().to(device) if model_type == "Student" else TeacherModel().to(device)
    com_pred = CoM_from_Patch_V12(*config.patch_size).to(device)
    com_gt = CoM_from_18col_Patch_V15(*config.patch_size).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"已成功加载最佳 {model_type} 模型 {model_path}")
    except Exception as exc:  # pylint: disable=broad-except
        print("--- ⚠️ 可视化失败! ⚠️ ---")
        print(f"错误: {exc}")
        return

    raw_data, clean_data, is_odd_data, peak_18_gt_data, peak_10_merging_data = load_data_for_viz_v16(
        config.viz_json_path, config.patch_size, device
    )
    if raw_data is None:
        print("可视化数据未加载，无法执行绘图。")
        return

    with torch.no_grad():
        clean_global_coords = com_gt(clean_data, peak_18_gt_data[:, 0], peak_18_gt_data[:, 1])
        raw_global_coords = com_pred(raw_data, is_odd_data, peak_10_merging_data[:, 0], peak_10_merging_data[:, 1])
        pred_patches = model(raw_data, is_odd_data)
        pred_global_coords = com_pred(pred_patches, is_odd_data, peak_10_merging_data[:, 0], peak_10_merging_data[:, 1])

    raw_coords_np = raw_global_coords.cpu().numpy()
    clean_coords_np = clean_global_coords.cpu().numpy()
    pred_coords_np = pred_global_coords.cpu().numpy()
    print("全局坐标计算完毕。")

    def transform_coords(coords_np):
        x_vals = coords_np[:, 0] * 64.0 + 32.0
        y_vals = coords_np[:, 1] * 64.0 + 32.0
        return x_vals, y_vals

    raw_x_viz, raw_y_viz = transform_coords(raw_coords_np)
    clean_x_viz, clean_y_viz = transform_coords(clean_coords_np)
    pred_x_viz, pred_y_viz = transform_coords(pred_coords_np)

    print("开始绘图...")
    # 创建保存目录
    save_dir = '/Users/onion/Desktop/code/sensor_decoupling/figs_val/v16_all'
    os.makedirs(save_dir, exist_ok=True)
    
    # 为整个JSON文件生成一张图像，包含所有样本
    plt.figure(figsize=(12, 12))
    
    # 绘制所有样本的数据点
    plt.scatter(clean_x_viz, clean_y_viz, marker="*", s=150, c="lime", edgecolors="black", label="真值 (Clean)", zorder=5)
    plt.scatter(raw_x_viz, raw_y_viz, marker="x", s=80, c="red", label="解耦前 (Raw)", zorder=4)
    plt.scatter(pred_x_viz, pred_y_viz, marker="o", s=80, c="blue", alpha=0.7, label=f"解耦后 ({model_type})", zorder=3)
    
    # 为每个样本绘制连接线
    for idx in range(len(clean_x_viz)):
        plt.plot([raw_x_viz[idx], clean_x_viz[idx]], [raw_y_viz[idx], clean_y_viz[idx]], "r--", linewidth=0.5, alpha=0.5)
        plt.plot([pred_x_viz[idx], clean_x_viz[idx]], [pred_y_viz[idx], clean_y_viz[idx]], "b--", linewidth=0.5, alpha=0.5)
    
    plt.title(f"V16 ({model_type}) 全局坐标校正 (来自 {os.path.basename(config.viz_json_path)})")
    plt.xlabel("X 坐标 (全局 18-col 物理坐标, 已变换: x*64+32)")
    plt.ylabel("Y 坐标 (全局 32-row 物理坐标, 已变换: y*64+32)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.axis("equal")
    
    # 保存图像，使用JSON文件名作为图像文件名
    json_filename = os.path.splitext(os.path.basename(config.viz_json_path))[0]
    save_path = os.path.join(save_dir, f"{json_filename}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存 {len(clean_x_viz)} 个样本的可视化结果到 {save_path}")
    
    print(f"JSON文件 {config.viz_json_path} 的可视化结果已保存到 {save_path}")


def main():
    config = TrainingConfig()
    device = prepare_device()
    train_loader, val_loader = build_dataloaders(config, device)
    if train_loader is None or val_loader is None:
        print("训练未开始，因为数据加载失败。请检查配置中的路径。")
        return

    if config.train_teacher:
        model_path, model_type = run_teacher_training(config, device, train_loader, val_loader)
    else:
        model_path, model_type = run_student_training(config, device, train_loader, val_loader)

    if config.enable_visualization:
        visualize_model_predictions(model_path, model_type, config, device)


if __name__ == "__main__":
    main()

