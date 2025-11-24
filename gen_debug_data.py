#!/usr/bin/env python3
"""Generate per-layer int8 debug data for the QAT student model.

Steps performed:
1. Load the first few samples from the validation JSON, reproducing the exact
   preprocessing logic from ``teacher_train.py`` (patch selection, padding,
   normalization, and odd/even row handling).
2. Run the quantized student model (qnnpack backend) on those samples and hook the
   post-convolution activations for each conv block, storing ``tensor.int_repr()``
   snapshots.
3. Run the float student model to produce high-precision reference patches.
4. Compute golden center-of-mass (CoM) coordinates via ``DifferentiableCoM`` for
   both float and int8 outputs.
5. Emit a self-contained ``debug_data.h`` file so the C inference runtime can
   compare SRAM buffers layer-by-layer against the Python "golden" answers.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.ao.quantization as tq

from models import make_student_model

# -----------------------------------------------------------------------------
# Constants (mirroring teacher_train.py definitions)
# -----------------------------------------------------------------------------
ODD_PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (10, 11), (12, 13), (14, 15), (16, 17)]
ODD_SINGLE = [(8, 4), (9, 5)]
ODD_MAP_18_TO_10 = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3, 8: 4, 9: 5, 10: 6, 11: 6, 12: 7, 13: 7, 14: 8, 15: 8, 16: 9, 17: 9}

EVEN_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
EVEN_SINGLE = [(0, 0), (17, 9)]
EVEN_MAP_18_TO_10 = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4, 9: 5, 10: 5, 11: 6, 12: 6, 13: 7, 14: 7, 15: 8, 16: 8, 17: 9}

PATCH_SIZE = (3, 5)
NUM_SAMPLES_DEFAULT = 5
INPUT_SCALE_DEFAULT = 1.0 / 255.0
INPUT_ZP_DEFAULT = 0  # qnnpack activations are stored as uint8. We later view them as int8 for C.

# -----------------------------------------------------------------------------
# Helper modules copied from teacher_train.py to guarantee identical math.
# -----------------------------------------------------------------------------


def compress_merging_to_10_col(matrix_18_col: np.ndarray) -> np.ndarray:
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


class DifferentiableCoM_Patch_3x5(torch.nn.Module):
    def __init__(self, height=3, width=5):
        super().__init__()
        self.height, self.width = height, width
        y, x = torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)
        x_grid, y_grid = x.view(1, -1).repeat(height, 1), y.view(-1, 1).repeat(1, width)
        self.register_buffer('x_grid_buf', x_grid)
        self.register_buffer('y_grid_buf', y_grid)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        img = image.squeeze(1)
        eps = 1e-8
        mass = torch.sum(img, dim=(1, 2)) + eps
        x_weighted = torch.sum(img * self.x_grid_buf, dim=(1, 2))
        y_weighted = torch.sum(img * self.y_grid_buf, dim=(1, 2))
        center_x, center_y = x_weighted / mass, y_weighted / mass
        return torch.stack([center_x, center_y], dim=1)


class CoM_from_Patch_V12(torch.nn.Module):
    def __init__(self, patch_h=3, patch_w=5):
        super().__init__()
        self.local_com_calc = DifferentiableCoM_Patch_3x5(patch_h, patch_w)
        self.ph_offset = patch_h // 2
        self.pw_offset_10col = patch_w // 2
        odd_grid_10_np = np.array([0.5, 2.5, 4.5, 6.5, 8.0, 9.0, 10.5, 12.5, 14.5, 16.5], dtype=np.float32)
        even_grid_10_np = np.array([0.0, 1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.0], dtype=np.float32)
        self.register_buffer('odd_grid_10', torch.from_numpy(odd_grid_10_np))
        self.register_buffer('even_grid_10', torch.from_numpy(even_grid_10_np))

    def forward(self, patches, is_odd_flags, peak_rs, peak_cs_10_col):
        local_coords_3x5 = self.local_com_calc(patches)
        local_x = local_coords_3x5[:, 0]
        local_y = local_coords_3x5[:, 1]
        global_x_10col = local_x + peak_cs_10_col - self.pw_offset_10col
        global_y = local_y + peak_rs - self.ph_offset
        x_clamped = torch.clamp(global_x_10col, 0, 9)
        x_floor, x_ceil = torch.floor(x_clamped).long(), torch.ceil(x_clamped).long()
        x_floor, x_ceil = torch.clamp(x_floor, 0, 9), torch.clamp(x_ceil, 0, 9)
        frac = x_clamped - x_floor.float()
        grids = torch.stack([self.even_grid_10, self.odd_grid_10], dim=0)
        selected_grids = grids[is_odd_flags.long()]
        val_floor = torch.gather(selected_grids, 1, x_floor.unsqueeze(-1)).squeeze(-1)
        val_ceil = torch.gather(selected_grids, 1, x_ceil.unsqueeze(-1)).squeeze(-1)
        global_x_18col = val_floor + (val_ceil - val_floor) * frac
        return torch.stack([global_x_18col, global_y], dim=1)


# -----------------------------------------------------------------------------
# Data loading / preprocessing
# -----------------------------------------------------------------------------


def load_validation_samples(json_path: Path, max_samples: int) -> List[Dict[str, torch.Tensor]]:
    with open(json_path, 'r', encoding='utf-8') as f:
        aligned_data = json.load(f)

    patch_h, patch_w = PATCH_SIZE
    pad2d_18 = (patch_w // 2, patch_w // 2, patch_h // 2, patch_h // 2)
    pad2d_10 = pad2d_18
    eps = 1e-8

    samples: List[Dict[str, torch.Tensor]] = []
    for key in sorted(aligned_data.keys()):
        pair_data = aligned_data[key]
        merging_18 = np.array(pair_data['merging']['normalized_matrix'], dtype=np.float32)
        target_18 = np.array(pair_data['nonmerging']['normalized_matrix'], dtype=np.float32)
        if merging_18.shape != (32, 18):
            continue

        peak_r_gt, peak_c_gt_18 = np.unravel_index(np.argmax(target_18), (32, 18))
        target_t = F.pad(torch.from_numpy(target_18), pad2d_18)
        clean_patch = target_t[peak_r_gt:peak_r_gt + patch_h, peak_c_gt_18:peak_c_gt_18 + patch_w]

        effective_merging_10 = compress_merging_to_10_col(merging_18)
        peak_r_m, peak_c_m_10 = np.unravel_index(np.argmax(effective_merging_10), (32, 10))
        is_odd = float(peak_r_m % 2 == 1)
        merging_t = F.pad(torch.from_numpy(effective_merging_10), pad2d_10)
        merging_patch = merging_t[peak_r_m:peak_r_m + patch_h, peak_c_m_10:peak_c_m_10 + patch_w]

        merging_patch = merging_patch.clamp_min(0)
        merging_patch = merging_patch / (merging_patch.sum() + eps)
        clean_patch = clean_patch.clamp_min(0)
        clean_patch = clean_patch / (clean_patch.sum() + eps)

        samples.append(
            {
                'raw_patch': merging_patch.to(torch.float32),
                'clean_patch': clean_patch.to(torch.float32),
                'is_odd': torch.tensor([is_odd], dtype=torch.float32),
                'peak_r_m': torch.tensor([float(peak_r_m)], dtype=torch.float32),
                'peak_c_m_10': torch.tensor([float(peak_c_m_10)], dtype=torch.float32),
                'peak_r_gt': torch.tensor([float(peak_r_gt)], dtype=torch.float32),
                'peak_c_gt_18': torch.tensor([float(peak_c_gt_18)], dtype=torch.float32),
            }
        )
        if len(samples) >= max_samples:
            break

    if len(samples) < max_samples:
        raise RuntimeError(f"Requested {max_samples} samples but only found {len(samples)} in {json_path}.")
    return samples


# -----------------------------------------------------------------------------
# Model builders (float + quantized)
# -----------------------------------------------------------------------------


def fuse_student(model: torch.nn.Module) -> None:
    for name in ['odd_net', 'even_net']:
        module = getattr(model, name)
        module.eval()
        fuse_list = [
            ['net.0', 'net.1', 'net.2'],
            ['net.3', 'net.4', 'net.5'],
            ['net.6', 'net.7', 'net.8'],
        ]
        torch.quantization.fuse_modules(module, fuse_list, inplace=True)


def load_float_student(state_path: Path) -> torch.nn.Module:
    torch.backends.quantized.engine = 'qnnpack'
    model = make_student_model()
    fuse_student(model)
    model.qconfig = tq.get_default_qat_qconfig('qnnpack')
    tq.prepare_qat(model, inplace=True)
    model.load_state_dict(torch.load(state_path, map_location='cpu'))
    model.eval()
    return model


def load_quantized_student(state_path: Path) -> torch.nn.Module:
    torch.backends.quantized.engine = 'qnnpack'
    model = make_student_model()
    fuse_student(model)
    model.qconfig = tq.get_default_qat_qconfig('qnnpack')
    tq.prepare_qat(model, inplace=True)
    model_int8 = tq.convert(model, inplace=False)
    model_int8.load_state_dict(torch.load(state_path, map_location='cpu'))
    model_int8.eval()
    return model_int8


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def qtensor_to_int8_array(qtensor: torch.Tensor) -> np.ndarray:
    arr = qtensor.int_repr().cpu().numpy()
    if arr.dtype == np.uint8:
        arr = arr.view(np.int8)
    else:
        arr = arr.astype(np.int8)
    return arr


def quantize_patch(patch: torch.Tensor, scale: float, zero_point: int) -> torch.Tensor:
    patch_4d = patch.unsqueeze(0).unsqueeze(0).contiguous()
    return torch.quantize_per_tensor(patch_4d, scale=scale, zero_point=zero_point, dtype=torch.quint8)


def float_patch_batch(patch: torch.Tensor) -> torch.Tensor:
    return patch.unsqueeze(0).unsqueeze(0).contiguous()


def collect_quant_outputs(net_module: torch.nn.Module, q_input: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    assert hasattr(net_module, 'net'), "Expected _PatchNetFactory"  # safeguard
    seq = net_module.net
    hooks, captured = [], {}
    idx_map = {0: 'L0', 3: 'L1', 6: 'L2', 9: 'OUTPUT'}

    def make_hook(key: str):
        def _hook(_, __, out):
            captured[key] = out.detach().clone()
        return _hook

    for idx, name in idx_map.items():
        hooks.append(seq[idx].register_forward_hook(make_hook(name)))

    with torch.no_grad():
        output = net_module(q_input)

    for handle in hooks:
        handle.remove()
    return captured, output


def format_c_array(data: List, value_formatter, indent: int = 1) -> str:
    indent_str = '    ' * indent
    if isinstance(data, (list, tuple)) and data and isinstance(data[0], (list, tuple)):
        lines = []
        for i, item in enumerate(data):
            inner = format_c_array(item, value_formatter, indent + 1)
            suffix = ',' if i < len(data) - 1 else ''
            lines.append(f"{indent_str}{'{'}\n{inner}\n{indent_str}{'}'}{suffix}")
        return "\n".join(lines)
    values = ", ".join(value_formatter(x) for x in data)
    return f"{indent_str}{values}"


def write_header(
    header_path: Path,
    input_scale: float,
    input_zp: int,
    samples_int8: List,
    layer_outputs: Dict[str, List],
    com_float: List[List[float]],
    com_int8: List[List[float]],
) -> None:
    num_samples = len(samples_int8)
    dims = {
        'inputs': [num_samples] + list(np.array(samples_int8[0]).shape),
        'L0': [num_samples] + list(np.array(layer_outputs['L0'][0]).shape),
        'L1': [num_samples] + list(np.array(layer_outputs['L1'][0]).shape),
        'L2': [num_samples] + list(np.array(layer_outputs['L2'][0]).shape),
        'OUTPUT': [num_samples] + list(np.array(layer_outputs['OUTPUT'][0]).shape),
    }

    def dim_str(dim_list: List[int]) -> str:
        return ''.join(f'[{d}]' for d in dim_list)

    int_formatter = lambda v: f"{int(v)}"
    float_formatter = lambda v: f"{float(v):.6f}f"

    with open(header_path, 'w', encoding='utf-8') as f:
        f.write("#ifndef DEBUG_DATA_H\n")
        f.write("#define DEBUG_DATA_H\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define NUM_DEBUG_SAMPLES {num_samples}\n")
        f.write(f"#define DEBUG_INPUT_SCALE {input_scale:.9f}f\n")
        f.write(f"#define DEBUG_INPUT_ZP {input_zp}\n\n")

        f.write(f"static const int8_t DEBUG_INPUT_INT8{dim_str(dims['inputs'])} = {{\n")
        f.write(format_c_array(samples_int8, int_formatter))
        f.write("\n};\n\n")

        for lname in ['L0', 'L1', 'L2', 'OUTPUT']:
            f.write(f"static const int8_t GOLDEN_{lname}_INT8{dim_str(dims[lname])} = {{\n")
            f.write(format_c_array(layer_outputs[lname], int_formatter))
            f.write("\n};\n\n")

        f.write(f"static const float GOLDEN_COM_FLOAT[{num_samples}][2] = {{\n")
        f.write(format_c_array(com_float, float_formatter))
        f.write("\n};\n\n")

        f.write(f"static const float GOLDEN_COM_INT8[{num_samples}][2] = {{\n")
        f.write(format_c_array(com_int8, float_formatter))
        f.write("\n};\n\n")

        f.write("#endif // DEBUG_DATA_H\n")


# -----------------------------------------------------------------------------
# Main orchestration
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate per-layer int8 debug data header.")
    parser.add_argument('--json', type=Path, default=root / 'training_data/validation_int/aligned_g26.json', help='Validation JSON path.')
    parser.add_argument('--int8-model', type=Path, default=root / 'distill/qat_student_runs/best_qat_int8.pt', help='Quantized student state dict path.')
    parser.add_argument('--float-model', type=Path, default=root / 'distill/qat_student_runs/best_qat_float.pt', help='Float (QAT) student checkpoint path.')
    parser.add_argument('--output', type=Path, default=root / 'debug_data.h', help='Output header path.')
    parser.add_argument('--num-samples', type=int, default=NUM_SAMPLES_DEFAULT, help='How many samples to export.')
    parser.add_argument('--input-scale', type=float, default=INPUT_SCALE_DEFAULT, help='Input activation quantization scale.')
    parser.add_argument('--input-zp', type=int, default=INPUT_ZP_DEFAULT, help='Input activation zero point (uint8 domain).')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_validation_samples(args.json, args.num_samples)
    float_model = load_float_student(args.float_model)
    quant_model = load_quantized_student(args.int8_model)
    com_module = CoM_from_Patch_V12(*PATCH_SIZE)

    debug_inputs: List = []
    layer_records: Dict[str, List] = {name: [] for name in ['L0', 'L1', 'L2', 'OUTPUT']}
    com_float_list: List[List[float]] = []
    com_int8_list: List[List[float]] = []

    for sample in samples:
        patch = sample['raw_patch']
        q_patch = quantize_patch(patch, args.input_scale, args.input_zp)
        debug_inputs.append(qtensor_to_int8_array(q_patch).squeeze(0).tolist())

        net_quant = quant_model.odd_net if sample['is_odd'].item() > 0.5 else quant_model.even_net
        captured, final_quant = collect_quant_outputs(net_quant, q_patch)
        for key in layer_records:
            arr = qtensor_to_int8_array(captured[key]).squeeze(0).tolist()
            layer_records[key].append(arr)

        net_float = float_model.odd_net if sample['is_odd'].item() > 0.5 else float_model.even_net
        float_patch = torch.relu(net_float(float_patch_batch(patch))).detach()
        quant_patch_float = torch.relu(final_quant.dequantize())

        with torch.no_grad():
            com_float = com_module(float_patch, sample['is_odd'], sample['peak_r_m'], sample['peak_c_m_10'])
            com_int8 = com_module(quant_patch_float, sample['is_odd'], sample['peak_r_m'], sample['peak_c_m_10'])
        com_float_list.append([float(com_float[0, 0].item()), float(com_float[0, 1].item())])
        com_int8_list.append([float(com_int8[0, 0].item()), float(com_int8[0, 1].item())])

    write_header(args.output, args.input_scale, args.input_zp, debug_inputs, layer_records, com_float_list, com_int8_list)
    print(f"Debug header written to {args.output}")


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    main()
