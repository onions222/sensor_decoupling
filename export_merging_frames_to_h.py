# -*- coding: utf-8 -*-
"""导出 32x18 merging 帧为 C 头文件 merging_frames.h（int16_t）。

不再使用命令行参数，直接运行本脚本即可：
    python export_merging_frames_to_h.py

会自动从 teacher_train.TrainingConfig.json_data_dir 对应的 JSON 数据中
提取所有样本的 32x18 merging 矩阵，并写入 infer3/merging_frames.h：

    #define FRAMES_NUM N
    static const int16_t frames_merging_18[FRAMES_NUM][32][18] = { ... };
"""

import os
import json
from typing import List

import numpy as np
from typing import Tuple

# from teacher_train import TrainingConfig  # 使用训练脚本里的配置路径

class TrainingConfig:
    """Centralized configuration used by ``main``."""

    json_data_dir: str = "/Users/onion/Desktop/code/sensor_decoupling/training_data/diag"
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

FRAME_H = 32
FRAME_W_18 = 18


def collect_merging_frames(max_frames: int = None) -> np.ndarray:
    """从 JSON 数据目录中收集 (N,32,18) 的 merging 矩阵。

    数据结构参考 teacher_train.SensorDataset_V16_Final._load_and_process_patches：
        merging_18 = pair_data["merging"]["normalized_matrix"]

    为了和 C 侧接口匹配，这里统一转换为 int16。
    """

    cfg = TrainingConfig()
    data_dir = cfg.json_data_dir

    if not os.path.isdir(data_dir):
        raise RuntimeError(f"JSON 数据目录不存在: {data_dir}")

    all_files = sorted(
        [fn for fn in os.listdir(data_dir) if fn.lower().endswith(".json")]
    )
    if not all_files:
        raise RuntimeError(f"在目录 {data_dir} 中未找到任何 JSON 文件")

    frames: List[np.ndarray] = []
    print(f"[INFO] 从目录中收集 32x18 merging 帧: {data_dir}")
    print(f"[INFO] 发现 JSON 文件数: {len(all_files)}")

    for filename in all_files:
        filepath = os.path.join(data_dir, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                aligned_data = json.load(f)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[WARN] 读取 JSON 失败: {filepath}, 错误: {exc}")
            continue

        # aligned_data 是一个 dict，value 是 pair_data
        for pair_data in aligned_data.values():
            merging_18 = np.array(
                pair_data["merging"]["normalized_matrix"], dtype=np.float32
            )
            if merging_18.shape != (FRAME_H, FRAME_W_18):
                # 跳过异常 shape
                continue

            # 如果是浮点，先四舍五入到最近整数，再截断到 int16 范围
            if np.issubdtype(merging_18.dtype, np.floating):
                merging_18 = np.rint(merging_18)

            merging_18 = np.clip(merging_18, -32768, 32767).astype(np.int16)
            frames.append(merging_18)

            if max_frames is not None and len(frames) >= max_frames:
                break
        if max_frames is not None and len(frames) >= max_frames:
            break

    if not frames:
        raise RuntimeError("未收集到任何 32x18 merging 帧，请检查 JSON 内容和字段名。")

    frames_arr = np.stack(frames, axis=0)  # (N,32,18)
    print(f"[INFO] 共收集帧数: {frames_arr.shape[0]}, shape={frames_arr.shape}")
    return frames_arr


def dump_frames_to_header(frames: np.ndarray, out_path: str) -> None:
    """将 (N,32,18) 的 int16 数组导出为 C 头文件。

    生成内容形如：

        #ifndef MERGING_FRAMES_H
        #define MERGING_FRAMES_H

        #include <stdint.h>

        #define FRAMES_NUM N

        static const int16_t frames_merging_18[FRAMES_NUM][32][18] = {
            { { ..18.. }, ...32.. },
            ... N ...
        };

        #endif // MERGING_FRAMES_H
    """

    assert frames.ndim == 3, f"Expect (N,32,18), got {frames.shape}"
    n_frames, h, w = frames.shape
    assert h == FRAME_H and w == FRAME_W_18, f"Expect (N,32,18), got {frames.shape}"

    header_guard = "MERGING_FRAMES_H"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"#ifndef {header_guard}\n")
        f.write(f"#define {header_guard}\n\n")
        f.write("#include <stdint.h>\n\n")
        f.write(f"#define FRAMES_NUM {n_frames}\n\n")
        f.write("static const int16_t frames_merging_18[FRAMES_NUM][32][18] = {\n")

        for i in range(n_frames):
            f.write("    {\n")  # frame i
            for r in range(h):
                row_vals = ", ".join(str(int(v)) for v in frames[i, r])
                if r < h - 1:
                    f.write(f"        {{ {row_vals} }},\n")
                else:
                    f.write(f"        {{ {row_vals} }}\n")
            if i < n_frames - 1:
                f.write("    },\n")
            else:
                f.write("    }\n")

        f.write("};\n\n")
        f.write(f"#endif // {header_guard}\n")

    print(f"[EXPORT] 已写入 {n_frames} 帧到 {out_path}")


def main() -> None:
    """入口函数：无需命令行参数，直接运行即可。"""

    # 如果你希望限制导出的帧数，可以把 max_frames 改成具体数字，例如 512。
    max_frames = None  # 导出全部帧
    out_path = os.path.join("infer3", "merging_frames.h")

    frames = collect_merging_frames(max_frames=max_frames)
    dump_frames_to_header(frames, out_path)


if __name__ == "__main__":
    main()