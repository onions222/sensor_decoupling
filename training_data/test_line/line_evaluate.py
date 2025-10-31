# -*- coding: utf-8 -*-

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# ----------------------
# 工具函数
# ----------------------

def calculate_centroid_x_torch(prob):
    """
    计算概率分布在x轴方向的质心坐标
    
    参数:
        prob: (B, C, H, W) 概率分布张量，B为批次大小
    
    返回:
        (B,) 每个样本在x轴方向的质心坐标
    """
    b, _, h, w = prob.shape
    # 创建x轴坐标网格
    x = torch.linspace(0, w - 1, w, device=prob.device).view(1, 1, 1, w)
    # 计算x轴方向的加权平均，得到质心坐标
    cx = (prob * x).sum(dim=(-1, -2)) / (prob.sum(dim=(-1,-2)) + 1e-8)
    return cx.squeeze(1)              # (B,)

def calculate_centroid_y_torch(prob):
    """
    计算概率分布在y轴方向的质心坐标
    参数:
        prob: (B, C, H, W)
    返回:
        (B,) 每个样本在y轴方向的质心坐标
    """
    b, _, h, w = prob.shape
    y = torch.linspace(0, h - 1, h, device=prob.device).view(1, 1, h, 1)
    cy = (prob * y).sum(dim=(-1, -2)) / (prob.sum(dim=(-1,-2)) + 1e-8)
    return cy.squeeze(1)               # (B,)

# ----------------------
# 先验：三值方向图（1 通道）
# ----------------------

def create_guidance_side_one_channel(rows=32, cols=18):
    """返回 (1, rows, cols)，左=-1，右=+1，独立/非成对=0"""
    side = np.zeros((rows, cols), dtype=np.float32)
    odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
    odd_single = [8,9]
    even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    even_single= [0,17]
    for i in range(rows):
        is_odd = (i % 2 == 1)
        pairs  = odd_pairs if is_odd else even_pairs
        singles= odd_single if is_odd else even_single
        for l, r in pairs:
            side[i, l] = -1.0
            side[i, r] = +1.0
        for s in singles:
            side[i, s] = 0.0
    return torch.from_numpy(side).unsqueeze(0).contiguous().float()

# ----------------------
# 数据集
# ----------------------

class TouchDatasetJSON(Dataset):
    def __init__(self, data_dir, nonmerging_data_dir, patch_size=(3,5)):
        self.data_dir = data_dir
        self.nonmerging_data_dir = nonmerging_data_dir
        self.patch_h, self.patch_w = patch_size
        self.side_map = create_guidance_side_one_channel(32, 18)
        self.samples = self._load_samples_into_memory()

    def _load_samples_into_memory(self):
        print(f"Pre-loading dataset into memory...")
        samples_in_memory = []
        all_files = [f for f in os.listdir(self.data_dir) if f.lower().endswith('.json')]
        all_files.sort()
        
        for filename in tqdm(all_files, desc="Loading JSON files"):
            filepath = os.path.join(self.data_dir, filename)
            nonmerging_filepath = os.path.join(self.nonmerging_data_dir, filename)

            if not os.path.exists(nonmerging_filepath):
                print(f"Warning: Corresponding nonmerging file not found for {filename}. Skipping.")
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    aligned_data = json.load(f)
                with open(nonmerging_filepath, 'r', encoding='utf-8') as f_nm:
                    nonmerging_data_full = json.load(f_nm)

                for point_id, pair_data in aligned_data.items():
                    if not all(len(row) == len(pair_data['normalized_matrix'][0]) for row in pair_data['normalized_matrix']):
                        continue
                    
                    nonmerging_point_data = nonmerging_data_full.get(point_id)
                    if not nonmerging_point_data:
                        continue

                    merging = np.array(pair_data['normalized_matrix'], dtype=np.float32)
                    tx_temp = np.array(pair_data['tx_temp'], dtype=np.float32)
                    rx_temp = np.array(pair_data['rx_temp'], dtype=np.float32)
                    sum_val = np.array(pair_data['sum_val'], dtype=np.float32)

                    nm_tx_temp = np.array(nonmerging_point_data['tx_temp'], dtype=np.float32)
                    nm_rx_temp = np.array(nonmerging_point_data['rx_temp'], dtype=np.float32)


                    info = {"source_file": filename, 
                            "tx_temp": tx_temp,
                            "rx_temp": rx_temp,
                            "nm_tx_temp": nm_tx_temp,
                            "nm_rx_temp": nm_rx_temp,
                            "sum_val": sum_val,
                            "id": point_id
                           }

                    samples_in_memory.append({
                        "merging": merging,
                        "info": info
                    })
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                print(f"Warning: Skipping file {filename} or an entry within it due to error: {e}")

        print(f"Successfully loaded {len(samples_in_memory)} samples into memory.")
        return samples_in_memory

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dp = self.samples[idx]
        merging, info = dp['merging'], dp['info']
        peak_r, peak_c = np.unravel_index(np.argmax(merging), merging.shape)

        ph, pw = self.patch_h//2, self.patch_w//2
        pad2d = (pw, pw, ph, ph)
        merging_t = F.pad(torch.from_numpy(merging), pad2d)
        side_full = F.pad(self.side_map, pad2d)

        r0, r1 = peak_r, peak_r + self.patch_h
        c0, c1 = peak_c, peak_c + self.patch_w
        merging_patch = merging_t[r0:r1, c0:c1]
        side_patch    = side_full[:, r0:r1, c0:c1]

        eps = 1e-8
        mp = merging_patch.clamp_min(0)
        mp = mp / (mp.sum() + eps)
        
        # 返回： patch输入, side_map输入, 完整的原始信号, 信息字典
        return mp.unsqueeze(0).float(), side_patch.float(), merging, info

# ----------------------
# 模型
# ----------------------

class AsymmetricPatchNetV4(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(2, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, out_channels, 1)
        )
    def forward(self, x):
        return self.net(x)

# ----------------------
# 硬先验
# ----------------------

def build_hard_mask_from_side(merging, side, tau=1e-3, top2_rows=True):
    pair_mask = (side.abs() > 0).float()
    row_sum = merging.sum(dim=3, keepdim=True)
    row_presence = (row_sum > tau).float()
    row_mask = row_presence.expand_as(pair_mask)
    if top2_rows:
        flat = row_sum.view(row_sum.size(0), -1)
        thr = torch.topk(flat, k=2, dim=1).values[:, -1].view(-1,1,1,1)
        top2 = (row_sum >= thr).float().expand_as(pair_mask)
        row_mask = row_mask * top2
    mask = pair_mask * row_mask
    empty = (mask.sum(dim=(2,3), keepdim=True) == 0)
    mask = torch.where(empty, pair_mask, mask)
    return mask

def masked_softmax(logits, mask, temperature=1.0):
    x = logits / max(temperature, 1e-6)
    x = x.masked_fill(mask == 0, float('-inf'))
    b, c, h, w = x.shape
    p = F.softmax(x.view(b, c, -1), dim=-1).view(b, c, h, w)
    bad = ~torch.isfinite(p).all(dim=(1,2,3), keepdim=True)
    if bad.any():
        uni = mask / (mask.sum(dim=(2,3), keepdim=True) + 1e-8)
        p = torch.where(bad, uni, p)
    return p

# ----------------------
# 坐标恢复
# ----------------------

def recover_absolute_coordinates_batch(merging_data_full_batch, relative_cx_batch, relative_cy_batch, patch_size=(3, 5)):
    b, h, w = merging_data_full_batch.shape
    device = merging_data_full_batch.device

    flat_indices = torch.argmax(merging_data_full_batch.view(b, -1), dim=1)
    peak_r_batch = flat_indices // w
    peak_c_batch = flat_indices % w

    patch_h, patch_w = patch_size
    h_radius = patch_h // 2
    w_radius = patch_w // 2

    top_left_abs_c_batch = peak_c_batch - w_radius
    top_left_abs_r_batch = peak_r_batch - h_radius

    absolute_cx_batch = top_left_abs_c_batch + relative_cx_batch.to(device)
    absolute_cy_batch = top_left_abs_r_batch + relative_cy_batch.to(device)

    return absolute_cx_batch, absolute_cy_batch

# ----------------------
# 绘图函数
# ----------------------

def plot_and_save_results(filename, data_points, plot_dir):
    """
    为单个文件绘制四个点集的散点图并保存。
    """
    # 提取绘图所需数据
    X_coords = [item['X_transformed'] for item in data_points]
    X_hat_coords = [item['X_hat_unprocessed'] for item in data_points]
    tx_temps = [item['tx_temp'] for item in data_points]
    rx_temps = [item['rx_temp'] for item in data_points]
    nm_tx_temps = [item['nm_tx_temp'] for item in data_points]
    nm_rx_temps = [item['nm_rx_temp'] for item in data_points]


    plt.figure(figsize=(12, 8))
    
    # 绘制解耦后的数据点集 (蓝色圆点)
    plt.scatter(X_coords, tx_temps, label='(Decoupled X, tx_temp)', alpha=0.7, s=50, c='blue')
    
    # 绘制原始温度数据点集 (红色叉)
    plt.scatter(rx_temps, tx_temps, label='(Merging rx_temp, tx_temp)', alpha=0.7, s=50, c='red', marker='x')

    # 绘制未解耦的数据点集 (绿色三角)
    plt.scatter(X_hat_coords, tx_temps, label='(Unprocessed X, tx_temp)', alpha=0.6, s=50, c='green', marker='^')
    
    # 绘制来自nonmerging文件夹的点集 (紫色方块)
    plt.scatter(nm_rx_temps, nm_tx_temps, label='(Nonmerging rx_temp, tx_temp)', alpha=0.7, s=50, c='purple', marker='s')

    # 设置图表属性
    plt.title(f'Coordinate vs. Temperature for {filename}', fontsize=16)
    plt.xlabel('X Coordinate / RX Temperature', fontsize=12)
    plt.ylabel('TX Temperature', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 构建保存路径并保存图像
    base_name = os.path.splitext(filename)[0]
    output_path = os.path.join(plot_dir, f"{base_name}_plot.png")
    
    try:
        plt.savefig(output_path, dpi=150)
    except Exception as e:
        print(f"Error saving plot {output_path}: {e}")
    finally:
        plt.close() # 释放内存

# ----------------------
# 推理与保存
# ----------------------

@torch.no_grad()
def run_inference_and_save_results(model_path, loader, device, temp=1.0, output_dir="inference_results"):
    # 1. 加载模型
    model = AsymmetricPatchNetV4(in_channels=1, out_channels=1)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Model loaded from {model_path}")

    # 2. 初始化结果存储字典
    results_by_file = defaultdict(list)

    # 3. 遍历所有数据进行推理
    for mp, sidep, merging, info in tqdm(loader, desc="Running Inference"):
        # 将数据移动到设备
        mp = mp.to(device, non_blocking=True)
        sidep = sidep.to(device, non_blocking=True)
        merging = merging.to(device, non_blocking=True)

        # === 解耦后坐标计算 (After) ===
        logits = model(mp)
        mask   = build_hard_mask_from_side(mp, sidep, tau=1e-3)
        p      = masked_softmax(logits, mask, temperature=temp)
        cx_pred_rel = calculate_centroid_x_torch(p)
        cy_pred_rel = calculate_centroid_y_torch(p)
        abs_cx, abs_cy = recover_absolute_coordinates_batch(merging, cx_pred_rel, cy_pred_rel)
        X_transformed = 64 * abs_cx + 32

        # === 未解耦坐标计算 (Before) ===
        # 在完整的32x18信号上计算重心，需要先增加一个channel维度
        cx_before_abs = calculate_centroid_x_torch(merging.unsqueeze(1))
        X_hat_unprocessed = 64 * cx_before_abs + 32

        # 4. 收集结果
        batch_size = mp.shape[0]
        abs_cx_list = abs_cx.cpu().tolist()
        abs_cy_list = abs_cy.cpu().tolist()
        X_transformed_list = X_transformed.cpu().tolist()
        X_hat_unprocessed_list = X_hat_unprocessed.cpu().tolist()
        
        for i in range(batch_size):
            filename = info['source_file'][i]
            point_id = info['id'][i]
            
            result_data = {
                'id': point_id,
                'absolute_cx': abs_cx_list[i],
                'absolute_cy': abs_cy_list[i],
                'X_transformed': X_transformed_list[i],
                'X_hat_unprocessed': X_hat_unprocessed_list[i],
                'tx_temp': info['tx_temp'][i].item(),
                'rx_temp': info['rx_temp'][i].item(),
                'nm_tx_temp': info['nm_tx_temp'][i].item(),
                'nm_rx_temp': info['nm_rx_temp'][i].item(),
                'sum_val': info['sum_val'][i].item()
            }
            results_by_file[filename].append(result_data)

    # 5. 将坐标结果保存到JSON文件
    os.makedirs(output_dir+'/res', exist_ok=True)
    print(f"\nSaving coordinate results to directory: {output_dir}")

    for filename, data_points in results_by_file.items():
        base_name = os.path.splitext(filename)[0]
        output_filename = f"res/{base_name}_coordinates.json"
        output_path = os.path.join(output_dir, output_filename)
        
        data_points.sort(key=lambda x: int(x['id']))
        final_output = {item['id']: item for item in data_points}
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=4, ensure_ascii=False)
        print(f"  - Saved coordinates for {filename} to {output_path}")

    # 6. 为每个文件生成并保存绘图
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"\nGenerating and saving plots to directory: {plot_dir}")
    
    for filename, data_points in tqdm(results_by_file.items(), desc="Generating Plots"):
        plot_and_save_results(filename, data_points, plot_dir)

# ----------------------
# 主程序
# ----------------------

def main():
    # 路径
    ALIGNED_DATA_DIR = '/work/hwc/SENSOR_STAGE2/training_data/test_line/norm_data'
    NONMERGING_DATA_DIR = '/work/hwc/SENSOR_STAGE2/training_data/test_line/nonmerging'
    MODEL_SAVE_PATH  = 'runs/v5_best_kl_57.2.pth' 

    # 配置
    BATCH_SIZE  = 256 # 可以适当调大以加快推理速度
    NUM_WORKERS = 4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # --- 数据集 ---
    dataset = TouchDatasetJSON(
        data_dir=ALIGNED_DATA_DIR,
        nonmerging_data_dir=NONMERGING_DATA_DIR,
        patch_size=(3,5)
    )
    if not dataset.samples:
        print("Error: No data found in the specified directory.")
        return

    use_pin_memory = (device.type == 'cuda')
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=NUM_WORKERS, pin_memory=use_pin_memory
    )

    # --- 运行推理并保存结果 ---
    run_inference_and_save_results(
        model_path=MODEL_SAVE_PATH,
        loader=loader,
        device=device,
        temp=1.0
    )

if __name__ == '__main__':
    main()

