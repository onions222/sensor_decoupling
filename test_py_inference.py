import numpy as np
import torch
import torch.ao.quantization as tq
from pathlib import Path
import sys
import os

# 添加当前目录到Python路径
sys.path.append('.')

# 从sparse_res.py导入必要的类和函数
from sparse_res import SparseCompensationMLP, TouchJSONDataset, SparseCompensationDataset

# 配置
MODEL_INT8_PATH = Path("runs_sparse_v2_3x6_qat_minmax/best_qat_int8.pt")
QENGINE = "fbgemm"
K_SPARSE_POINTS = 8
WIDTH = 8

def main():
    # 加载模型
    print("Loading PyTorch INT8 model...")
    model_def = SparseCompensationMLP(k_points=K_SPARSE_POINTS, width=WIDTH, hidden=16)
    model_def.fuse_model()
    model_load = model_def.cpu()
    model_load.qconfig = tq.get_default_qat_qconfig(QENGINE)
    tq.prepare_qat(model_load, inplace=True)
    model_load.eval()
    model_int8 = tq.convert(model_load, inplace=False)
    model_int8.eval()
    state_dict = torch.load(MODEL_INT8_PATH, map_location='cpu')
    model_int8.load_state_dict(state_dict)
    print("Model loaded successfully.")
    
    # 加载测试数据
    base_dataset = TouchJSONDataset("val_data")
    dataset = SparseCompensationDataset(base_dataset)
    sample_data = dataset[0]  # 获取第一个样本
    
    x = sample_data["x"]
    meta = sample_data["meta"]
    raw_g = sample_data["raw_g"]
    file_str = sample_data["file"]
    
    print(f"Processing sample from file: {file_str}")
    print(f"Input x shape: {x.shape}")
    print(f"Meta shape: {meta.shape}")
    print(f"Raw global CoG: {raw_g}")
    
    # 执行推理
    with torch.no_grad():
        model_int8.eval()
        pred_offset = model_int8(x.unsqueeze(0), meta.unsqueeze(0))
        pred_offset = pred_offset.squeeze(0)
        
    dx, dy = pred_offset[0].item(), pred_offset[1].item()
    print(f"Predicted offset: dx={dx:.8f}, dy={dy:.8f}")
    
    # 计算最终CoG
    xc_py = raw_g[0].item() + dx
    yc_py = raw_g[1].item() + 0.0 * dy  # ALPHA_Y = 0.0
    
    print(f"Final CoG: x={xc_py:.8f}, y={yc_py:.8f}")

if __name__ == "__main__":
    main()