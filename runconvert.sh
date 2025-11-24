#!/bin/bash
set -e

# 配置 TinyMaix 工具路径 (请修改为你实际 clone 的路径)
TM_PATH="./TinyMaix" 

# 设置环境变量避免重复导入问题
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3

# 1. 定义文件名
ONNX_MODEL="/work/hwc/SPARSE/distill/qat_student_runs/student_qat.onnx"
TFLITE_DIR="tflite_output"
TFLITE_MODEL="${TFLITE_DIR}/${ONNX_MODEL%.*}_float32.tflite" # onnx2tf 默认输出名可能带 float32 后缀，实际上是量化模型
HEADER_FILE="model_qat.h"

echo ">>> [Step 1] converting ONNX to TFLite Int8..."
# -osd: 尝试将特殊算子降级为标准 Int8 算子
# -oiqt: 强制输入/输出为 Int8 (Input/Output Int8 Quantization)
# 注意：onnx2tf 会自动读取 QDQ 节点并生成完全量化的 TFLite
onnx2tf -i "$ONNX_MODEL" -o "$TFLITE_DIR" -osd

echo ">>> [Step 2] Converting TFLite to TinyMaix Header..."
if [ ! -f "$TFLITE_MODEL" ]; then
    echo "Error: TFLite file not found at $TFLITE_MODEL"
    # 尝试查找目录下唯一的 .tflite 文件
    TFLITE_MODEL=$(find "$TFLITE_DIR" -name "*.tflite" | head -n 1)
    echo "Found: $TFLITE_MODEL"
fi

# 调用 TinyMaix 转换工具
# 参数: <tflite> <output_h> <format:h> <quant:int8> <in_dims> <out_dims>
cd "$TM_PATH/tools"
PYTHONPATH=. python3 -c "
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
sys.path.insert(0, '.')
import tflite2tmdl
sys.argv = ['tflite2tmdl.py', '$TFLITE_MODEL', '$HEADER_FILE', 'h', '1', '1,3,5,1', '1,1,1,1']
tflite2tmdl.main()
"
cd -

echo ">>> 转换完成！"
echo "生成的头文件: $HEADER_FILE"
echo "请检查 $HEADER_FILE 中的 TM_IN_SIZE 宏，确认输入 Buffer 大小。"