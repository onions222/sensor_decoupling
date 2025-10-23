#!/bin/bash

# 发生错误时立即退出
set -e -o pipefail # Exit on error and report error in pipes

# 设置目录
ROOT_DIR=$(pwd)
SRC_DIR="src"
BUILD_DIR="build"
VERIFY_DATA_DIR="verify_data"
TOOLS_DIR="tools"
DATA_DIR="data" # Directory for input JSON
DATA_FILE="aligned_g26.json" # Default input file

# --- 修正: 指向您新训练的、包含有效 QParams 的模型文件 ---
# (基于您的 train_offset_QAT.py 日志输出)
MODEL_FILE_INT8="/work/hwc/SENSOR_STAGE3/runs_offset_v12_final/best_int8.pt"

C_DUMP_DIR="c_intermediate_outputs" # Must match config.h

# --- 允许通过命令行参数指定数据文件 ---
if [ "$#" -ge 1 ]; then
    DATA_FILE="$1"
    echo "Using specified data file: $DATA_FILE"
fi
DATA_PATH="$DATA_DIR/$DATA_FILE"


# 清理旧的验证数据和 C dumps
echo "=============== 0. Cleaning previous run ==============="
rm -rf ${VERIFY_DATA_DIR} ${C_DUMP_DIR} # 清理两个目录
mkdir -p ${VERIFY_DATA_DIR} ${C_DUMP_DIR} # 重新创建


# 检查依赖
if [ ! -f "$MODEL_FILE_INT8" ]; then
    echo "错误: 未找到 $MODEL_FILE_INT8。"
    echo "请确保您已经使用最新的 train_offset_QAT.py 成功训练，"
    echo "并且该文件位于: ${MODEL_FILE_INT8}"
    exit 1
fi
if [ ! -f "$DATA_PATH" ]; then
    echo "错误: 未找到数据文件 $DATA_PATH。"
    echo "请确保输入 JSON 文件存在于 $DATA_DIR/ 目录。"
    exit 1
fi

echo "=============== 1. [Python] 导出 C 权重和元数据 ==============="
# (Run export first)
python3 ${TOOLS_DIR}/export_weights.py --model-in ${MODEL_FILE_INT8} --out-dir ${SRC_DIR}

echo "=============== 2. [C] 编译 C 验证程序 (Debug Mode) ==============="
# --- 修正: 使用 make verify_debug ---
make verify_debug

echo "=============== 3. [Python] 生成 Python 黄金数据 (包括中间层) ==============="
python3 ${TOOLS_DIR}/verify_with_python.py \
    --model-in ${MODEL_FILE_INT8} \
    --data-dir ${DATA_DIR} \
    --data-file ${DATA_FILE} \
    --out-dir ${VERIFY_DATA_DIR}

echo "=============== 4. [C] 运行 C 库验证 (生成 Dumps 并执行比较) ==============="
# The verify program now handles loading golden, running C (which dumps), loading C dumps, and comparing
${BUILD_DIR}/verify ${VERIFY_DATA_DIR} # Pass data dir as argument

echo "=============== 验证流程完成 ==============="
# The verify program's exit code indicates success (0) or failure (non-zero)

