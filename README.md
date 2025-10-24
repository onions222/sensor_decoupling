传感器去耦偏移（Sensor Decoupling Offset）- 推理与验证工程

概述
- 本项目包含一个用于二维小补丁输入的轻量模型（OffsetNetWithMeta），通过量化感知训练（QAT）后导出到 C 端，在嵌入式/PC 上进行纯定点推理与验证。
- 提供从 PyTorch QAT 模型导出权重/量化参数到 C（`weights.[ch]`、`model_meta.h`），以及一套可在 PC 上验证的流程：生成 Python 端黄金数据、运行 C 端推理、逐层对齐比对。

目录结构（关键）
- `convert/offset/`
  - `tools/`
    - `export_weights.py`：从 QAT 模型导出 C 端权重与量化元数据。
    - `verify_with_python.py`：按 FX 图模拟量化前向，导出浮点输入/输出与各中间层的整型“黄金数据”。
  - `src/`
    - `inference_core.c/.h`：核心整型算子与前向（Conv/Linear/GAP/Cat/Requant）。
    - `inference.c/.h`：对外 API（初始化、float 封装、纯量化接口）。
    - `weights.c/.h`、`model_meta.h`：由导出脚本生成的权重与量化参数。
  - `include/config.h`：日志等级、容错阈值、dump 配置等。
  - `tests/verify.c`：PC 端验证程序（加载黄金数据、运行 C、逐层与最终结果比对）。
  - `scripts/run_verify.sh`：一键导出→编译→生成黄金→验证 的脚本。
  - `verify_data/`：样例黄金数据输出目录（Python 生成）。
  - `c_intermediate_outputs/`：C 端中间层 dump 目录（Debug 构建自动生成）。

环境准备
- Python 3.8+，依赖：`torch`、`numpy`（训练/导出环境需能加载量化模型）。
- 构建工具：`gcc` 与 `make`（PC 验证）。
- QAT 权重：`best_int8.pt`（或你的量化权重路径）。
- 输入数据：与训练一致的 JSON 数据（用于生成黄金数据）。

快速开始（推荐脚本）
1) 检查并修改脚本中的路径（非常重要）
- 文件：`convert/offset/scripts/run_verify.sh`
  - `MODEL_FILE_INT8`：你的量化权重路径（默认指向示例路径）。
  - `DATA_DIR` 与 `DATA_FILE`：你的输入 JSON 目录与文件名。

2) 一键运行
- 在项目根目录执行：
  - `bash convert/offset/scripts/run_verify.sh [可选:your.json]`
- 流程说明：
  - 导出 C 端权重/量化元数据到 `convert/offset/src/`。
  - 使用 Debug 配置编译验证程序（启用中间层 dump）。
  - 生成 Python 黄金数据到 `convert/offset/verify_data/`。
  - 运行 C 端验证：逐层（uint8）与最终（float，经后处理）对齐比对。

手动流程（如需定制）
- 导出权重与元数据
  - `python3 convert/offset/tools/export_weights.py --model-in best_int8.pt --out-dir convert/offset/src`
- 编译（Release 或 Debug）
  - Release：`make -C convert/offset verify`
  - Debug（开启中间层 dump）：`make -C convert/offset verify_debug`
- 生成黄金数据
  - `python3 convert/offset/tools/verify_with_python.py --model-in best_int8.pt --data-dir <dir> --data-file <file.json> --out-dir convert/offset/verify_data`
- 运行验证
 - `convert/offset/build/verify convert/offset/verify_data`

直接推理（CLI）
- 构建 CLI
  - `make -C convert/offset infer`
- 运行（两种模式）
  - 浮点模式（默认，内部自动量化→推理→反量化与后处理）
    - `convert/offset/build/infer --x <x_f32.bin> --meta <meta_f32.bin> [--out out_f32.bin]`
    - 示例（复用验证生成的输入）：
      - `convert/offset/build/infer --x convert/offset/verify_data/input_x.bin --meta convert/offset/verify_data/input_meta.bin --out convert/offset/verify_data/infer_out.bin`
    - 输入/输出格式：
      - `x_f32.bin`：float32，长度 `MODEL_INPUT_X_SHAPE_SIZE`（[1,2,3,7]）。
      - `meta_f32.bin`：float32，长度 `MODEL_INPUT_META_SHAPE_SIZE`（[1,2]）。
      - 输出 `out_f32.bin`：float32（dx, dy），长度 `MODEL_OUTPUT_SHAPE_SIZE`。
  - 量化模式（纯 uint8 输入/输出）
    - `convert/offset/build/infer --quantized --x <x_u8.bin> --meta <meta_u8.bin> [--out-q out_u8.bin] [--out out_f32.bin]`
    - 功能：直接喂入 uint8 输入，输出 uint8，同时打印并可保存反量化后处理的 float 结果。
    - 输入/输出格式：
      - `x_u8.bin`：uint8，长度 `MODEL_INPUT_X_SHAPE_SIZE`。
      - `meta_u8.bin`：uint8，长度 `MODEL_INPUT_META_SHAPE_SIZE`。
      - `out_u8.bin`：uint8，长度 `MODEL_OUTPUT_SHAPE_SIZE`（可选，通过 `--out-q` 保存）。
      - `out_f32.bin`：float32，长度 `MODEL_OUTPUT_SHAPE_SIZE`（可选，通过 `--out` 保存，值为经 tanh 与缩放的最终输出）。

文件/参数说明与小贴士
- `include/config.h`
  - `INFERENCE_C_LOG_LEVEL`：日志等级（3=INFO 验证摘要，4=DEBUG 算子级，5=TRACE 逐元素）。
  - `DUMP_INTERMEDIATES` 与 `INTERMEDIATE_DUMP_DIR`：中间层 dump 开关与目录。Debug 目标已自动开启。
  - 误差容忍：`INFERENCE_C_VERIFY_TOLERANCE_ABS_FLOAT/REL_FLOAT`、`INFERENCE_C_VERIFY_TOLERANCE_ABS_UINT8`。
- `tools/verify_with_python.py`
  - 将输入按 FX 图量化流程处理，保存：`input_x.bin`、`input_meta.bin`、`golden_output.bin` 及各中间层 `golden_*.bin`（多为 `uint8`）。
  - 需能正确加载你的 QAT 图与 state_dict；如 FX 命名/融合策略有差异，需在脚本里同步适配。
- `tools/export_weights.py`
  - 读取 QAT 转换图，导出 `weights.[ch]` 和 `model_meta.h`；内含 per-tensor/per-channel 的 requant 参数计算逻辑（TFLite 风格 multiplier/shift，round-half-up）。
  - 如果你的模型结构或量化节点命名不同，也需要同步适配此脚本的取参逻辑。

C 侧 API（嵌入式集成）
- 头文件：`convert/offset/src/inference.h`
- 主要函数：
  - `int inference_init(void);` 初始化（如清理 arena）。
  - `int inference_run_quantized(const uint8_t* x_q, const uint8_t* meta_q, uint8_t* out_q);`
  - `int inference_run_float(const float* x_f, const float* meta_f, float* out_f);` 浮点封装：内部按导出量化参数执行量化→整型推理→反量化与后处理（tanh 与缩放）。
- MCU 端集成时：
  - 将 `src/*.c/.h` 与 `include/config.h` 拷入工程；
  - 依据目标芯片设置编译选项（如 `-mcpu/-mthumb/-mfpu`），禁用文件 IO 与高日志等级。

常见问题
- 路径不存在或模型加载失败：请先确认 `run_verify.sh` 中 `MODEL_FILE_INT8` 与 `DATA_DIR/FILE` 可用；或按“手动流程”逐步执行定位问题。
- 中间层不匹配：
  - 确保使用 Debug 构建（会编译 `-DDUMP_INTERMEDIATES` 并在 `c_intermediate_outputs/` 生成 C dump）。
  - 确保 Python 端量化/重量化逻辑与 C 端一致（尤其 GAP 的 dequant→平均→re-quant 精度、cat 两路 qparam 一致）。
- 量化参数缺失/命名不一致：根据你的 FX 图调整 `export_weights.py` 与 `verify_with_python.py` 中的取参代码。

许可与致谢
- 本仓库用于工程化验证与移植示例。若需要开源协议说明，请在此处补充。
