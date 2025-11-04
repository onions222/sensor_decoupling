# Quick Start — C 整数推理（快速上手）

如果你只想立即用 C 端的整数推理运行并得到最终预测结果（不导出中间 JSON），请按下面步骤：

1. 进入 `infer/` 并编译：

```bash
cd /work/hwc/SPARSE/infer
make -j
```

2. 运行单个样本（示例：文件 `val_data/aligned_g26.json` 中的样本 ID=1）：

```bash
./inference_app val_data/aligned_g26.json 1
# 输出为一行 `xc,yc`（最终修正后的 CoG）
```

3. （可选）只导出 C 端的整数中间层以便比对：

```bash
export DUMP_INTEGER_PATH=./intermediates/int_only_id1_c.json
./inference_app val_data/aligned_g26.json 1
```

4. （可选）导出完整的中间层（浮点 + 整数），以便与 Python 的中间层逐项比对：

```bash
export DUMP_INTERMEDIATES_PATH=./intermediates/inter_id1_c.json
./inference_app val_data/aligned_g26.json 1
```

以上命令使用的是 C 的整数推理路径（默认行为）。如果你在运行时设置了 `DUMP_INTERMEDIATES_PATH`，C 端会在导出 JSON 时使用与 Python 相同的 dequantize→compute→requantize 路径来保证中间结果可比（这用于调试/验证）。

# Inference & Validation Guide — infer/

本文档说明如何在本仓库中编译并运行 C 推理二进制，如何与 Python（PyTorch INT8）参考实现比较，以及常用的调试与中间结果导出命令。

目录
- 环境与依赖
- 构建 C 推理二进制
- 快速运行（单样本）
- 导出中间结果（JSON）
- 导出整数中间结果（整数序列）
- 使用 Python 验证（validate_c_inference.py）
- 可选的调试/编译开关说明
- 中间 JSON 文件格式说明
- 常见问题与排查建议

## 环境与依赖

建议使用 Python 3.8+（示例在 Ubuntu / conda 环境下验证）：
- Python 包（可用 pip/conda 安装）：
  - numpy
  - torch (包含 quantization API)
  - cJSON 源码位于 `infer/cJSON.c`，C 端使用此库来写 JSON（无需额外安装）

示例（使用 pip）:

```bash
python3 -m pip install --user numpy torch
```

注意：PyTorch 的版本会影响量化工具链。仓库中使用的转换/导出脚本（`export_weights.py` / `validate_c_inference.py`）假定训练/导出使用了 QAT/convert 路径。如果你遇到加载 model 的问题，请确认 PyTorch 版本与模型导出时的版本兼容。

## 构建 C 推理二进制

进入 `infer/` 目录并运行：

```bash
cd /work/hwc/SPARSE/infer
make -j
```

成功后会生成可执行文件 `inference_app`（链接了 `main.o`, `inference.o`, `model_weights.o`, `cJSON.o`）。

如果你需要启用额外的编译宏（例如启用调试打印），可以直接手动编译或临时修改 Makefile。例如要开启 `DEBUG_INT8_CHECK`：

```bash
# 直接用 gcc 编译（示例，Makefile 也可以改）
gcc -Wall -Wextra -O2 -I. -DDEBUG_INT8_CHECK main.c inference.c model_weights.c cJSON.c -o inference_app -lm
```

## 快速运行（单样本）

基本命令（C 程序期望参数：JSON 文件、样本 ID）：

```bash
# 在 infer/ 下
./inference_app val_data/aligned_g26.json 0
```

该命令会在 stdout 输出最终预测的 corrected CoG（xc,yc），以及若启用了调试宏还会打印额外信息。

## 导出中间结果（JSON）

若要让 C 程序把与 Python 参考实现结构一致的中间层（dequantized floats & integer representations）写入 JSON，用环境变量 `DUMP_INTERMEDIATES_PATH` 指定输出路径：

```bash
# 导出中间浮点/整数值
export DUMP_INTERMEDIATES_PATH=./intermediates/inter_test_sample_c.json
./inference_app val_data/aligned_g26.json 1
# 结果写入 ./intermediates/inter_test_sample_c.json
```

`validate_c_inference.py` 会自动为 Python 与 C 产生相同结构的 JSON，并将其写入 `--dump-dir` 指定的目录。

## 导出整数中间结果（仅整数表示）

若你想仅导出 C 端的整数中间数组（`net_0_q_int`, `net_2_q_int`, `net_4_q_int`），可以用 `DUMP_INTEGER_PATH`：

```bash
export DUMP_INTEGER_PATH=./inter_int_c.json
./inference_app val_data/aligned_g26.json 1
# 输出文件 ./inter_int_c.json 会包含 net_?_q_int 数组
```

注意：`DUMP_INTERMEDIATES_PATH` 与 `DUMP_INTEGER_PATH` 均受程序内检查控制，若同时设置，两者都会写出相应文件（`DUMP_INTERMEDIATES_PATH` 包含更丰富的浮点信息）。

## 使用 Python 验证（validate_c_inference.py）

仓库提供 `validate_c_inference.py` 来：
- 在 Python 中运行 INT8 模型（捕获 Python 侧的中间层），
- 调用 C 可执行文件（可让 C 导出中间层），
- 比较最终输出是否一致（默认比较 xc,yc 浮点值，允许设置 tolerance）。

示例：在 `infer/` 下运行前 10 个样本并导出中间结果：

```bash
python3 validate_c_inference.py --limit 10 --dump-intermediates --c_exe ./inference_app --dump-dir ./intermediates
```

说明：
- `--c_exe` 指定 C 可执行文件路径；
- `--dump-intermediates` 会令 Python 把它捕获的中间层写入 `--dump-dir` 指定的目录，并在调用 C 时通过环境变量告知 C 也写入对应的 JSON；
- 若不指定 `--limit` 会验证全部验证集样本（较慢）。

运行后你会看到每个样本的 `MATCH` / `MISMATCH` 报告，和 validation 汇总。

## 可选的调试/编译开关（在 C 代码中可用）

infer 中实现了若干辅助开关用于调试（通过在编译时定义宏实现）：

- `DEBUG_INT8_CHECK`：打印各层的 debug 信息（包括 acc32、quantized multipliers、bias_int），便于追踪整数路径与浮点参考的差异。
- `FORCE_FLOAT_REQUANT`：在整数累加后使用浮点路径做 requantization（可用来验证 float 路径是否精确匹配 Python）。
- `FORCE_FULL_FLOAT_MLP`：直接将整个 MLP 在浮点上重算（使用 dequantized weights/activations），此路径与 Python 的浮点参考最接近，用于确认权重/偏置导出是否正确。

示例：用 gcc 直接编译并开启 `DEBUG_INT8_CHECK`：

```bash
gcc -Wall -Wextra -O2 -I. -DDEBUG_INT8_CHECK main.c inference.c model_weights.c cJSON.c -o inference_app -lm
```

## 中间 JSON 文件格式说明

`validate_c_inference.py` 与 C 的 `DUMP_INTERMEDIATES_PATH` 会生成结构一致的 JSON，关键字段：

- `after_quant_in_q_int`: [[...]] — MLP 输入的整数表示（batch dim 1）
- `after_quant_in_q_scale`, `after_quant_in_q_zero_point` — 输入的量化参数
- `after_quant_in`: [[...]] — dequantized float 输入（batch dim 1）
- `net_0`, `net_1`, `net_2`, `net_3`, `net_4` — 每层的 dequantized float activations (net_1 mirrors net_0 for fused LinearReLU)
- `net_0_q_int`, `net_2_q_int`, `net_4_q_int` — 对应的整数表示（array of ints)
- `net_?_q_scale`, `net_?_q_zero_point` — 对应层的量化参数
- `after_dequant_out`, `final` — 最终 dequantized logits 与 tanh-scaled outputs
- `baseline_cog` — baseline center-of-gravity 由前置预处理计算，便于比较预处理一致性

这些文件可直接用 diff / jq / Python 脚本比较对齐位置和逐元素差异。

## 常见问题与排查建议

1. 若 C 与 Python 在某些样本上不一致：
   - 先比较 `after_quant_in_q_int`（输入量化整数表示）。若不一致，问题通常出现在稀疏点选择、归一化或量化实现顺序上（例如排序 tie-break 或 quantize rounding）。
   - 若 `after_quant_in_q_int` 一致，再比较 `net_0_q_int` 等中间整数输出。差异可能由乘法后移位的舍入策略或 bias 量化导致。
   - 可以启用 `DEBUG_INT8_CHECK` 并用单样本执行以打印逐项贡献（in_idx 的 int 和 float 贡献），帮助定位问题源头。

2. 如果 Python 无法加载 INT8 模型（torch.load 失败）：请检查 PyTorch 版本与模型导出时的版本一致性，或尝试使用 `weights_only=True`（如果模型是仅权重存档）。

3. 如果 `intermediates/` 没有生成预期 JSON：确保 `DUMP_INTERMEDIATES_PATH` 环境变量已正确设置并且 C 可执行拥有写权限的目标目录。

## 进阶：如何导出权重为 C（如果需要重导出）

仓库包含 `export_weights.py`，用于将 PyTorch INT8 模型的 `_packed_params`（量化权重 + 偏置）导出为 `model_weights.c/.h`：

```bash
# 在仓库根或 infer/ 下运行（脚本默认路径已在文件中硬编码）
python3 export_weights.py
```

导出完成后，重新构建 C 程序：

```bash
make -C infer clean && make -C infer -j
```

## 联系与后续

如果你希望我：
- 运行全量验证并生成完整 mismatch 报表（可能耗时）；
- 将修复说明并入 `README.md` 的顶部或项目根 README；
- 在 C 中替换 qsort 为自带稳定 top-K 选择实现以完全消除排序依赖；
请告诉我你的偏好，我会继续执行。

## Changelog — 排序稳定性修复 (2025-10-30)

说明：
本次提交修复了 C 与 Python 在稀疏特征（top-K）选择时因排序不稳定导致的输入槽（feature slot）错位问题。该问题引发了若干样本在整数量化路径上产生差异（最终输出在量化单元上出现较大偏差）。

影响文件：
- `infer/inference.c`
  - 修改：`compare_value_index_desc` comparator，加入明确的 tie-break（当值接近或相等时按原始索引排序）并增加注释说明。
  - 目的：确保 C 端对稀疏点按稳定顺序选择 top-K，避免 qsort 的不确定性在相等或近似值时引入位置交换。

- `infer/validate_c_inference.py`
  - 修改：在计算 Python 侧的 top-K 时使用 `np.argsort(..., kind='stable')`，保证 Python 端排序为稳定排序，从而与 C 端的稳定比较器一致。

验证：
- 在 `infer/` 目录执行：

```bash
make -C infer -j
python3 infer/validate_c_inference.py --limit 10 --dump-intermediates --c_exe ./infer/inference_app
```

运行结果：10/10 样本匹配（Validation PASSED）。

根本原因：
- Python 默认的 `np.argsort` 在某些实现/版本中并非稳定（取决于数据与实现），而 C 使用的 `qsort` 也不保证稳定排序。两个端对于等值或极其接近的 value 没有统一的 tie-break，导致在稀疏点排序时出现不一致，从而在位置敏感的 MLP 中产生传播性误差。

修复策略与建议：
1. 已采用的策略：在两端都引入确定性的排序规则（Python 使用 stable argsort，C 在比较函数中以原始索引做次级键），这是最低风险且可移植的修复。
2. 运行时兼容性：如果你将来改用不同的 numpy 版本或替换排序实现，请确保 `validate_c_inference.py` 中仍保留 `kind='stable'`，或在导出/推理阶段显式记录并使用相同的稳定顺序。
3. 生产化建议：如果需要完全让 C 端独立于 Python，可让 C 在选取 top-K 时实现自己稳定选择算法（例如扫描并收集 top-K，同时在 tie 时保持扫描顺序），或把已排序的 input order 作为预处理固定导出。

补充：
- 本次提交没有更改模型权重的导出逻辑（`export_weights.py`），该脚本导出的权重与 C 访问的内存布局是匹配的。

如需我：
- 追加将此说明写入 `infer/README.md` 或在代码中添加更多注释；
- 对全部验证集运行完整检查（会更慢，但更充分）；
- 将 C 中的 comparator 改写为自带稳定排序实现（避免依赖 qsort）；
请告诉我你想做哪一项。

---
文件：`/work/hwc/SPARSE/infer/README.md` 已创建，包含上述步骤与命令示例。