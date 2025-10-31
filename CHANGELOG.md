# Changelog — 排序稳定性修复 (2025-10-30)

说明：
本次提交修复了 C 与 Python 在稀疏特征（top-K）选择时因排序不稳定导致的输入槽（feature slot）错位问题。该问题引发了若干样本在整数量化路径上产生差异（最终输出在量化单元上出现较大偏差）。

影响文件：
- infer/inference.c
  - 修改：`compare_value_index_desc` comparator，加入明确的 tie-break（当值接近或相等时按原始索引排序）并增加注释说明。
  - 目的：确保 C 端对稀疏点按稳定顺序选择 top-K，避免 qsort 的不确定性在相等或近似值时引入位置交换。

- infer/validate_c_inference.py
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