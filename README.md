## Float Teacher → QAT → Python Int8 → C 整数推理全流程
---
#### Quick start
1. 训练 Teacher 模型
```python
python teacher_train.py
```

输出：distill/decoupler_model_v16_teacher_best.pth

2. 对 Teacher 做 QAT 微调训练
```python
python teacher_qat_train.py
```

输出：distill/decoupler_model_v16_teacher_best_qat_fp32.pth

脚本会自动评估：

	•	Float Teacher vs QAT Teacher
	•	Float Teacher vs QAT-int8（Python 仿真）

3. 导出 QAT Int8 权重（给 Python / C 用）
```python
python teacher_qat_export_int8.py
```

输出：

	•	distill/decoupler_model_v16_teacher_best_qat_int8_params.pt
	•	distill/decoupler_model_v16_teacher_best_qat_act_calib.pt

4. Python 中验证 Int8 推理精度
```python
python teacher_qat_int8_deploy_infer.py
```

如果 Python-int8 版本和浮点版误差很小 → 说明你的量化和 scale 完全正确

5. 导出 C 语言推理所需头文件
```python
python teacher_qat_export_c.py
```

输出到 infer3/：

	•	teacher_int8_params.h
	•	teacher_int8_scales.h
	•	teacher_int8_shapes.h

这些文件包含：

	•	权重 int8
	•	偏置 float
	•	w_scale / x_scale / zero point
	•	卷积核尺寸、stride、padding

6. 生成 C 语言对齐验证数据（val 样本）
```python
python teacher_generate_debug_data.py
```

输出到：

	•	infer3/debug_data.h

包含：

	•	输入 patch
	•	is_odd
	•	Python-int8 输出 patch
	•	粗峰坐标
	•	Python 计算的 全局坐标（18列）

7. 编译并验证 C 推理结果

```shell
cd infer3
make
./test_infer
```


