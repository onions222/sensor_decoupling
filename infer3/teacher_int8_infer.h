#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Patch 尺寸（和 Python Teacher 一致）
#define PATCH_H 3
#define PATCH_W 5

// odd / even 分支的整型推理接口
// in_patch:  输入 float[PATCH_H * PATCH_W]
// out_patch: 输出 float[PATCH_H * PATCH_W]
void teacher_int8_forward_odd(const float* in_patch, float* out_patch);
void teacher_int8_forward_even(const float* in_patch, float* out_patch);

#ifdef __cplusplus
}
#endif