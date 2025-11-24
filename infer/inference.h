#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stdio.h>
#include "weights.h"
#include "input_sample.h"

// 简单的 Tensor 结构体，用于管理缓冲区
typedef struct {
    int8_t* data;
    int channels;
    int height;
    int width;
} Tensor;

// 工具函数：限制数值范围 (Clamp)
static inline int32_t clamp(int32_t x, int32_t min_val, int32_t max_val) {
    if (x < min_val) return min_val;
    if (x > max_val) return max_val;
    return x;
}

// 核心数学函数：Re-quantization
// 计算: output = (input * multiplier) >> shift
// 模拟定点小数乘法
static inline int32_t multiply_by_quantized_multiplier(int32_t x, int32_t multiplier, int shift) {
    // MCU 优化提示: 这里使用 64 位乘法。
    // 在 Cortex-M4/M7 上，SMULL 指令可以高效完成。
    int64_t total = (int64_t)x * (int64_t)multiplier;
    
    // 加上 Rounding 偏移量 (1 << 30) 用于四舍五入
    int64_t val = total + (1LL << 30); 
    
    // 取高 32 位 (相当于乘 2^-31)
    int32_t result = (int32_t)(val >> 31);
    
    // 应用 shift (Shift 对应 31 - exponent)
    // 注意：export_system.py 计算的 shift 是正数，表示右移
    if (shift > 0) {
        // 带舍入的右移，使用 64 位掩码以避免在 32 位上移位超限的未定义行为
        int64_t mask = 1LL << (shift - 1);
        int64_t tmp = (int64_t)result + mask;
        result = (int32_t)(tmp >> shift);
    } else if (shift < 0) {
        result = result << (-shift);
    }
    
    return result;
}

// 函数声明
void quantize_input(const float* input_float, int8_t* output_int8, int count, float scale, int32_t zp);
// conv2d_layer 增加 input_zp 参数以支持非对称量化 (input zero point)
void conv2d_layer(const Tensor* input, Tensor* output, 
                  const int8_t* weights, const int32_t* bias,
                  int k_h, int k_w, int pad, int stride,
                  int32_t multiplier, int shift, int32_t input_zp, int32_t out_zp,
                  int32_t out_min, int32_t out_max);
float dequantize_output(int8_t val, float scale, int32_t zp);

#endif // INFERENCE_H