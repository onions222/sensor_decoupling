#ifndef INFERENCE_CORE_H
#define INFERENCE_CORE_H

#include "../include/config.h" // Includes acc_type_t, act_type_t, weight_type_t

// --- 算子参数 ---
typedef struct {
    // 形状
    int32_t H, W, C;         // 输入 (H, W, C)
    int32_t OH, OW, OC;      // 输出 (OH, OW, OC)
    int32_t K, S, P;         // Kernel, Stride, Padding
    int32_t G;               // Groups
    // 量化
    // --- 修正: Use act_type_t for zero points ---
    act_type_t  in_zp;       // 输入零点 (uint8)
    act_type_t  out_zp;      // 输出零点 (uint8)
    int         is_per_channel;
    int         relu;            // Use out_zp as lower bound for ReLU6
} ConvParams;

typedef struct {
    int32_t In;
    int32_t Out;
    // 量化
    // --- 修正: Use act_type_t for zero points ---
    act_type_t  in_zp;
    act_type_t  out_zp;
    int         is_per_channel;
    int         relu;
} LinearParams;

// --- 核心模型图 ---
/**
 * @brief 核心前向推理 (纯整型)
 * --- 修正: Use act_type_t for activation buffers ---
 */
int model_forward_s8( // Keep name for now, but uses uint8 internally
    const act_type_t* x_q,       // Quantized input X (uint8)
    const act_type_t* meta_q,    // Quantized input Meta (uint8)
    act_type_t* out_q,           // Quantized output (uint8)
    uint8_t* arena               // 临时工作区 (uint8_t is fine for generic buffer)
);

// --- 量化/反量化辅助函数 ---
// (仅用于 inference_run_float 封装)

/**
 * @brief 封装量化 (浮点->整型)
 * --- 修正: Output act_type_t ---
 */
void model_quantize_inputs(
    const float* x_f,
    const float* meta_f,
    act_type_t* x_q,      // uint8
    act_type_t* meta_q    // uint8
);

/**
 * @brief 封装反量化与后处理 (整型->浮点)
 * --- 修正: Input act_type_t ---
 */
void model_dequantize_and_postprocess(
    const act_type_t* out_q, // uint8
    float* final_out_f
);


#endif // INFERENCE_CORE_H

