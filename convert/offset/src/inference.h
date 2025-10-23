#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stddef.h> // for size_t
#include "../include/config.h" // For act_type_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 初始化模型 (e.g., zero arena)
 */
int inference_init(void);

// Optional deinit if using dynamic memory
// void inference_deinit(void);

/**
 * @brief 运行核心量化推理 (纯整型)
 * --- 修正: Use act_type_t ---
 *
 * @param x_q       量化后的输入 X [1, 2, 3, 7] (uint8)
 * @param meta_q    量化后的输入 Meta [1, 2] (uint8)
 * @param out_q     量化后的输出 [1, 2] (uint8)
 * @return 0 成功, -1 失败
 */
int inference_run_quantized(
    const act_type_t* x_q,
    const act_type_t* meta_q,
    act_type_t* out_q
);

/**
 * @brief (验证用) 运行浮点封装的推理
 * (内部处理 量化 -> 整型推理 -> 反量化 -> 后处理)
 * --- 修正: Input args are const ---
 *
 * @param x_f       浮点输入 X [1, 2, 3, 7]
 * @param meta_f    浮点输入 Meta [1, 2]
 * @param out_f     最终浮点输出 [1, 2] (已应用 Tanh 和 scaling)
 * @return 0 成功, -1 失败
 */
int inference_run_float(
    const float* x_f,
    const float* meta_f,
    float* out_f
);

/**
 * @brief 获取所需的工作区 (arena) 大小
 */
size_t inference_get_arena_size(void);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_H

