#include "inference.h"
#include "inference_core.h" // Includes model_forward_s8, quantize, dequantize
#include "model_meta.h"   // For buffer sizes
#include <string.h>       // for memset
#include <stdlib.h>       // For malloc/free (only if not using static arena)

// Option 1: Static global arena (No malloc/free)
static uint8_t g_arena[MODEL_ARENA_SIZE];
#define USE_STATIC_ARENA 1

// Option 2: Dynamic arena (Requires malloc/free, less suitable for MCU)
// static uint8_t* g_arena_dynamic = NULL;

int inference_init(void) {
    LOG_INFO("Inference library initializing...");
#if USE_STATIC_ARENA
    memset(g_arena, 0, MODEL_ARENA_SIZE);
    LOG_INFO("Static arena initialized (%zu bytes).", (size_t)MODEL_ARENA_SIZE);
#else
    // if (g_arena_dynamic == NULL) {
    //     g_arena_dynamic = (uint8_t*)malloc(MODEL_ARENA_SIZE);
    //     if (g_arena_dynamic == NULL) {
    //         LOG_ERROR("Failed to allocate dynamic arena (%d bytes)!", MODEL_ARENA_SIZE);
    //         return -1;
    //     }
    //     memset(g_arena_dynamic, 0, MODEL_ARENA_SIZE);
    //     LOG_INFO("Dynamic arena allocated (%d bytes).", MODEL_ARENA_SIZE);
    // }
#endif
    return 0;
}

// Optional: Add a deinit function if using dynamic arena
// void inference_deinit(void) {
// #if !USE_STATIC_ARENA
//     if (g_arena_dynamic != NULL) {
//         free(g_arena_dynamic);
//         g_arena_dynamic = NULL;
//         LOG_INFO("Dynamic arena freed.");
//     }
// #endif
// }


size_t inference_get_arena_size(void) {
    return MODEL_ARENA_SIZE;
}

// --- 修正: Use act_type_t ---
int inference_run_quantized(
    const act_type_t* x_q,   // uint8
    const act_type_t* meta_q,// uint8
    act_type_t* out_q        // uint8
) {
#if USE_STATIC_ARENA
    return model_forward_s8(x_q, meta_q, out_q, g_arena);
#else
    // if (g_arena_dynamic == NULL) {
    //     LOG_ERROR("Inference library not initialized (arena missing).");
    //     return -1;
    // }
    // return model_forward_s8(x_q, meta_q, out_q, g_arena_dynamic);
    LOG_ERROR("Dynamic arena selected but not fully implemented/recommended.");
    return -1; // Or implement dynamic path fully
#endif
}

// --- 修正: Use act_type_t for intermediate buffers ---
int inference_run_float(
    const float* x_f,
    const float* meta_f,
    float* out_f
) {
    // 1. 分配临时量化 buffer (可在栈上，因为很小)
    // --- 修正: Use act_type_t ---
    act_type_t x_q[MODEL_INPUT_X_SHAPE_SIZE];
    act_type_t meta_q[MODEL_INPUT_META_SHAPE_SIZE];
    act_type_t out_q[MODEL_OUTPUT_SHAPE_SIZE];

    // 2. 浮点 -> 整型 (uint8)
    model_quantize_inputs(x_f, meta_f, x_q, meta_q);

    // 3. 核心整型推理
    int ret = inference_run_quantized(x_q, meta_q, out_q); // Calls model_forward_s8
    if (ret != 0) {
        LOG_ERROR("Core quantized inference failed (model_forward_s8 returned %d)", ret);
        return ret;
    }

    // 4. 整型 (uint8) -> 浮点 (带后处理)
    model_dequantize_and_postprocess(out_q, out_f);

    return 0; // Success
}

