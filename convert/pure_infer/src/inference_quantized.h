#ifndef INFERENCE_QUANTIZED_H
#define INFERENCE_QUANTIZED_H

#include <stdint.h>     // For standard integer types (uint8_t, etc.)
#include <stddef.h>     // For size_t
#include "../include/config.h" // For act_type_t definition

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Initializes the inference engine.
 *
 * This function must be called once before any calls to
 * `inference_run_quantized`. It typically initializes internal resources,
 * such as clearing the memory arena.
 *
 * @return 0 on success, non-zero on failure.
 */
int inference_init(void);

/**
 * @brief Runs the core quantized inference graph.
 *
 * Takes quantized uint8 inputs and produces a quantized uint8 output.
 *
 * @param x_q Pointer to the quantized input tensor X (shape [1, 2, 3, 7]).
 * Must be quantized using `MODEL_INPUT_X_SCALE` and `MODEL_INPUT_X_ZERO_POINT`.
 * @param meta_q Pointer to the quantized input tensor Meta (shape [1, 2]).
 * IMPORTANT: Must be quantized using `MODEL_GAP_OUT_SCALE` and
 * `MODEL_GAP_OUT_ZERO_POINT` (which should match `MODEL_INPUT_META_SCALE/ZP`
 * after the GAP fix) to ensure compatibility with the `Cat` layer.
 * @param out_q Pointer to the buffer where the quantized output tensor (shape [1, 2])
 * will be stored.
 * @return 0 on success, non-zero on failure.
 */
int inference_run_quantized(
    const act_type_t* x_q,
    const act_type_t* meta_q,
    act_type_t* out_q
);

/**
 * @brief (Optional Helper) Dequantizes the uint8 output and applies post-processing.
 *
 * If you need the final float output (e.g., dx, dy values), you can use this
 * function after `inference_run_quantized`. It performs dequantization,
 * applies the Tanh activation, and scales the results according to the model's
 * `MODEL_FINAL_SCALE_DX` and `MODEL_FINAL_SCALE_DY`.
 *
 * @param out_q Pointer to the quantized output tensor (uint8, shape [1, 2])
 * obtained from `inference_run_quantized`.
 * @param final_out_f Pointer to the float buffer (shape [1, 2]) where the
 * final processed float output (dx, dy) will be stored.
 */
void inference_dequantize_postprocess_output(
    const act_type_t* out_q,
    float* final_out_f
);

#ifdef __cplusplus
}
#endif

#endif // INFERENCE_QUANTIZED_H
