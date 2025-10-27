#include "inference_quantized.h" // Public API header
#include "inference_core.h"    // Internal core graph execution (model_forward_s8)
#include "model_meta.h"      // Access to MODEL_ARENA_SIZE, scales, ZPs
#include <string.h>          // For memset
#include <math.h>            // For tanhf in postprocessing
#include "../include/config.h" // For LOG_* macros

// --- Arena Definition ---
// Using a static global arena is common for embedded systems to avoid dynamic allocation.
// Ensure MODEL_ARENA_SIZE in model_meta.h is large enough.
static uint8_t g_arena[MODEL_ARENA_SIZE];
#define USE_STATIC_ARENA 1


/**
 * @brief Initializes the inference engine (clears static arena).
 */
int inference_init(void) {
    LOG_INFO("Initializing quantized inference library...");
#if USE_STATIC_ARENA
    // Clear the static arena to ensure a clean state
    memset(g_arena, 0, sizeof(g_arena));
    LOG_INFO("Static arena cleared (%zu bytes).", (size_t)MODEL_ARENA_SIZE);
    return 0; // Success
#else
    // Dynamic allocation is not supported in this pure inference version
    LOG_ERROR("Dynamic arena allocation is not supported in this configuration.");
    return -1; // Indicate error
#endif
}


/**
 * @brief Runs the core quantized inference graph.
 */
int inference_run_quantized(
    const act_type_t* x_q,   // Quantized input X (uint8)
    const act_type_t* meta_q,// Quantized input Meta (uint8, using GAP scale/zp)
    act_type_t* out_q        // Output buffer for quantized result (uint8)
) {
#if USE_STATIC_ARENA
    // --- Optional Check (at INFO level) ---
    // Verify if the expected meta quantization parameters match the actual parameters used for GAP output.
    // This check helps catch potential inconsistencies during integration.
    #if (INFERENCE_C_LOG_LEVEL >= 3)
        if (MODEL_INPUT_META_ZERO_POINT != MODEL_GAP_OUT_ZERO_POINT || fabsf(MODEL_INPUT_META_SCALE - MODEL_GAP_OUT_SCALE) > 1e-9f) {
             LOG_INFO("Note: Meta input ZP/Scale (%u / %.4e) differs from GAP output ZP/Scale (%u / %.4e). "
                      "Ensure 'meta_q' was quantized using the GAP parameters for correct Cat operation.",
                      MODEL_INPUT_META_ZERO_POINT, MODEL_INPUT_META_SCALE,
                      MODEL_GAP_OUT_ZERO_POINT, MODEL_GAP_OUT_SCALE);
        }
    #endif

    // Call the core model execution function using the static arena
    int status = model_forward_s8(x_q, meta_q, out_q, g_arena);
    if (status != 0) {
        LOG_ERROR("Core model execution (model_forward_s8) failed with status %d.", status);
        return status; // Propagate error code
    }
    return 0; // Success
#else
    LOG_ERROR("Dynamic arena not supported in this configuration.");
    return -1; // Indicate error
#endif
}


// --- Helper Function: Dequantize uint8 to float32 ---
// Used only by the optional postprocessing function below.
static float dequantize_u8_to_f32(act_type_t val, float scale, act_type_t zero_point) {
    // Basic dequantization formula: (quantized_value - zero_point) * scale
    float result = ((float)val - (float)zero_point) * scale;

    // Optional: Log dequantization details if trace level is enabled
    LOG_TRACE("dequantize_u8_to_f32(val=%u, scale=%.4e, zp=%u) -> %.4e", val, scale, zero_point, result);

    // Optional: Check for NaN/Inf, which might indicate issues with scale/zp
    if (isnan(result) || isinf(result)) {
        LOG_WARN("Dequantization resulted in NaN or Inf!");
    }
    return result;
}

/**
 * @brief (Optional Helper) Dequantizes uint8 output and applies Tanh and scaling.
 */
void inference_dequantize_postprocess_output(
    const act_type_t* out_q, // Quantized output from inference_run_quantized
    float* final_out_f       // Buffer for final float results [dx, dy]
) {
    LOG_DEBUG("Dequantizing and post-processing model output...");

    // Buffer for intermediate float values after dequantization
    float dequantized_f[MODEL_OUTPUT_SHAPE_SIZE];

    // Step 1: Dequantize the two output values
    dequantized_f[0] = dequantize_u8_to_f32(out_q[0], MODEL_OUTPUT_SCALE, MODEL_OUTPUT_ZERO_POINT);
    dequantized_f[1] = dequantize_u8_to_f32(out_q[1], MODEL_OUTPUT_SCALE, MODEL_OUTPUT_ZERO_POINT);
    LOG_TRACE("  Dequantized output: [%.4e, %.4e]", dequantized_f[0], dequantized_f[1]);


    // Step 2: Apply Tanh activation
    float tanh_out_f[MODEL_OUTPUT_SHAPE_SIZE];
    tanh_out_f[0] = tanhf(dequantized_f[0]); // Use tanhf from math.h
    tanh_out_f[1] = tanhf(dequantized_f[1]);
    LOG_TRACE("  After Tanh: [%.4f, %.4f]", tanh_out_f[0], tanh_out_f[1]);


    // Step 3: Apply final scaling factors
    final_out_f[0] = tanh_out_f[0] * MODEL_FINAL_SCALE_DX;
    final_out_f[1] = tanh_out_f[1] * MODEL_FINAL_SCALE_DY;
    LOG_DEBUG("Final float output (dx, dy): [%f, %f]", final_out_f[0], final_out_f[1]);

}
