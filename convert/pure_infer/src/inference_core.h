#ifndef INFERENCE_CORE_H
#define INFERENCE_CORE_H

#include "../include/config.h" // Includes acc_type_t, act_type_t, weight_type_t, LOG_*

// --- Operator Parameter Structures ---

// Parameters for a Convolutional Layer
typedef struct {
    // Input Tensor Shape
    int32_t H, W, C;
    // Output Tensor Shape
    int32_t OH, OW, OC;
    // Kernel Parameters
    int32_t K, S, P;         // Kernel size, Stride, Padding
    int32_t G;               // Groups (1 for standard/pointwise, C for depthwise)
    // Quantization Parameters
    act_type_t  in_zp;       // Input activation zero point (uint8)
    act_type_t  out_zp;      // Output activation zero point (uint8)
    // Flags
    int         is_per_channel; // Whether output requantization is per-channel
    int         relu;          // Apply ReLU activation (using out_zp as lower bound)
} ConvParams;

// Parameters for a Fully Connected (Linear) Layer
typedef struct {
    // Input/Output Features
    int32_t In;
    int32_t Out;
    // Quantization Parameters
    act_type_t  in_zp;       // Input activation zero point (uint8)
    act_type_t  out_zp;      // Output activation zero point (uint8)
    // Flags
    int         is_per_channel; // Whether output requantization is per-channel
    int         relu;          // Apply ReLU activation (using out_zp as lower bound)
} LinearParams;


// --- Core Model Graph Function ---

/**
 * @brief Executes the forward pass of the quantized model graph.
 *
 * This function takes quantized inputs (uint8) and computes the quantized
 * output (uint8) using integer arithmetic operations defined in this file.
 * It utilizes an arena buffer for intermediate activations.
 *
 * @param x_q Pointer to the quantized input tensor X (shape [1, 2, 3, 7]).
 * @param meta_q Pointer to the quantized input tensor Meta (shape [1, 2]).
 * IMPORTANT: Must be quantized using MODEL_GAP_OUT_SCALE/ZP.
 * @param out_q Pointer to the buffer where the quantized output (shape [1, 2]) will be stored.
 * @param arena Pointer to the working memory buffer (arena) for intermediate results.
 * Must be at least MODEL_ARENA_SIZE bytes.
 * @return 0 on success, non-zero on error (although current implementation always returns 0).
 */
int model_forward_s8(
    const act_type_t* x_q,
    const act_type_t* meta_q,
    act_type_t* out_q,
    uint8_t* arena
);

// --- Removed float helper function declarations ---
// void model_quantize_inputs(...)
// void model_dequantize_and_postprocess(...)


#endif // INFERENCE_CORE_H
