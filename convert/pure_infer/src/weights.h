#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stdint.h> // For int8_t, int32_t types

// These extern declarations make the weight and parameter arrays
// defined in weights.c globally accessible to other C files that
// include this header.

// --- Block 1 Depthwise Convolution ---
extern const int8_t g_block1_dw_weight[18];        // Weights (OC, KH, KW, IC/G = 2, 3, 3, 1)
extern const int32_t g_block1_dw_bias[2];          // Original bias (int32)
extern const int32_t g_block1_dw_multiplier[2];    // Output requantization multiplier (per channel)
extern const int32_t g_block1_dw_shift[2];         // Output requantization shift (per channel)

// --- Block 1 Pointwise Convolution (Fused ReLU) ---
extern const int8_t g_block1_pw_weight[8];         // Weights (OC, KH, KW, IC = 4, 1, 1, 2)
extern const int32_t g_block1_pw_bias[4];          // Original bias (int32)
extern const int32_t g_block1_pw_multiplier[4];    // Output requantization multiplier (per channel)
extern const int32_t g_block1_pw_shift[4];         // Output requantization shift (per channel)

// --- Block 2 Depthwise Convolution ---
extern const int8_t g_block2_dw_weight[36];        // Weights (OC, KH, KW, IC/G = 4, 3, 3, 1)
extern const int32_t g_block2_dw_bias[4];          // Original bias (int32)
extern const int32_t g_block2_dw_multiplier[4];    // Output requantization multiplier (per channel)
extern const int32_t g_block2_dw_shift[4];         // Output requantization shift (per channel)

// --- Block 2 Pointwise Convolution (Fused ReLU) ---
extern const int8_t g_block2_pw_weight[16];        // Weights (OC, KH, KW, IC = 4, 1, 1, 4)
extern const int32_t g_block2_pw_bias[4];          // Original bias (int32)
extern const int32_t g_block2_pw_multiplier[4];    // Output requantization multiplier (per channel)
extern const int32_t g_block2_pw_shift[4];         // Output requantization shift (per channel)

// --- Head Linear Layer 0 (Fused ReLU) ---
extern const int8_t g_head_0_weight[48];           // Weights (Out, In = 8, 6)
extern const int32_t g_head_0_bias[8];             // Original bias (int32)
extern const int32_t g_head_0_multiplier[8];       // Output requantization multiplier (per channel)
extern const int32_t g_head_0_shift[8];            // Output requantization shift (per channel)

// --- Head Linear Layer 2 ---
extern const int8_t g_head_2_weight[16];           // Weights (Out, In = 2, 8)
extern const int32_t g_head_2_bias[2];             // Original bias (int32)
extern const int32_t g_head_2_multiplier[2];       // Output requantization multiplier (per channel)
extern const int32_t g_head_2_shift[2];            // Output requantization shift (per channel)

// --- GAP Requantization Parameters ---
// NOTE: These are exported but NOT used by the current GAP implementation,
// which uses scales directly to simulate the Python behavior.
extern const int32_t g_gap_out_multiplier[1];      // Output requantization multiplier (per tensor)
extern const int32_t g_gap_out_shift[1];           // Output requantization shift (per tensor)

#endif // WEIGHTS_H
