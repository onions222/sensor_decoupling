#include "inference_core.h"
#include "weights.h"      // Auto-generated weights and requantization parameters
#include "model_meta.h"   // Auto-generated model configuration (shapes, scales, ZPs)
#include <string.h>       // For memcpy
#include <math.h>         // For floor (in quantize helper), isnan, isinf
#include <limits.h>       // For INT32_MAX, INT32_MIN, UINT8_MAX
#include <inttypes.h>     // For PRId32, PRIu8 format specifiers in logs
#include <stdbool.h>      // For bool type

// --- Removed DUMP_INTERMEDIATES related file I/O ---

// =================================================================
// 1. Quantized Math Helpers (TFLite Micro Style, 32-bit focus)
// =================================================================

/**
 * @brief Performs a rounding right shift on a 64-bit value.
 *
 * Implements round-half-up rounding. Handles saturation to int32 range.
 * This is a key component for requantization.
 *
 * @param val The 64-bit integer value to shift.
 * @param shift The number of bits to shift right (must be non-negative).
 * @return The 32-bit saturated result after shifting and rounding.
 */
static inline int32_t ShiftRightRounded_S64(int64_t val, int32_t shift) {
    // Handle non-positive shift (left shift or no shift) - less common in requant
    if (shift <= 0) {
        int32_t left_shift = -shift;
        const int64_t max_val = INT32_MAX; const int64_t min_val = INT32_MIN;
        // Check for potential overflow before shifting left
        if (left_shift >= 63 || (val != 0 && (val > (max_val >> left_shift) || val < (min_val >> left_shift)))) {
             return (val > 0) ? INT32_MAX : INT32_MIN;
        }
        int64_t result_64 = val << left_shift;
        // Saturate after left shift
        if (result_64 > max_val) { return INT32_MAX; }
        if (result_64 < min_val) { return INT32_MIN; }
        return (int32_t)result_64;
    }
    // Handle large right shifts
    if (shift >= 63) { return (val >= 0) ? 0 : -1; } // Effectively divides by a huge number

    const int64_t max_val = INT32_MAX; const int64_t min_val = INT32_MIN;
    // Use unsigned absolute value for rounding calculation
    uint64_t abs_val = (val < 0) ? (uint64_t)(-val) : (uint64_t)val;
    uint64_t divisor = (uint64_t)1 << shift;
    // Calculate quotient and remainder using bitwise ops
    uint64_t quotient = abs_val >> shift;
    uint64_t remainder = abs_val & (divisor - 1);
    uint64_t halfway = divisor >> 1; // Rounding threshold (0.5 in fixed point)

    // Round half up: if remainder >= halfway, round up
    bool round_up = (remainder >= halfway);

    if (round_up) {
        // Check for overflow before incrementing quotient
        if (quotient == UINT64_MAX) {
            // This case is extremely unlikely given the input is int64_t
            // If it happens, saturate based on the original sign
             return (val > 0) ? INT32_MAX : INT32_MIN;
        }
        quotient++;
    }

    // Apply original sign and saturate to int32 range
    int64_t result_64 = (val < 0) ? -(int64_t)quotient : (int64_t)quotient;
    if (result_64 > max_val) { return INT32_MAX; }
    if (result_64 < min_val) { return INT32_MIN; }
    return (int32_t)result_64;
}


/**
 * @brief Multiplies an accumulator value by a quantized multiplier and performs shifting.
 *
 * This function simulates the core requantization scaling step using 64-bit
 * intermediate products and the ShiftRightRounded_S64 helper.
 *
 * @param x The 32-bit accumulator value.
 * @param quantized_multiplier The 32-bit quantized multiplier.
 * @param shift The right shift amount (non-negative).
 * @return The 32-bit scaled and rounded result.
 */
static int32_t MultiplyByQuantizedMultiplier(acc_type_t x, int32_t quantized_multiplier, int32_t shift) {
    // Calculate the 64-bit product (potential overflow is handled by 64-bit type)
    int64_t product = (int64_t)x * quantized_multiplier;
    LOG_TRACE("  MulByQuantMult: x=%" PRId32 ", mult=%" PRId32 ", shift=%d -> product=%" PRId64, x, quantized_multiplier, shift, product);

    // Apply rounding right shift
    int32_t result = ShiftRightRounded_S64(product, shift);
    LOG_TRACE("  MulByQuantMult: -> shifted_rounded=%" PRId32, result);
    return result;
}

/**
 * @brief Requantizes a 32-bit accumulator value to an 8-bit unsigned integer.
 *
 * Applies the scaling (multiplier + shift), adds the output zero point,
 * and saturates the result to the uint8 range [0, 255].
 *
 * @param accum The 32-bit signed accumulator value.
 * @param multiplier The 32-bit quantized multiplier for output scaling.
 * @param shift The right shift amount for output scaling.
 * @param output_zp The zero point of the output quantization (uint8).
 * @return The final requantized value as uint8.
 */
static act_type_t requantize_s32_to_u8( acc_type_t accum, int32_t multiplier, int32_t shift, act_type_t output_zp) {
    LOG_TRACE("Requantize: accum=%" PRId32 ", mult=%" PRId32 ", shift=%d, out_zp=%u", accum, multiplier, shift, output_zp);

    // Step 1: Apply the scaling (multiplier and shift)
    int32_t scaled_acc = MultiplyByQuantizedMultiplier(accum, multiplier, shift);
    LOG_TRACE("  Requantize: scaled_acc = %" PRId32, scaled_acc);

    // Step 2: Add the output zero point (using saturation)
    acc_type_t shifted_acc_with_zp = saturate_add_s32(scaled_acc, (acc_type_t)output_zp);
    LOG_TRACE("  Requantize: shifted_acc (with zp) = %" PRId32, shifted_acc_with_zp);

    // Step 3: Saturate the result to the uint8 range [0, 255]
    act_type_t final_val = saturate_s32_to_u8(shifted_acc_with_zp);
    LOG_TRACE("  Requantize: final_val (uint8) = %u", final_val);
    return final_val;
}

// =================================================================
// 2. Quantized Operators Implementation
// =================================================================

// Helper to calculate flat index for NCHW layout
static inline int32_t get_index_nchw(int32_t h, int32_t w, int32_t c, int32_t H, int32_t W) {
    // Assumes N=1
    return c * H * W + h * W + w;
}

// Helper to calculate flat index for Convolution weights (OHWI layout assumed for export)
// For Depthwise, IC_per_G = 1, G = OC. For Pointwise/Standard, G=1, IC_per_G = IC
static inline int32_t get_weight_index_conv(int32_t oc, int32_t kh, int32_t kw, int32_t ic_g, // ic_g is index within the group
                                         int32_t K, int32_t IC_per_G) {
    // Weight layout: [OC][KH][KW][IC_per_G]
    return oc * (K * K * IC_per_G) +   // Offset for output channel
           kh * (K * IC_per_G) +       // Offset for kernel height
           kw * (IC_per_G) +           // Offset for kernel width
           ic_g;                       // Offset for input channel within group
}

/**
 * @brief Performs quantized 2D convolution (uint8 input, int8 weight, int32 acc, uint8 output).
 * Supports standard, depthwise, and grouped convolutions. Handles ReLU activation.
 */
static void layer_conv2d_s8(
    act_type_t* out,                // Output buffer (uint8)
    const act_type_t* in,           // Input buffer (uint8)
    const weight_type_t* weight,    // Weight buffer (int8)
    const acc_type_t* bias,         // Bias buffer (int32, original unquantized bias scaled)
    const ConvParams* p,            // Layer parameters
    const int32_t* out_multiplier,  // Output requant multiplier(s)
    const int32_t* out_shift        // Output requant shift(s)
) {
    const int32_t IC_per_G = p->C / p->G; // Input channels per group
    // For ReLU activation, the lower bound is the output zero point
    const act_type_t relu_limit = p->out_zp;
    bool is_block1_pw_trace = (p->C == 2 && p->K == 1 && p->G == 1); // Specific flag for detailed trace example

    LOG_DEBUG("Conv2D: In(%ld,%ld,%ld) Out(%ld,%ld,%ld) K=%ld S=%ld P=%ld G=%ld PerChan=%d InZp=%u OutZp=%u ReLU=%d",
        (long)p->H, (long)p->W, (long)p->C, (long)p->OH, (long)p->OW, (long)p->OC,
        (long)p->K, (long)p->S, (long)p->P, (long)p->G, p->is_per_channel, p->in_zp, p->out_zp, p->relu);

    // Loop over output spatial dimensions (height, width)
    for (int32_t oh = 0; oh < p->OH; ++oh) {
        for (int32_t ow = 0; ow < p->OW; ++ow) {
            // Loop over output channels
            for (int32_t oc = 0; oc < p->OC; ++oc) {
                acc_type_t acc = 0; // Initialize accumulator for this output pixel/channel

                // Determine the group index for this output channel (relevant for grouped/depthwise conv)
                const int32_t g_idx = (p->G > 1) ? (oc / (p->OC / p->G)) : 0;

                // Loop over kernel spatial dimensions (height, width)
                for (int32_t kh = 0; kh < p->K; ++kh) {
                    for (int32_t kw = 0; kw < p->K; ++kw) {
                        // Loop over input channels within the group
                        for (int32_t ic_g = 0; ic_g < IC_per_G; ++ic_g) {
                            // Calculate corresponding input coordinates (h, w)
                            const int32_t ih = oh * p->S + kh - p->P;
                            const int32_t iw = ow * p->S + kw - p->P;

                            // Check for padding: skip if input coordinates are out of bounds
                            if (ih < 0 || ih >= p->H || iw < 0 || iw >= p->W) {
                                continue; // Effectively multiply by zero due to padding
                            }

                            // Calculate absolute input channel index
                            const int32_t ic_abs = g_idx * IC_per_G + ic_g;

                            // Get flat indices for input activation and weight
                            int32_t in_idx = get_index_nchw(ih, iw, ic_abs, p->H, p->W);
                            int32_t w_idx = get_weight_index_conv(oc, kh, kw, ic_g, p->K, IC_per_G);

                            // Perform the core MAC operation with zero point adjustments
                            acc_type_t in_val_u8 = in[in_idx];
                            acc_type_t in_zp_val = (acc_type_t)p->in_zp; // Cast ZP to accumulator type
                            acc_type_t weight_val_s8 = (acc_type_t)weight[w_idx]; // Cast weight

                            // Calculate term: (input - input_zp) * weight
                            acc_type_t term = (in_val_u8 - in_zp_val) * weight_val_s8;

                            // Accumulate using saturating addition
                            acc = saturate_add_s32(acc, term);

                            // Example detailed trace for one specific output element
                            bool trace_this_pixel = is_block1_pw_trace && (oh==0 && ow==0 && oc==0);
                            if(trace_this_pixel && ic_g < 2) { // Limit trace lines
                                LOG_TRACE("  MAC (oc=%ld, kh=%ld, kw=%ld, ic_g=%ld): in[%ld]=%u, w[%ld]=%d => term=%ld -> acc=%ld",
                                          (long)oc, (long)kh, (long)kw, (long)ic_g, (long)in_idx, in_val_u8,
                                          (long)w_idx, (int)weight_val_s8, (long)term, (long)acc);
                            }
                        } // End input channel loop
                    } // End kernel width loop
                } // End kernel height loop

                // Add bias (scaled original bias) after MAC loops, using saturation
                if (bias != NULL) {
                     acc = saturate_add_s32(acc, bias[oc]);
                }
                bool trace_this_pixel = is_block1_pw_trace && (oh==0 && ow==0 && oc==0);
                if(trace_this_pixel) LOG_TRACE("  Final Acc + Bias (oc=%ld) before Requant: %ld", (long)oc, (long)acc);

                // Get the correct requantization multiplier and shift for this output channel
                int32_t current_multiplier;
                int32_t current_shift;
                if (p->is_per_channel) {
                    current_multiplier = out_multiplier[oc];
                    current_shift = out_shift[oc];
                } else {
                    current_multiplier = out_multiplier[0]; // Per-tensor uses the first value
                    current_shift = out_shift[0];
                }

                 if(trace_this_pixel) LOG_TRACE("  Requant Params (oc=%ld): mult=%ld, shift=%ld, out_zp=%u", (long)oc, (long)current_multiplier, (long)current_shift, p->out_zp);

                // Requantize the accumulated value to uint8
                act_type_t out_val = requantize_s32_to_u8(acc, current_multiplier, current_shift, p->out_zp);
                 if(trace_this_pixel) LOG_TRACE("  Requant Result (pre-ReLU): %u", out_val);


                // Apply ReLU activation if enabled
                if (p->relu) {
                    // ReLU lower bound is the output zero point
                    out_val = (out_val < relu_limit) ? relu_limit : out_val;
                    if(trace_this_pixel) LOG_TRACE("  After ReLU (limit=%u): %u", relu_limit, out_val);
                }

                // Store the final uint8 output value
                out[get_index_nchw(oh, ow, oc, p->OH, p->OW)] = out_val;
            } // End output channel loop
        } // End output width loop
    } // End output height loop
}


// --- Forward declarations needed for GAP's dequant/quant helpers ---
static double dequantize_u8_to_f64(act_type_t val, double scale, act_type_t zero_point);
static act_type_t quantize_f64_to_u8(double val, double scale, act_type_t zero_point);


/**
 * @brief Performs quantized Global Average Pooling.
 *
 * This implementation simulates the behavior observed in the exported PyTorch FX graph:
 * 1. Dequantize each input element (uint8) to double-precision float.
 * 2. Calculate the average of these float values (in double precision) over the spatial dimensions (H, W).
 * 3. Requantize the resulting float average back to uint8 using the target output scale and zero point.
 * This approach aims to match Python's numerical precision more closely, especially for GAP.
 *
 * @param out Pointer to the output buffer (uint8, shape [C]).
 * @param in Pointer to the input buffer (uint8, shape [H, W, C]).
 * @param H Input height.
 * @param W Input width.
 * @param C Number of channels.
 * @param in_zp Input zero point (uint8).
 * @param out_zp Target output zero point (uint8).
 * @param out_multiplier (Not used in this implementation).
 * @param out_shift (Not used in this implementation).
 * @param is_per_channel (Not used, assumed per-tensor based on model export).
 */
static void layer_global_avg_pool_s8(
    act_type_t* out,
    const act_type_t* in,
    int32_t H, int32_t W, int32_t C,
    act_type_t in_zp,
    act_type_t out_zp,
    const int32_t* out_multiplier, // Ignored
    const int32_t* out_shift,      // Ignored
    int is_per_channel             // Ignored
) {
    (void)out_multiplier; // Mark as unused
    (void)out_shift;      // Mark as unused
    (void)is_per_channel; // Mark as unused

    LOG_DEBUG("GAP (Simulating PyTorch Dequant->Avg->Requant): In(%ld,%ld,%ld) InZp=%u OutZp=%u",
              (long)H, (long)W, (long)C, in_zp, out_zp);

    const int32_t pool_size = H * W; // Number of elements to average per channel
    if (pool_size <= 0) {
        LOG_ERROR("GAP pool size is zero or negative!");
        // Set output to zero point if pool size is invalid
        for(int32_t c=0; c<C; ++c) out[c] = out_zp;
        return;
    }

    // --- Get Input Scale (from Block2 output) and Target Output Scale (for Cat input) ---
    // These scales are required for the dequantization and requantization steps.
    const double in_scale_f64 = (double)MODEL_BLOCK2_OUT_SCALE;
    const double out_scale_f64 = (double)MODEL_GAP_OUT_SCALE; // Target scale = Cat input scale

    if (out_scale_f64 <= 0.0) {
        LOG_ERROR("GAP output scale (MODEL_GAP_OUT_SCALE) is non-positive!");
        for(int32_t c=0; c<C; ++c) out[c] = out_zp;
        return;
    }
    if (in_scale_f64 <= 0.0) {
         LOG_WARN("GAP input scale (MODEL_BLOCK2_OUT_SCALE) is non-positive! Dequantization will be inaccurate.");
         // Continue, but results will likely be wrong.
    }


    // Loop over each channel
    for (int32_t c = 0; c < C; ++c) {
        // --- Step 1: Accumulate sum in double precision ---
        double float_sum = 0.0;
        for (int32_t h = 0; h < H; ++h) {
            for (int32_t w = 0; w < W; ++w) {
                // Get the quantized input value
                act_type_t q_val = in[get_index_nchw(h, w, c, H, W)];
                // Dequantize to double
                double dq_val = dequantize_u8_to_f64(q_val, in_scale_f64, in_zp);
                // Accumulate
                float_sum += dq_val;
            }
        }

        // --- Step 2: Calculate float average (in double precision) ---
        double float_avg = float_sum / (double)pool_size;
        LOG_TRACE("  GAP Ch %ld: float_sum=%.6e, float_avg=%.6e", (long)c, float_sum, float_avg);

        // --- Step 3: Requantize the float average back to uint8 ---
        // Use the target output scale and zero point (from MODEL_GAP_OUT_SCALE/ZP)
        act_type_t out_val = quantize_f64_to_u8(float_avg, out_scale_f64, out_zp);
        LOG_TRACE("  GAP Ch %ld: requantized_avg (uint8)=%u (using scale=%.4e, zp=%u)",
                  (long)c, out_val, out_scale_f64, out_zp);

        // Store the result
        out[c] = out_val;
    }
}


// Helper to calculate flat index for Linear layer weights (Out, In layout)
static inline int32_t get_weight_index_linear(int32_t out_c, int32_t in_c, int32_t In) {
    return out_c * In + in_c;
}

/**
 * @brief Performs quantized Linear (fully connected) layer operation.
 * (uint8 input, int8 weight, int32 acc, uint8 output). Handles ReLU activation.
 */
static void layer_linear_s8(
    act_type_t* out,                // Output buffer (uint8)
    const act_type_t* in,           // Input buffer (uint8, shape [In])
    const weight_type_t* weight,    // Weight buffer (int8, shape [Out, In])
    const acc_type_t* bias,         // Bias buffer (int32)
    const LinearParams* p,          // Layer parameters
    const int32_t* out_multiplier,  // Output requant multiplier(s)
    const int32_t* out_shift        // Output requant shift(s)
) {
    LOG_DEBUG("Linear: In(%ld) Out(%ld) PerChan=%d InZp=%u OutZp=%u ReLU=%d",
        (long)p->In, (long)p->Out, p->is_per_channel, p->in_zp, p->out_zp, p->relu);
    // For ReLU activation, the lower bound is the output zero point
    const act_type_t relu_limit = p->out_zp;

    // Loop over output features
    for (int32_t out_c = 0; out_c < p->Out; ++out_c) {
        acc_type_t acc = 0; // Initialize accumulator for this output feature

        // Loop over input features (MAC operation)
        for (int32_t in_c = 0; in_c < p->In; ++in_c) {
            // Get flat index for weight matrix
            int32_t w_idx = get_weight_index_linear(out_c, in_c, p->In);

            // Perform MAC with zero point adjustments
            acc_type_t in_val_u8 = in[in_c];
            acc_type_t in_zp_val = (acc_type_t)p->in_zp;
            acc_type_t weight_val_s8 = (acc_type_t)weight[w_idx];
            acc_type_t term = (in_val_u8 - in_zp_val) * weight_val_s8;

            // Accumulate using saturating addition
            acc = saturate_add_s32(acc, term);
        } // End input feature loop

        // Add bias (scaled original bias) after MAC loops, using saturation
        if (bias != NULL) {
            acc = saturate_add_s32(acc, bias[out_c]);
        }
        LOG_TRACE("  Linear Out %ld: Acc + Bias = %ld", (long)out_c, (long)acc);

        // Get the correct requantization multiplier and shift for this output channel
        int32_t current_multiplier;
        int32_t current_shift;
        if (p->is_per_channel) {
            current_multiplier = out_multiplier[out_c];
            current_shift = out_shift[out_c];
        } else {
            current_multiplier = out_multiplier[0]; // Per-tensor uses the first value
            current_shift = out_shift[0];
        }
        LOG_TRACE("  Linear Out %ld: Requant Params: mult=%ld, shift=%ld, out_zp=%u", (long)out_c, (long)current_multiplier, (long)current_shift, p->out_zp);


        // Requantize the accumulated value to uint8
        act_type_t out_val = requantize_s32_to_u8(acc, current_multiplier, current_shift, p->out_zp);
        LOG_TRACE("  Linear Out %ld: Requant Result (pre-ReLU): %u", (long)out_c, out_val);


        // Apply ReLU activation if enabled
        if (p->relu) {
            out_val = (out_val < relu_limit) ? relu_limit : out_val;
             LOG_TRACE("  Linear Out %ld: After ReLU (limit=%u): %u", (long)out_c, relu_limit, out_val);
        }

        // Store the final uint8 output value
        out[out_c] = out_val;
    } // End output feature loop
}


/**
 * @brief Concatenates two uint8 tensors along the channel dimension (axis=1).
 * Assumes inputs have the same scale and zero point (verified during export/GAP changes).
 */
static void layer_cat_s8(
    act_type_t* out,          // Output buffer (uint8, shape [C1 + C2])
    const act_type_t* in1,    // First input buffer (uint8, shape [C1])
    int32_t C1,               // Number of channels in first input
    const act_type_t* in2,    // Second input buffer (uint8, shape [C2])
    int32_t C2                // Number of channels in second input
) {
    LOG_DEBUG("Cat: In1(%ld channels), In2(%ld channels) -> Out(%ld channels)", (long)C1, (long)C2, (long)(C1+C2));

    // Simple concatenation using memcpy, as qparams are assumed to match
    memcpy(out,      in1, C1 * sizeof(act_type_t)); // Copy first input
    memcpy(out + C1, in2, C2 * sizeof(act_type_t)); // Copy second input after first
}


// =================================================================
// 3. Model Graph Execution
// =================================================================

// Define layer parameters using constants from model_meta.h
// Convolution Layers
static const ConvParams P_BLOCK1_DW = {
    .H = 3, .W = 7, .C = 2, .OH = 3, .OW = 7, .OC = 2, // Shapes
    .K = 3, .S = 1, .P = 1, .G = 2, // Kernel, Stride, Padding, Groups (Depthwise G=C)
    .in_zp = MODEL_INPUT_X_ZERO_POINT, .out_zp = MODEL_BLOCK1_DW_OUT_ZERO_POINT, // Zero Points
    .relu = 0, .is_per_channel = MODEL_BLOCK1_DW_IS_PER_CHANNEL // Flags
};
static const ConvParams P_BLOCK1_PW = {
    .H = 3, .W = 7, .C = 2, .OH = 3, .OW = 7, .OC = 4, // Shapes
    .K = 1, .S = 1, .P = 0, .G = 1, // Kernel, Stride, Padding, Groups (Pointwise G=1)
    .in_zp = MODEL_BLOCK1_DW_OUT_ZERO_POINT, .out_zp = MODEL_BLOCK1_OUT_ZERO_POINT,
    .relu = 1, .is_per_channel = MODEL_BLOCK1_PW_IS_PER_CHANNEL
};
static const ConvParams P_BLOCK2_DW = {
    .H = 3, .W = 7, .C = 4, .OH = 3, .OW = 7, .OC = 4,
    .K = 3, .S = 1, .P = 1, .G = 4,
    .in_zp = MODEL_BLOCK1_OUT_ZERO_POINT, .out_zp = MODEL_BLOCK2_DW_OUT_ZERO_POINT,
    .relu = 0, .is_per_channel = MODEL_BLOCK2_DW_IS_PER_CHANNEL
};
static const ConvParams P_BLOCK2_PW = {
    .H = 3, .W = 7, .C = 4, .OH = 3, .OW = 7, .OC = 4,
    .K = 1, .S = 1, .P = 0, .G = 1,
    .in_zp = MODEL_BLOCK2_DW_OUT_ZERO_POINT, .out_zp = MODEL_BLOCK2_OUT_ZERO_POINT, // Note: output ZP was wrong in export, using DW ZP
    .relu = 1, .is_per_channel = MODEL_BLOCK2_PW_IS_PER_CHANNEL
};

// Linear Layers
static const LinearParams P_HEAD_0 = {
    .In = 6, .Out = 8, // Shapes (Input = GAP channels + Meta channels = 4 + 2)
    .in_zp = MODEL_CAT_OUT_ZERO_POINT, .out_zp = MODEL_HEAD0_OUT_ZERO_POINT,
    .relu = 1, .is_per_channel = MODEL_HEAD_0_IS_PER_CHANNEL
};
static const LinearParams P_HEAD_2 = {
    .In = 8, .Out = 2, // Shapes
    .in_zp = MODEL_HEAD0_OUT_ZERO_POINT, .out_zp = MODEL_OUTPUT_ZERO_POINT,
    .relu = 0, .is_per_channel = MODEL_HEAD_2_IS_PER_CHANNEL
};

/**
 * @brief Executes the forward pass of the quantized model graph.
 * (See header file for detailed description)
 */
int model_forward_s8(
    const act_type_t* x_q,      // Quantized input X (uint8)
    const act_type_t* meta_q,   // Quantized input Meta (uint8, using GAP/Cat scale/zp)
    act_type_t* out_q,          // Output buffer for quantized result (uint8)
    uint8_t* arena              // Working memory buffer
) {
    LOG_DEBUG("Starting Quantized Model Forward Pass...");

    // --- Arena Buffer Allocation ---
    // Define pointers into the arena for intermediate activation tensors.
    // Sizes are calculated based on layer output shapes.
    // Offset calculations ensure non-overlapping buffers.
    const size_t size_block1_dw_out = 1 * 2 * 3 * 7; // 42
    const size_t size_block1_pw_out = 1 * 4 * 3 * 7; // 84
    const size_t size_block2_dw_out = 1 * 4 * 3 * 7; // 84
    const size_t size_block2_pw_out = 1 * 4 * 3 * 7; // 84
    const size_t size_gap_out       = 1 * 4;         // 4
    const size_t size_cat_out       = 1 * (4 + 2);   // 6 (GAP channels + Meta channels)
    const size_t size_head0_out     = 1 * 8;         // 8
    // Final output (size 2) goes directly to out_q, not arena.

    size_t offset = 0;
    act_type_t* buf_block1_dw_out = (act_type_t*)(arena + offset); offset += size_block1_dw_out;
    act_type_t* buf_block1_pw_out = (act_type_t*)(arena + offset); offset += size_block1_pw_out;
    act_type_t* buf_block2_dw_out = (act_type_t*)(arena + offset); offset += size_block2_dw_out;
    act_type_t* buf_block2_pw_out = (act_type_t*)(arena + offset); offset += size_block2_pw_out;
    act_type_t* buf_gap_out       = (act_type_t*)(arena + offset); offset += size_gap_out;
    act_type_t* buf_cat_out       = (act_type_t*)(arena + offset); offset += size_cat_out;
    act_type_t* buf_head0_out     = (act_type_t*)(arena + offset); offset += size_head0_out;

    const size_t required_arena = offset; // Total size used

    // Compile-time check for sufficient arena size
    #if MODEL_ARENA_SIZE < required_arena
        #error "MODEL_ARENA_SIZE in model_meta.h is too small for activation buffers!"
         // If this triggers, increase MODEL_ARENA_SIZE to at least required_arena (312).
    #endif
     LOG_DEBUG("Arena allocation: required=%zu, available=%d", required_arena, MODEL_ARENA_SIZE);


    // --- Execute Layer by Layer ---

    // --- Block 1 ---
    layer_conv2d_s8(buf_block1_dw_out, x_q, g_block1_dw_weight, g_block1_dw_bias, &P_BLOCK1_DW, g_block1_dw_multiplier, g_block1_dw_shift);
    layer_conv2d_s8(buf_block1_pw_out, buf_block1_dw_out, g_block1_pw_weight, g_block1_pw_bias, &P_BLOCK1_PW, g_block1_pw_multiplier, g_block1_pw_shift);

    // --- Block 2 ---
    layer_conv2d_s8(buf_block2_dw_out, buf_block1_pw_out, g_block2_dw_weight, g_block2_dw_bias, &P_BLOCK2_DW, g_block2_dw_multiplier, g_block2_dw_shift);
    layer_conv2d_s8(buf_block2_pw_out, buf_block2_dw_out, g_block2_pw_weight, g_block2_pw_bias, &P_BLOCK2_PW, g_block2_pw_multiplier, g_block2_pw_shift);

    // --- Global Average Pooling ---
    // Uses the modified implementation simulating Python's dequant->avg->requant.
    layer_global_avg_pool_s8(
        buf_gap_out,                // Output buffer
        buf_block2_pw_out,          // Input buffer
        3, 7, 4,                    // Input H, W, C
        MODEL_BLOCK2_OUT_ZERO_POINT,// Input ZP
        MODEL_GAP_OUT_ZERO_POINT,   // Target Output ZP (must match Meta ZP for Cat)
        NULL,                       // Multiplier (ignored)
        NULL,                       // Shift (ignored)
        0                           // is_per_channel (ignored)
    );

    // --- Concatenation ---
    // Concatenates GAP output (buf_gap_out, 4 channels) and Meta input (meta_q, 2 channels)
    layer_cat_s8(buf_cat_out, buf_gap_out, 4, meta_q, 2);

    // --- Head (Linear Layers) ---
    layer_linear_s8(buf_head0_out, buf_cat_out, g_head_0_weight, g_head_0_bias, &P_HEAD_0, g_head_0_multiplier, g_head_0_shift);
    layer_linear_s8(out_q, buf_head0_out, g_head_2_weight, g_head_2_bias, &P_HEAD_2, g_head_2_multiplier, g_head_2_shift);

    LOG_DEBUG("Quantized Model Forward Pass Completed.");
    return 0; // Success
}


// =================================================================
// 4. Quantization/Dequantization Helpers (used only by GAP)
// =================================================================

/**
 * @brief Quantizes a double-precision float value to uint8.
 * Uses round-half-up (floor(val + 0.5)) and clamps to [0, 255].
 * Added specifically for the GAP layer's requantization step.
 */
static act_type_t quantize_f64_to_u8(double val, double scale, act_type_t zero_point) {
    LOG_TRACE("quantize_f64_to_u8(val=%.6e, scale=%.6e, zp=%u)", val, scale, zero_point);
    // Check for invalid scale
    if (scale <= 0.0 || isnan(scale) || isinf(scale)) {
        LOG_WARN("Quantization scale is non-positive, NaN or Inf (%.4e)! Returning zero_point.", scale);
        return zero_point;
    }

    // Calculate scaled value
    double scaled_val = val / scale;
    // Check for intermediate NaN/Inf
    if (isnan(scaled_val) || isinf(scaled_val)){
        LOG_WARN("Intermediate division result is NaN or Inf (val=%.6e / scale=%.6e). Clamping.", val, scale);
        // Clamp based on sign of original value (matches common library behavior)
        return (val >= 0.0) ? UINT8_MAX : 0;
    }

    // Add zero point
    double shifted_val = scaled_val + (double)zero_point;
    LOG_TRACE("  Intermediate shifted_val = %.6f", shifted_val);

    // Apply rounding (round-half-up: floor(x + 0.5))
    double rounded_val = floor(shifted_val + 0.5);
    LOG_TRACE("  Rounded (floor(x+0.5)) = %.1f", rounded_val);

    // Clamp to uint8 range [0, 255]
    if (rounded_val > 255.0) rounded_val = 255.0;
    if (rounded_val < 0.0) rounded_val = 0.0;

    act_type_t result = (act_type_t)rounded_val;
    LOG_TRACE("  Clamped uint8 result = %u", result);
    return result;
}

/**
 * @brief Dequantizes a uint8 value to double-precision float.
 * Added specifically for the GAP layer's dequantization step.
 */
static double dequantize_u8_to_f64(act_type_t val, double scale, act_type_t zero_point) {
    // Note: Logging disabled for f64 due to lower relevance for tracing integer ops
    // LOG_TRACE("dequantize_u8_to_f64(val=%u, scale=%.6e, zp=%u)", val, scale, zero_point);
    double result = ((double)val - (double)zero_point) * scale;
    // Check for NaN/Inf, although less likely here unless scale is invalid
    // if (isnan(result) || isinf(result)) { LOG_WARN("Dequantization to f64 resulted in NaN or Inf!"); }
    return result;
}

// --- Removed model_quantize_inputs and model_dequantize_and_postprocess ---
// (Moved postprocessing part to inference_quantized.c)
