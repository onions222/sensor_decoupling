#ifndef MODEL_META_H
#define MODEL_META_H

#include <stdint.h>

// --- Model Arena Size ---
// Calculated based on the buffer sizes needed in model_forward_s8
// Sizes: 42 + 84 + 84 + 84 + 4 + 6 + 8 = 312 bytes.
// Add some padding for safety/alignment.
#define MODEL_ARENA_SIZE 320 // Must be >= required size (312)

// --- Model Shapes ---
#define MODEL_INPUT_X_SHAPE_SIZE (1*2*3*7) // 42 elements
#define MODEL_INPUT_META_SHAPE_SIZE (1*2)  // 2 elements
#define MODEL_OUTPUT_SHAPE_SIZE (1*2) // 2 elements

// --- Final Post-Processing Scales ---
// These are used only if calling inference_dequantize_postprocess_output
#define MODEL_FINAL_SCALE_DX 4.0000000000f
#define MODEL_FINAL_SCALE_DY 0.2500000000f

// --- Activation Quantization Params (Scales and Zero Points) ---
// Used for quantizing inputs (if starting from float) and dequantizing outputs
#define MODEL_INPUT_X_SCALE 0.0306432154f
#define MODEL_INPUT_X_ZERO_POINT 30
#define MODEL_INPUT_META_SCALE 0.0044242386f // Note: This is the original meta scale
#define MODEL_INPUT_META_ZERO_POINT 0       // Note: This is the original meta ZP

#define MODEL_BLOCK1_DW_OUT_SCALE 0.0193685554f
#define MODEL_BLOCK1_DW_OUT_ZERO_POINT 61
#define MODEL_BLOCK1_OUT_SCALE 0.0224804208f // Output of block1.pw (ReLU)
#define MODEL_BLOCK1_OUT_ZERO_POINT 0

#define MODEL_BLOCK2_DW_OUT_SCALE 0.0164499283f
#define MODEL_BLOCK2_DW_OUT_ZERO_POINT 87
#define MODEL_BLOCK2_OUT_SCALE 0.0159236975f // Output of block2.pw (ReLU)
#define MODEL_BLOCK2_OUT_ZERO_POINT 0

// IMPORTANT: GAP output is requantized to match META's expected scale/zp for the Cat layer
#define MODEL_GAP_OUT_SCALE MODEL_INPUT_META_SCALE // Use Meta's scale for GAP output requantization
#define MODEL_GAP_OUT_ZERO_POINT MODEL_INPUT_META_ZERO_POINT // Use Meta's ZP for GAP output requantization

// CAT output inherits the scale/zp from its inputs (which should now match)
#define MODEL_CAT_OUT_SCALE MODEL_GAP_OUT_SCALE
#define MODEL_CAT_OUT_ZERO_POINT MODEL_GAP_OUT_ZERO_POINT

#define MODEL_HEAD0_OUT_SCALE 0.0045876936f // Output of head.0 (ReLU)
#define MODEL_HEAD0_OUT_ZERO_POINT 0

#define MODEL_OUTPUT_SCALE 0.0006359584f // Output of head.2 (Linear, before dequant/tanh)
#define MODEL_OUTPUT_ZERO_POINT 74

// --- Layer Requantization Params (Per-Tensor/Per-Channel) ---
// These defines indicate if a layer uses per-channel quantization for its output
// and provide multipliers/shifts. They are directly used by the core operators.

// --- Requantization Params for Layer: block1.dw ---
#define MODEL_BLOCK1_DW_IS_PER_CHANNEL 1
#define MODEL_BLOCK1_DW_NUM_CHANNELS 2 // Matches OC

// --- Requantization Params for Layer: block1.pw ---
#define MODEL_BLOCK1_PW_IS_PER_CHANNEL 1
#define MODEL_BLOCK1_PW_NUM_CHANNELS 4 // Matches OC

// --- Requantization Params for Layer: block2.dw ---
#define MODEL_BLOCK2_DW_IS_PER_CHANNEL 1
#define MODEL_BLOCK2_DW_NUM_CHANNELS 4 // Matches OC

// --- Requantization Params for Layer: block2.pw ---
#define MODEL_BLOCK2_PW_IS_PER_CHANNEL 1
#define MODEL_BLOCK2_PW_NUM_CHANNELS 4 // Matches OC

// --- Requantization Params for Layer: head.0 ---
#define MODEL_HEAD_0_IS_PER_CHANNEL 1
#define MODEL_HEAD_0_NUM_CHANNELS 8 // Matches OutFeatures

// --- Requantization Params for Layer: head.2 ---
#define MODEL_HEAD_2_IS_PER_CHANNEL 1
#define MODEL_HEAD_2_NUM_CHANNELS 2 // Matches OutFeatures

// --- Requantization Params for Layer: GAP ---
// Note: The GAP implementation simulates Python's dequant->avg->requant.
// It uses MODEL_BLOCK2_OUT_SCALE and MODEL_GAP_OUT_SCALE directly.
// The multipliers/shifts below are technically exported but NOT USED by the current GAP op.
#define MODEL_GAP_OUT_IS_PER_CHANNEL 0 // Exported as per-tensor
#define MODEL_GAP_OUT_NUM_CHANNELS 1

#endif // MODEL_META_H
