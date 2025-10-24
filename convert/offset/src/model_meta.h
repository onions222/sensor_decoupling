#ifndef MODEL_META_H
#define MODEL_META_H

#include <stdint.h>

// --- Model Shapes ---
#define MODEL_INPUT_X_SHAPE_SIZE (1*2*3*7)
#define MODEL_INPUT_META_SHAPE_SIZE (1*2)
#define MODEL_OUTPUT_SHAPE_SIZE (1*2)

// --- Final Post-Processing Scales ---
#define MODEL_FINAL_SCALE_DX 4.0000000000f
#define MODEL_FINAL_SCALE_DY 0.2500000000f

// --- Activation Quantization Params ---
#define MODEL_INPUT_X_SCALE 0.0306432154f
#define MODEL_INPUT_X_ZERO_POINT 30
#define MODEL_INPUT_META_SCALE 0.0044242386f
#define MODEL_INPUT_META_ZERO_POINT 0
#define MODEL_BLOCK1_DW_OUT_SCALE 0.0193685554f
#define MODEL_BLOCK1_DW_OUT_ZERO_POINT 61
#define MODEL_BLOCK1_OUT_SCALE 0.0224804208f
#define MODEL_BLOCK1_OUT_ZERO_POINT 0
#define MODEL_BLOCK2_DW_OUT_SCALE 0.0164499283f
#define MODEL_BLOCK2_DW_OUT_ZERO_POINT 87
#define MODEL_BLOCK2_OUT_SCALE 0.0159236975f
#define MODEL_BLOCK2_OUT_ZERO_POINT 0
#define MODEL_GAP_OUT_SCALE 0.0044242386f
#define MODEL_GAP_OUT_ZERO_POINT 0
#define MODEL_CAT_OUT_SCALE 0.0044242386f
#define MODEL_CAT_OUT_ZERO_POINT 0
#define MODEL_HEAD0_OUT_SCALE 0.0045876936f
#define MODEL_HEAD0_OUT_ZERO_POINT 0
#define MODEL_OUTPUT_SCALE 0.0006359584f
#define MODEL_OUTPUT_ZERO_POINT 74

// --- Layer Requantization Params (Per-Tensor/Per-Channel) ---

// --- Requantization Params for Layer: block1.dw ---
#define MODEL_BLOCK1_DW_IS_PER_CHANNEL 1
#define MODEL_BLOCK1_DW_NUM_CHANNELS 2

// --- Requantization Params for Layer: block1.pw ---
#define MODEL_BLOCK1_PW_IS_PER_CHANNEL 1
#define MODEL_BLOCK1_PW_NUM_CHANNELS 4

// --- Requantization Params for Layer: block2.dw ---
#define MODEL_BLOCK2_DW_IS_PER_CHANNEL 1
#define MODEL_BLOCK2_DW_NUM_CHANNELS 4

// --- Requantization Params for Layer: block2.pw ---
#define MODEL_BLOCK2_PW_IS_PER_CHANNEL 1
#define MODEL_BLOCK2_PW_NUM_CHANNELS 4

// --- Requantization Params for Layer: head.0 ---
#define MODEL_HEAD_0_IS_PER_CHANNEL 1
#define MODEL_HEAD_0_NUM_CHANNELS 8

// --- Requantization Params for Layer: head.2 ---
#define MODEL_HEAD_2_IS_PER_CHANNEL 1
#define MODEL_HEAD_2_NUM_CHANNELS 2

// --- Requantization Params for Layer: GAP ---
// Note: Maps block2_out scale to the re-quantized scale before cat (_scale_0)
#define MODEL_GAP_OUT_IS_PER_CHANNEL 0
#define MODEL_GAP_OUT_NUM_CHANNELS 1

#endif // MODEL_META_H