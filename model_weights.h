#ifndef MODEL_WEIGHTS_H_MODEL_WEIGHTS
#define MODEL_WEIGHTS_H_MODEL_WEIGHTS

#include <stdint.h>

// --- Quantized Model Weights and Biases (Extern Declarations) ---
// Extracted from: best_qat_int8.pt
// Model Structure: MLP Input=26 -> 32 -> 16 -> 2

// Weight: net.0.weight (Size: 832)
extern const int8_t net_0_weight[832];

// Bias: net.0.bias (Size: 32)
extern const float net_0_bias[32];

// Weight: net.2.weight (Size: 512)
extern const int8_t net_2_weight[512];

// Bias: net.2.bias (Size: 16)
extern const float net_2_bias[16];

// Weight: net.4.weight (Size: 32)
extern const int8_t net_4_weight[32];

// Bias: net.4.bias (Size: 2)
extern const float net_4_bias[2];

#endif // MODEL_WEIGHTS_H_MODEL_WEIGHTS