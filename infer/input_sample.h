/* Generated input sample header for validation */
#ifndef INPUT_SAMPLE_H
#define INPUT_SAMPLE_H

// Sample Input Parameters
#define INPUT_SCALE 0.003921569f
#define INPUT_ZP -128
#define INPUT_IS_ODD 1

// Peak info (to be filled per-sample)
// Use PEAK_R_M (row index in 0..31) and PEAK_C_M_10 (col index in 0..9)
#define PEAK_R_M 7
#define PEAK_C_M_10 1

// Normalized Float Patch (3x5)
extern const float TEST_INPUT_FLOAT[3][5];

// Pre-quantized Int8 Input (for debugging)
extern const int8_t TEST_INPUT_INT8[3 * 5];

#endif // INPUT_SAMPLE_H
