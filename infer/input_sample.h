/* Auto-generated golden sample header */
#ifndef INPUT_SAMPLE_H
#define INPUT_SAMPLE_H

#include <stdint.h>

#define INPUT_SCALE 0.003921569f
#define INPUT_ZP -128
#define INPUT_IS_ODD 1
#define PEAK_R_M 5
#define PEAK_C_M_10 1

#define GOLDEN_COM_X_FLOAT 0.500000000f
#define GOLDEN_COM_Y_FLOAT 4.000000000f
#define GOLDEN_COM_X_INT8 0.500000000f
#define GOLDEN_COM_Y_INT8 4.000000000f

extern const float TEST_INPUT_FLOAT[3][5];
extern const int8_t TEST_INPUT_INT8[15];
extern const int8_t GOLDEN_OUTPUT_INT8[15];
extern const float GOLDEN_OUTPUT_FLOAT[15];

#endif // INPUT_SAMPLE_H
