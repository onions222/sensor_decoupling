#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <stdint.h>
extern const int8_t g_block1_dw_weight[18];
extern const int32_t g_block1_dw_bias[2];
extern const int32_t g_block1_dw_multiplier[2];
extern const int32_t g_block1_dw_shift[2];
extern const int8_t g_block1_pw_weight[8];
extern const int32_t g_block1_pw_bias[4];
extern const int32_t g_block1_pw_multiplier[4];
extern const int32_t g_block1_pw_shift[4];
extern const int8_t g_block2_dw_weight[36];
extern const int32_t g_block2_dw_bias[4];
extern const int32_t g_block2_dw_multiplier[4];
extern const int32_t g_block2_dw_shift[4];
extern const int8_t g_block2_pw_weight[16];
extern const int32_t g_block2_pw_bias[4];
extern const int32_t g_block2_pw_multiplier[4];
extern const int32_t g_block2_pw_shift[4];
extern const int8_t g_head_0_weight[48];
extern const int32_t g_head_0_bias[8];
extern const int32_t g_head_0_multiplier[8];
extern const int32_t g_head_0_shift[8];
extern const int8_t g_head_2_weight[16];
extern const int32_t g_head_2_bias[2];
extern const int32_t g_head_2_multiplier[2];
extern const int32_t g_head_2_shift[2];

#endif // WEIGHTS_H