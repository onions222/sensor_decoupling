#ifndef TEACHER_INT8_INFER_H
#define TEACHER_INT8_INFER_H

#include <stdint.h>

// === 全局尺寸定义 ===
#define FRAME_H      32
#define FRAME_W_18   18
#define FRAME_W_10   10

#define PATCH_H      3
#define PATCH_W      5

// === 你的推理函数 ===
void teacher_full_pipeline(
    const int16_t merging_18[FRAME_H][FRAME_W_18],
    float *out_x18,
    float *out_y
);

void teacher_int8_forward_odd(const float in_patch[PATCH_H*PATCH_W],
                              float out_patch[PATCH_H*PATCH_W]);

void teacher_int8_forward_even(const float in_patch[PATCH_H*PATCH_W],
                               float out_patch[PATCH_H*PATCH_W]);

#endif  // TEACHER_INT8_INFER_H