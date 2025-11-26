#include <stdint.h>
#include <math.h>
#include <string.h>
#include <limits.h>

#include "teacher_int8_params.h"
#include "teacher_int8_shapes.h"
#include "teacher_int8_scales.h"
#include "teacher_int8_infer.h"

// 原始 merging 帧的尺寸与压缩后尺寸
#define FRAME_H        32
#define FRAME_W_18     18
#define FRAME_W_10     10

// -----------------------------------------
// 基础工具
// -----------------------------------------

// 对称量化: x_real / scale -> int8, 截断到 [-128, 127]
static inline int8_t quantize_symmetric(float x, float scale) {
    if (scale <= 0.0f) {
        return 0;
    }
    float q = x / scale;
    if (q > 127.0f) q = 127.0f;
    if (q < -128.0f) q = -128.0f;
    int32_t qi = (int32_t)lrintf(q);  // 四舍五入
    if (qi > 127) qi = 127;
    if (qi < -128) qi = -128;
    return (int8_t)qi;
}

// 通用 int8 部署卷积：
//   输入: x_real[IC * H * W]
//   权重: w_int8[OC * IC * KH * KW]（导出时是 PyTorch 的 Conv2d 权重展平顺序）
//   bias: bias[OC]（浮点）
//   输出: y_real[OC * H_out * W_out]
// 计算: acc = sum( x_q * (w_int8 - w_zp) )，然后
//       y = acc * (x_scale * w_scale) + bias，最后 ReLU
static void conv2d_int8_deploy(
    const float* x_real,
    int in_c,
    int in_h,
    int in_w,
    const int8_t* w_int8,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride,
    int pad,
    int w_zp,
    float x_scale,
    float w_scale,
    const float* bias,
    float* y_real
) {
    // 输出尺寸（假设 stride 相同 & pad 对称）
    int out_h = (in_h + 2 * pad - kernel_h) / stride + 1;
    int out_w = (in_w + 2 * pad - kernel_w) / stride + 1;

    // 遍历输出通道 & 空间位置
    for (int oc = 0; oc < out_c; ++oc) {
        for (int oh = 0; oh < out_h; ++oh) {
            for (int ow = 0; ow < out_w; ++ow) {

                int64_t acc = 0;  // 累加用 64bit，防止溢出

                for (int ic = 0; ic < in_c; ++ic) {
                    for (int kh = 0; kh < kernel_h; ++kh) {
                        for (int kw = 0; kw < kernel_w; ++kw) {

                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;

                            if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
                                continue;  // 超出边界视为 0
                            }

                            // 取输入 x_real(ic, ih, iw)
                            int x_idx = (ic * in_h + ih) * in_w + iw;
                            float x_val = x_real[x_idx];
                            int8_t x_q = quantize_symmetric(x_val, x_scale);

                            // 取权重 w_int8(oc, ic, kh, kw)
                            int w_idx =
                                ((oc * in_c + ic) * kernel_h + kh) * kernel_w + kw;
                            int8_t w_q = w_int8[w_idx];

                            int32_t w_center = (int32_t)w_q - (int32_t)w_zp;
                            int32_t prod = (int32_t)x_q * w_center;

                            acc += (int64_t)prod;
                        }
                    }
                }

                float y = (float)acc * (x_scale * w_scale);
                if (bias != NULL) {
                    y += bias[oc];
                }
                // ReLU
                if (y < 0.0f) {
                    y = 0.0f;
                }

                int y_idx = (oc * out_h + oh) * out_w + ow;
                y_real[y_idx] = y;
            }
        }
    }
}

// -----------------------------------------
// odd 分支推理: 4 层卷积
// -----------------------------------------

void teacher_int8_forward_odd(const float* in_patch, float* out_patch) {
    // 输入尺寸: 1 x PATCH_H x PATCH_W
    const int in_c  = 1;
    const int in_h  = PATCH_H;
    const int in_w  = PATCH_W;

    // 从 shapes 中取各层通道数
    const int c0_in  = ODD_ODD_NET_NET_0_IC;
    const int c0_out = ODD_ODD_NET_NET_0_OC;
    const int c1_in  = ODD_ODD_NET_NET_3_IC;
    const int c1_out = ODD_ODD_NET_NET_3_OC;
    const int c2_in  = ODD_ODD_NET_NET_6_IC;
    const int c2_out = ODD_ODD_NET_NET_6_OC;
    const int c3_in  = ODD_ODD_NET_NET_9_IC;
    const int c3_out = ODD_ODD_NET_NET_9_OC;

    // 这里假定每层输出空间尺寸仍为 PATCH_H x PATCH_W
    const int buf_hw = PATCH_H * PATCH_W;

    // 中间缓冲区
    float buf0[/* c0_out * H * W */ 16 * PATCH_H * PATCH_W];
    float buf1[/* c1_out * H * W */ 16 * PATCH_H * PATCH_W];
    float buf2[/* c2_out * H * W */ 16 * PATCH_H * PATCH_W];

    // layer 0: odd_net.net.0
    conv2d_int8_deploy(
        in_patch,
        c0_in, in_h, in_w,
        odd_odd_net_net_0_w,
        c0_out,
        ODD_ODD_NET_NET_0_KH,
        ODD_ODD_NET_NET_0_KW,
        ODD_ODD_NET_NET_0_STR,
        ODD_ODD_NET_NET_0_PAD,
        odd_odd_net_net_0_w_zp,
        odd_odd_net_net_0_x_scale,
        odd_odd_net_net_0_w_scale,
        odd_odd_net_net_0_bias,
        buf0
    );

    // layer 1: odd_net.net.3
    conv2d_int8_deploy(
        buf0,
        c1_in, in_h, in_w,
        odd_odd_net_net_3_w,
        c1_out,
        ODD_ODD_NET_NET_3_KH,
        ODD_ODD_NET_NET_3_KW,
        ODD_ODD_NET_NET_3_STR,
        ODD_ODD_NET_NET_3_PAD,
        odd_odd_net_net_3_w_zp,
        odd_odd_net_net_3_x_scale,
        odd_odd_net_net_3_w_scale,
        odd_odd_net_net_3_bias,
        buf1
    );

    // layer 2: odd_net.net.6
    conv2d_int8_deploy(
        buf1,
        c2_in, in_h, in_w,
        odd_odd_net_net_6_w,
        c2_out,
        ODD_ODD_NET_NET_6_KH,
        ODD_ODD_NET_NET_6_KW,
        ODD_ODD_NET_NET_6_STR,
        ODD_ODD_NET_NET_6_PAD,
        odd_odd_net_net_6_w_zp,
        odd_odd_net_net_6_x_scale,
        odd_odd_net_net_6_w_scale,
        odd_odd_net_net_6_bias,
        buf2
    );

    // layer 3: odd_net.net.9 (最后一层 输出应该为 1 通道)
    float buf3[1 * PATCH_H * PATCH_W];
    conv2d_int8_deploy(
        buf2,
        c3_in, in_h, in_w,
        odd_odd_net_net_9_w,
        c3_out,
        ODD_ODD_NET_NET_9_KH,
        ODD_ODD_NET_NET_9_KW,
        ODD_ODD_NET_NET_9_STR,
        ODD_ODD_NET_NET_9_PAD,
        odd_odd_net_net_9_w_zp,
        odd_odd_net_net_9_x_scale,
        odd_odd_net_net_9_w_scale,
        odd_odd_net_net_9_bias,
        buf3
    );

    // 输出 flatten 成 1 x PATCH_H x PATCH_W
    // c3_out 应为 1
    memcpy(out_patch, buf3, sizeof(float) * buf_hw);
}

// -----------------------------------------
// even 分支推理: 4 层卷积
// -----------------------------------------

void teacher_int8_forward_even(const float* in_patch, float* out_patch) {
    // 输入尺寸: 1 x PATCH_H x PATCH_W
    const int in_c  = 1;
    const int in_h  = PATCH_H;
    const int in_w  = PATCH_W;

    const int c0_in  = EVEN_EVEN_NET_NET_0_IC;
    const int c0_out = EVEN_EVEN_NET_NET_0_OC;
    const int c1_in  = EVEN_EVEN_NET_NET_3_IC;
    const int c1_out = EVEN_EVEN_NET_NET_3_OC;
    const int c2_in  = EVEN_EVEN_NET_NET_6_IC;
    const int c2_out = EVEN_EVEN_NET_NET_6_OC;
    const int c3_in  = EVEN_EVEN_NET_NET_9_IC;
    const int c3_out = EVEN_EVEN_NET_NET_9_OC;

    const int buf_hw = PATCH_H * PATCH_W;

    float buf0[16 * PATCH_H * PATCH_W];
    float buf1[16 * PATCH_H * PATCH_W];
    float buf2[16 * PATCH_H * PATCH_W];
    float buf3[1  * PATCH_H * PATCH_W];

    // layer 0: even_net.net.0
    conv2d_int8_deploy(
        in_patch,
        c0_in, in_h, in_w,
        even_even_net_net_0_w,
        c0_out,
        EVEN_EVEN_NET_NET_0_KH,
        EVEN_EVEN_NET_NET_0_KW,
        EVEN_EVEN_NET_NET_0_STR,
        EVEN_EVEN_NET_NET_0_PAD,
        even_even_net_net_0_w_zp,
        even_even_net_net_0_x_scale,
        even_even_net_net_0_w_scale,
        even_even_net_net_0_bias,
        buf0
    );

    // layer 1: even_net.net.3
    conv2d_int8_deploy(
        buf0,
        c1_in, in_h, in_w,
        even_even_net_net_3_w,
        c1_out,
        EVEN_EVEN_NET_NET_3_KH,
        EVEN_EVEN_NET_NET_3_KW,
        EVEN_EVEN_NET_NET_3_STR,
        EVEN_EVEN_NET_NET_3_PAD,
        even_even_net_net_3_w_zp,
        even_even_net_net_3_x_scale,
        even_even_net_net_3_w_scale,
        even_even_net_net_3_bias,
        buf1
    );

    // layer 2: even_net.net.6
    conv2d_int8_deploy(
        buf1,
        c2_in, in_h, in_w,
        even_even_net_net_6_w,
        c2_out,
        EVEN_EVEN_NET_NET_6_KH,
        EVEN_EVEN_NET_NET_6_KW,
        EVEN_EVEN_NET_NET_6_STR,
        EVEN_EVEN_NET_NET_6_PAD,
        even_even_net_net_6_w_zp,
        even_even_net_net_6_x_scale,
        even_even_net_net_6_w_scale,
        even_even_net_net_6_bias,
        buf2
    );

    // layer 3: even_net.net.9
    conv2d_int8_deploy(
        buf2,
        c3_in, in_h, in_w,
        even_even_net_net_9_w,
        c3_out,
        EVEN_EVEN_NET_NET_9_KH,
        EVEN_EVEN_NET_NET_9_KW,
        EVEN_EVEN_NET_NET_9_STR,
        EVEN_EVEN_NET_NET_9_PAD,
        even_even_net_net_9_w_zp,
        even_even_net_net_9_x_scale,
        even_even_net_net_9_w_scale,
        even_even_net_net_9_bias,
        buf3
    );

    memcpy(out_patch, buf3, sizeof(float) * buf_hw);
}

// -----------------------------------------
// 从 32x18 merging 帧到全局坐标的 C 部署流水线
// -----------------------------------------

// TODO: 根据 teacher_train.py 中的 ODD/EVEN 映射表，
// 将这里的占位实现替换为真实的 18->10 列 merge 逻辑。

static const int ODD_PAIRS[][2] = {
    /* {col_left, col_right}, ... */
    /* 例如: {0, 1}, {2, 3}, ... */
    {0, 1}, {2, 3}, {4, 5}, {6, 7}, {10, 11}, {12, 13}, {14, 15}, {16, 17}
};

static const int EVEN_PAIRS[][2] = {
    /* {col_left, col_right}, ... */
    {1, 2}, {3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12}, {13, 14}, {15, 16}
};

static const int ODD_SINGLE[][2] = {
    /* {col_18, col_10}, ... */
    {8, 4}, {9, 5}
};

static const int EVEN_SINGLE[][2] = {
    /* {col_18, col_10}, ... */
    {0, 0}, {17, 9}
};

static const int ODD_PAIRS_COUNT   = sizeof(ODD_PAIRS)   / sizeof(ODD_PAIRS[0]);
static const int EVEN_PAIRS_COUNT  = sizeof(EVEN_PAIRS)  / sizeof(EVEN_PAIRS[0]);
static const int ODD_SINGLE_COUNT  = sizeof(ODD_SINGLE)  / sizeof(ODD_SINGLE[0]);
static const int EVEN_SINGLE_COUNT = sizeof(EVEN_SINGLE) / sizeof(EVEN_SINGLE[0]);

// Python 里的 ODD_MAP_18_TO_10 / EVEN_MAP_18_TO_10 是长度 18 的数组：
static const int ODD_MAP_18_TO_10[18] = {
    0, 0,  // 0 -> 0, 1 -> 0
    1, 1,  // 2 -> 1, 3 -> 1
    2, 2,  // 4 -> 2, 5 -> 2
    3, 3,  // 6 -> 3, 7 -> 3
    4,     // 8 -> 4
    5,     // 9 -> 5
    6, 6,  // 10 -> 6, 11 -> 6
    7, 7,  // 12 -> 7, 13 -> 7
    8, 8,  // 14 -> 8, 15 -> 8
    9, 9   // 16 -> 9, 17 -> 9
};

static const int EVEN_MAP_18_TO_10[18] = {
    0,     // 0 -> 0
    1, 1,  // 1 -> 1, 2 -> 1
    2, 2,  // 3 -> 2, 4 -> 2
    3, 3,  // 5 -> 3, 6 -> 3
    4, 4,  // 7 -> 4, 8 -> 4
    5, 5,  // 9 -> 5, 10 -> 5
    6, 6,  // 11 -> 6, 12 -> 6
    7, 7,  // 13 -> 7, 14 -> 7
    8, 8,  // 15 -> 8, 16 -> 8
    9      // 17 -> 9
};

static void compress_merging_to_10_col(
    const int16_t merging_18[FRAME_H][FRAME_W_18],
    int16_t merging_10[FRAME_H][FRAME_W_10]
) {
    // 先清零
    for (int r = 0; r < FRAME_H; ++r) {
        for (int c = 0; c < FRAME_W_10; ++c) {
            merging_10[r][c] = 0;
        }
    }

    for (int r = 0; r < FRAME_H; ++r) {
        int is_odd = (r % 2 == 1);

        if (is_odd) {
            // odd 行：先处理成对的列
            for (int i = 0; i < ODD_PAIRS_COUNT; ++i) {
                int col_left  = ODD_PAIRS[i][0];
                int col_right = ODD_PAIRS[i][1];
                int col_10    = ODD_MAP_18_TO_10[col_left];

                int16_t v_left  = merging_18[r][col_left];
                int16_t v_right = merging_18[r][col_right];
                int16_t avg_val = (int16_t)(((int32_t)v_left + (int32_t)v_right) / 2);

                merging_10[r][col_10] = avg_val;
            }
            // 再处理单列直拷
            for (int i = 0; i < ODD_SINGLE_COUNT; ++i) {
                int col_18 = ODD_SINGLE[i][0];
                int col_10 = ODD_SINGLE[i][1];
                merging_10[r][col_10] = merging_18[r][col_18];
            }
        } else {
            // even 行
            for (int i = 0; i < EVEN_PAIRS_COUNT; ++i) {
                int col_left  = EVEN_PAIRS[i][0];
                int col_right = EVEN_PAIRS[i][1];
                int col_10    = EVEN_MAP_18_TO_10[col_left];

                int16_t v_left  = merging_18[r][col_left];
                int16_t v_right = merging_18[r][col_right];
                int16_t avg_val = (int16_t)(((int32_t)v_left + (int32_t)v_right) / 2);

                merging_10[r][col_10] = avg_val;
            }
            for (int i = 0; i < EVEN_SINGLE_COUNT; ++i) {
                int col_18 = EVEN_SINGLE[i][0];
                int col_10 = EVEN_SINGLE[i][1];
                merging_10[r][col_10] = merging_18[r][col_18];
            }
        }
    }
}

// 在 32x10 上寻找 peak 以及 is_odd 标记
static void find_peak_m_10(
    const int16_t merging_10[FRAME_H][FRAME_W_10],
    int *peak_r_m,
    int *peak_c_m_10,
    int *is_odd
) {
    int32_t max_v = INT32_MIN;
    int best_r = 0;
    int best_c = 0;

    for (int r = 0; r < FRAME_H; ++r) {
        for (int c = 0; c < FRAME_W_10; ++c) {
            int32_t v = (int32_t)merging_10[r][c];
            if (v > max_v) {
                max_v = v;
                best_r = r;
                best_c = c;
            }
        }
    }

    *peak_r_m    = best_r;
    *peak_c_m_10 = best_c;
    *is_odd      = (best_r % 2 == 1);
}

// 截取与 PyTorch 中 pad + slice 等价的 3x5 patch
static void extract_merging_patch_3x5(
    const int16_t merging_10[FRAME_H][FRAME_W_10],
    int peak_r_m,
    int peak_c_m_10,
    float patch[PATCH_H * PATCH_W]
) {
    const int ph = PATCH_H;   // 3
    const int pw = PATCH_W;   // 5
    const int half_h = ph / 2; // 1
    const int half_w = pw / 2; // 2

    for (int dr = -half_h; dr <= half_h; ++dr) {
        for (int dc = -half_w; dc <= half_w; ++dc) {
            int r  = peak_r_m    + dr;
            int c  = peak_c_m_10 + dc;
            int pr = dr + half_h; // 0..2
            int pc = dc + half_w; // 0..4
            int idx_p = pr * pw + pc;

            float v = 0.0f;
            if (r >= 0 && r < FRAME_H && c >= 0 && c < FRAME_W_10) {
                v = (float)merging_10[r][c];
            }
            patch[idx_p] = v;
        }
    }
}

// 对 3x5 patch 做 clamp_min(0) 和 归一化到和为 1
static void normalize_patch_3x5(float patch[PATCH_H * PATCH_W]) {
    float sum = 0.0f;
    for (int i = 0; i < PATCH_H * PATCH_W; ++i) {
        if (patch[i] < 0.0f) {
            patch[i] = 0.0f;
        }
        sum += patch[i];
    }
    const float eps = 1e-8f;
    float inv = 1.0f / (sum + eps);
    for (int i = 0; i < PATCH_H * PATCH_W; ++i) {
        patch[i] *= inv;
    }
}

// 3x5 patch 上的 CoM 计算（与 DifferentiableCoM_Patch_3x5 对应）
static void compute_local_com_3x5(
    const float patch[PATCH_H * PATCH_W],
    float *local_x,
    float *local_y
) {
    float mass = 0.0f;
    float xw = 0.0f;
    float yw = 0.0f;

    for (int r = 0; r < PATCH_H; ++r) {
        for (int c = 0; c < PATCH_W; ++c) {
            int idx = r * PATCH_W + c;
            float v = patch[idx];
            mass += v;
            xw += v * (float)c; // x: 0..4
            yw += v * (float)r; // y: 0..2
        }
    }

    const float eps = 1e-8f;
    float inv = 1.0f / (mass + eps);
    *local_x = xw * inv;
    *local_y = yw * inv;
}

// 将 10 列坐标映射到 18 列坐标（与 CoM_from_Patch_V12 的栅格插值一致）
// 注意：这里的 odd_grid_10 / even_grid_10 请根据 teacher_train.py 填写真实数值。
static float map_x10_to_x18(float x10, int is_odd) {
    static const float odd_grid_10[10]  = {
        0.5f, 2.5f, 4.5f, 6.5f, 8.0f, 9.0f, 10.5f, 12.5f, 14.5f, 16.5f
    };
    static const float even_grid_10[10] = {
        0.0f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f, 15.5f, 17.0f
    };
    const float *grid = is_odd ? odd_grid_10 : even_grid_10;

    if (x10 < 0.0f) x10 = 0.0f;
    if (x10 > 9.0f) x10 = 9.0f;

    int x_floor = (int)floorf(x10);
    int x_ceil  = (int)ceilf (x10);
    if (x_floor < 0) x_floor = 0;
    if (x_floor > 9) x_floor = 9;
    if (x_ceil  < 0) x_ceil  = 0;
    if (x_ceil  > 9) x_ceil  = 9;

    float vf = grid[x_floor];
    float vc = grid[x_ceil];
    float frac = x10 - (float)x_floor;
    return vf + (vc - vf) * frac;
}

// 将 dec_patch + peak 信息组合成 18 列全局坐标
static void compute_global_coords_from_patch(
    const float dec_patch[PATCH_H * PATCH_W],
    int peak_r_m,
    int peak_c_m_10,
    int is_odd,
    float *x18,
    float *y
) {
    float local_x = 0.0f;
    float local_y = 0.0f;
    compute_local_com_3x5(dec_patch, &local_x, &local_y);

    // 与 CoM_from_Patch_V12 的公式一致：
    // global_x10 = local_x + peak_cs_10_col - 2
    // global_y   = local_y + peak_rs        - 1
    float global_x10 = local_x + (float)peak_c_m_10 - 2.0f;
    float global_y   = local_y + (float)peak_r_m    - 1.0f;

    float global_x18 = map_x10_to_x18(global_x10, is_odd);

    *x18 = global_x18;
    *y   = global_y;
}

// 对外暴露的完整流水线：从 32x18 merging 帧到全局 18 列坐标
void teacher_full_pipeline(
    const int16_t merging_18[FRAME_H][FRAME_W_18],
    float *out_x18,
    float *out_y
) {
    // 1) 32x18 -> 32x10（整型）
    int16_t merging_10[FRAME_H][FRAME_W_10];
    compress_merging_to_10_col(merging_18, merging_10);

    // 2) 在 32x10 上找 peak + is_odd（整型）
    int peak_r_m    = 0;
    int peak_c_m_10 = 0;
    int is_odd      = 0;
    find_peak_m_10(merging_10, &peak_r_m, &peak_c_m_10, &is_odd);

    // 3) 截 3x5 patch 并归一化（这里才转换为 float）
    float patch_in[PATCH_H * PATCH_W];
    extract_merging_patch_3x5(merging_10, peak_r_m, peak_c_m_10, patch_in);
    normalize_patch_3x5(patch_in);

    // 4) Teacher int8 推理
    float patch_dec[PATCH_H * PATCH_W];
    if (is_odd) {
        teacher_int8_forward_odd(patch_in, patch_dec);
    } else {
        teacher_int8_forward_even(patch_in, patch_dec);
    }

    // 5) 从 dec_patch 计算 18 列全局坐标
    compute_global_coords_from_patch(
        patch_dec,
        peak_r_m,
        peak_c_m_10,
        is_odd,
        out_x18,
        out_y
    );
}