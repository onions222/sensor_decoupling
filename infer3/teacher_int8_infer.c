#include <stdint.h>
#include <math.h>
#include <string.h>

#include "teacher_int8_params.h"
#include "teacher_int8_shapes.h"
#include "teacher_int8_scales.h"
#include "teacher_int8_infer.h"

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