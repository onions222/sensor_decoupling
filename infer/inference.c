#include "inference.h"

// 输入量化: Float -> Int8
void quantize_input(const float* input_float, int8_t* output_int8, int count, float scale, int32_t zp) {
    for (int i = 0; i < count; i++) {
        // Formula: q = round(r / scale) + zp
        float val = input_float[i] / scale;
        int32_t q = (int32_t)(val >= 0 ? val + 0.5f : val - 0.5f) + zp;
        output_int8[i] = (int8_t)clamp(q, -128, 127);
    }
}

// 卷积层推理 (支持 3x3 和 1x1)
// 包含 Bias Add, Re-quantization 和 Activation (ReLU via out_min)
void conv2d_layer(const Tensor* input, Tensor* output, 
                  const int8_t* weights, const int32_t* bias,
                  int k_h, int k_w, int pad, int stride,
                  int32_t multiplier, int shift, int32_t input_zp, int32_t out_zp,
                  int32_t out_min, int32_t out_max) {
    
    int in_c = input->channels;
    int out_c = output->channels;
    int in_h = input->height;
    int in_w = input->width;
    int out_h = output->height;
    int out_w = output->width;

    // 遍历输出的每一个点 (h_out, w_out) 和每一个输出通道 (oc)
    for (int oc = 0; oc < out_c; oc++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {

                // 初始化累加器为 Bias
                int32_t acc = bias[oc];

                // 计算输入特征图上的起始坐标
                int in_y_origin = (oy * stride) - pad;
                int in_x_origin = (ox * stride) - pad;

                // 卷积核循环
                for (int ic = 0; ic < in_c; ic++) {
                    for (int ky = 0; ky < k_h; ky++) {
                        for (int kx = 0; kx < k_w; kx++) {
                            int iy = in_y_origin + ky;
                            int ix = in_x_origin + kx;

                            // 边界检查 (Padding 0)
                            if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w) {
                                int input_idx = ic * (in_h * in_w) + iy * in_w + ix;
                                int weight_idx = oc * (in_c * k_h * k_w) + ic * (k_h * k_w) + ky * k_w + kx;

                                int32_t input_q = (int32_t)input->data[input_idx];
                                int32_t w_q = (int32_t)weights[weight_idx];

                                // 减去输入零点，再乘以权重
                                int32_t adj_in = input_q - input_zp;
                                acc += adj_in * w_q;
                            }
                        }
                    }
                }

                // Re-quantization: 将 int32 acc -> int32 通过乘法和移位
                int32_t out_val = multiply_by_quantized_multiplier(acc, multiplier, shift);
                out_val += out_zp;

                // Activation / clamp
                out_val = clamp(out_val, out_min, out_max);

                // 写入输出 (NCHW 风格扁平)
                int out_idx = oc * (out_h * out_w) + oy * out_w + ox;
                output->data[out_idx] = (int8_t)out_val;
            }
        }
    }
}

// 输出反量化: Int8 -> Float
float dequantize_output(int8_t val, float scale, int32_t zp) {
    return ((float)val - (float)zp) * scale;
}