#include "inference_core.h"
#include "weights.h"      // (Auto-generated) - Contains weight_type_t arrays
#include "model_meta.h"   // (Auto-generated) - Contains act_type_t zero points
#include <string.h>       // for memset, strerror
// --- 确保 <math.h> 包含 nearbyint (C99) ---
#include <math.h>         // for float helpers (floor, nearbyint, roundf, tanhf, fabsf, isnan, isinf, modff, fmodf)
#include <limits.h>       // for INT_MAX, INT_MIN, UINT8_MAX
#include <inttypes.h>     // For PRId32, PRIu8 format specifier
#include <stdbool.h>      // For bool type

// --- Include stdio.h only if dumping intermediates ---
#ifdef DUMP_INTERMEDIATES
#include <stdio.h>    // For file operations (FILE, fopen, etc.)
#include <sys/stat.h> // For struct stat, stat, mkdir
#include <errno.h>    // For errno, EEXIST
    #ifdef _WIN32
    #include <direct.h> // For _mkdir on Windows
    #endif

static int ensure_dir_exists(const char* path) {
    struct stat st = {0};
    if (stat(path, &st) == -1) {
        #ifdef _WIN32
            if (_mkdir(path) != 0 && errno != EEXIST) {
        #else
            if (mkdir(path, 0700) != 0 && errno != EEXIST) {
        #endif
            LOG_ERROR("Failed to create directory %s: %s", path, strerror(errno));
            return -1; // Return error
        }
        LOG_INFO("Created directory: %s", path);
    }
    return 0; // Return success
}
static void dump_intermediate_tensor(const char* name, const act_type_t* data, size_t size) {
    if (ensure_dir_exists(INTERMEDIATE_DUMP_DIR) != 0) { return; }
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "%s/c_output_%s.bin", INTERMEDIATE_DUMP_DIR, name);
    FILE* f = fopen(filepath, "wb");
    if (!f) { LOG_ERROR("Failed to open file for writing intermediate dump: %s", filepath); return; }
    size_t write_count = fwrite(data, sizeof(act_type_t), size, f);
    fclose(f);
    if (write_count != size) { LOG_ERROR("File write error for intermediate dump: %s. Expected %zu, wrote %zu", filepath, size, write_count); }
    else { LOG_DEBUG("Dumped intermediate: %s (Size: %zu, Type: uint8)", name, size); }
}
#else
#define dump_intermediate_tensor(name, data, size)
#endif

// =================================================================
// 1. TFLite Micro 风格的定点数学 (纯 32-bit)
// =================================================================
// 辅助函数: (a * b) 的高 32 位 (未使用, 但保留)
static inline int32_t SaturatingRoundingDoublingHighMul_S64(int32_t a, int32_t b) {
    int64_t a_64 = a;
    int64_t b_64 = b;
    int64_t ab_64 = a_64 * b_64;
    bool overflow = (a == INT32_MIN && b == INT32_MIN);
    int64_t nudge = (1LL << 30); // 舍入 0.5 for Q31
    int64_t result_64 = (ab_64 + nudge) >> 31;
    if (overflow) { return INT32_MAX; }
    if (result_64 > INT32_MAX) return INT32_MAX;
    if (result_64 < INT32_MIN) return INT32_MIN;
    return (int32_t)result_64;
}

// 辅助函数: (val + round) >> shift (Round-Half-Up)
// --- 保持 requantization 的舍入方式为 round-half-up (TFLite 常用) ---
static inline int32_t ShiftRightRounded_S64(int64_t val, int32_t shift) {
    if (shift <= 0) {
        int32_t left_shift = -shift;
        const int64_t max_val = INT32_MAX; const int64_t min_val = INT32_MIN;
        if (left_shift >= 63) { return (val == 0) ? 0 : ((val > 0) ? INT32_MAX : INT32_MIN); }
        int64_t result_64 = val << left_shift;
        if (result_64 > max_val) { return INT32_MAX; }
        if (result_64 < min_val) { return INT32_MIN; }
        return (int32_t)result_64;
    }
    if (shift >= 63) { return (val >= 0) ? 0 : -1; }

    const int64_t max_val = INT32_MAX; const int64_t min_val = INT32_MIN;
    uint64_t abs_val = (val < 0) ? (uint64_t)(-val) : (uint64_t)val;
    uint64_t divisor = (uint64_t)1 << shift;
    uint64_t quotient = abs_val >> shift;
    uint64_t remainder = abs_val & (divisor - 1);
    uint64_t halfway = divisor >> 1;

    bool round_up = false;
    // --- Round half up logic ---
    if (remainder >= halfway) {
        round_up = true;
    }

    if (round_up) {
        quotient += 1ULL;
    }

    int64_t result_64 = (val < 0) ? -(int64_t)quotient : (int64_t)quotient;

    if (result_64 > max_val) { return INT32_MAX; }
    if (result_64 < min_val) { return INT32_MIN; }
    return (int32_t)result_64;
}


static int32_t MultiplyByQuantizedMultiplier(acc_type_t x, int32_t quantized_multiplier, int32_t shift) {
    // Calculate the 64-bit product
    int64_t product = (int64_t)x * quantized_multiplier;
    LOG_TRACE("  MulByQuantMult: x=%" PRId32 ", mult=%" PRId32 ", shift=%d -> product=%" PRId64, x, quantized_multiplier, shift, product);
    // Use the standard round-half-up shift
    int32_t result = ShiftRightRounded_S64(product, shift);
    LOG_TRACE("  MulByQuantMult: -> shifted_rounded=%" PRId32, result);
    return result;
}

static act_type_t requantize_s32_to_u8( acc_type_t accum, int32_t multiplier, int32_t shift, act_type_t output_zp) {
    LOG_TRACE("Requantize: accum=%" PRId32 ", mult=%" PRId32 ", shift=%d, out_zp=%u", accum, multiplier, shift, output_zp);
    // Call multiplier function
    int32_t scaled_acc = MultiplyByQuantizedMultiplier(accum, multiplier, shift);
    LOG_TRACE("  Requantize: scaled_acc = %" PRId32, scaled_acc);
    acc_type_t shifted_acc = saturate_add_s32(scaled_acc, (acc_type_t)output_zp);
    LOG_TRACE("  Requantize: shifted_acc (with zp) = %" PRId32, shifted_acc);
    act_type_t final_val = saturate_s32_to_u8(shifted_acc);
    LOG_TRACE("  Requantize: final_val (uint8) = %u", final_val);
    return final_val;
}

// =================================================================
// 2. 算子实现
// =================================================================
static inline int32_t get_index_nchw(int32_t h, int32_t w, int32_t c,int32_t H, int32_t W) {
    return c * H * W + h * W + w;
}
static inline int32_t get_weight_index(int32_t oc, int32_t kh, int32_t kw, int32_t ic, int32_t K, int32_t IC_per_G) { return oc * (K * K * IC_per_G) + kh * (K * IC_per_G) + kw * (IC_per_G) + ic; }

static void layer_conv2d_s8(
    act_type_t* out,
    const act_type_t* in,
    const weight_type_t* weight,
    const acc_type_t* bias, // Original (unfused) bias
    const ConvParams* p,
    const int32_t* out_multiplier_per_channel,
    const int32_t* out_shift_per_channel
) {
    const int32_t IC_per_G = p->C / p->G;
    const act_type_t relu_limit = p->out_zp;
    bool is_block1_pw = (p->C == 2 && p->K == 1 && p->G == 1); // For debug trace

    LOG_DEBUG("Conv2D: In(%" PRId32 ",%" PRId32 ",%" PRId32 ") Out(%" PRId32 ",%" PRId32 ",%" PRId32 ") K=%" PRId32 " S=%" PRId32 " P=%" PRId32 " G=%" PRId32 " PerChan=%d InZp=%"PRIu8" OutZp=%"PRIu8,
        p->H, p->W, p->C, p->OH, p->OW, p->OC, p->K, p->S, p->P, p->G, p->is_per_channel, p->in_zp, p->out_zp);

    for (int32_t oh = 0; oh < p->OH; ++oh) {
        for (int32_t ow = 0; ow < p->OW; ++ow) {
            for (int32_t oc = 0; oc < p->OC; ++oc) {
                acc_type_t acc = 0; // Initialize acc to 0
                bool trace_this_pixel = is_block1_pw && (oh==0 && ow==0 && oc==0); // For specific pixel trace
                if(trace_this_pixel) LOG_TRACE("--- Tracing B1PW [0,0,%d] ---", oc);

                const int32_t g_idx = (p->G > 1) ? (oc / (p->OC / p->G)) : 0;
                for (int32_t kh = 0; kh < p->K; ++kh) {
                    for (int32_t kw = 0; kw < p->K; ++kw) {
                        for (int32_t ic_g = 0; ic_g < IC_per_G; ++ic_g) {
                            const int32_t ih = oh * p->S + kh - p->P;
                            const int32_t iw = ow * p->S + kw - p->P;
                            if (ih < 0 || ih >= p->H || iw < 0 || iw >= p->W) { continue; }
                            const int32_t ic_abs = g_idx * IC_per_G + ic_g;
                            int32_t in_idx = get_index_nchw(ih, iw, ic_abs, p->H, p->W);
                            int32_t w_idx = get_weight_index(oc, kh, kw, ic_g, p->K, IC_per_G);

                            acc_type_t in_val = (acc_type_t)in[in_idx];
                            acc_type_t in_zp_val = (acc_type_t)p->in_zp;
                            acc_type_t weight_val = (acc_type_t)weight[w_idx];
                            acc_type_t term = (in_val - in_zp_val) * weight_val;
                            acc = saturate_add_s32(acc, term);

                            if(trace_this_pixel) LOG_TRACE("  MAC: kh=%d, kw=%d, ic_g=%d | in=%u, w=%d | term=%d -> acc=%" PRId32, kh, kw, ic_g, (uint8_t)in_val, (int8_t)weight_val, term, acc);
                        }
                    }
                }
                acc = saturate_add_s32(acc, bias[oc]); // Add bias AFTER MAC loop
                if(trace_this_pixel) LOG_TRACE("Final Acc + Bias before Requant: %" PRId32, acc);

                int32_t current_multiplier; int32_t current_shift;
                if (p->is_per_channel) { current_multiplier = out_multiplier_per_channel[oc]; current_shift = out_shift_per_channel[oc]; }
                else { current_multiplier = *out_multiplier_per_channel; current_shift = *out_shift_per_channel; }

                if(trace_this_pixel) LOG_TRACE("Requant Params: mult=%" PRId32 ", shift=%d, out_zp=%u", current_multiplier, current_shift, p->out_zp);
                act_type_t out_val = requantize_s32_to_u8(acc, current_multiplier, current_shift, p->out_zp);

                if(trace_this_pixel) LOG_TRACE("Requant Result (pre-ReLU): %u", out_val);

                if (p->relu) {
                    out_val = (out_val < relu_limit) ? relu_limit : out_val;
                    if(trace_this_pixel) LOG_TRACE("After ReLU (limit=%u): %u", relu_limit, out_val);
                }
                out[get_index_nchw(oh, ow, oc, p->OH, p->OW)] = out_val;
            }
        }
    }
}

// --- Forward declarations needed for GAP modification ---
static float dequantize_u8_to_f32(act_type_t val, float scale, act_type_t zero_point);
// --- 最终修复 1: 添加一个新的 f64 dequantize 函数 ---
static double dequantize_u8_to_f64(act_type_t val, double scale, act_type_t zero_point);
static act_type_t quantize_f64_to_u8(double val, double scale, act_type_t zero_point);


// --- MODIFIED: layer_global_avg_pool_s8 to simulate Python's dequant->avg->requant ---
// --- FIX: Use double precision for float accumulator ---
static void layer_global_avg_pool_s8(
    act_type_t* out,          // Output buffer (uint8)
    const act_type_t* in,     // Input buffer (uint8)
    int32_t H, int32_t W, int32_t C,
    act_type_t in_zp,         // Input zero point
    act_type_t out_zp,        // Target output zero point (for final requantization)
    const int32_t* out_multiplier, // NOT USED in this version
    const int32_t* out_shift,      // NOT USED in this version
    int is_per_channel             // Should be 0
) {
    LOG_DEBUG("GAP (Simulating Python): In(%" PRId32 ",%" PRId32 ",%" PRId32 ") InZp=%"PRIu8" OutZp=%"PRIu8, H, W, C, in_zp, out_zp);
    const int32_t pool_size = H * W;
    if (pool_size <= 0) {
        LOG_ERROR("GAP pool size is zero or negative!");
        memset(out, out_zp, C * sizeof(act_type_t));
        return;
    }

    // --- Need Input Scale and Target Output Scale ---
    // These scales are defined in model_meta.h
    const float in_scale = MODEL_BLOCK2_OUT_SCALE;   // Scale of the input tensor (output of block2.pw)
    const float out_scale = MODEL_GAP_OUT_SCALE; // Target scale for the output (input scale for Cat)

    if (out_scale <= 0.0f) {
        LOG_ERROR("GAP output scale is non-positive!");
        memset(out, out_zp, C * sizeof(act_type_t));
        return;
    }


    for (int32_t c = 0; c < C; ++c) {
        // --- 1. Accumulate sum in double (float64) to match PyTorch's .mean() precision ---
        double float_sum = 0.0; // Use double
        for (int32_t h = 0; h < H; ++h) {
            for (int32_t w = 0; w < W; ++w) {
                act_type_t q_val = in[get_index_nchw(h, w, c, H, W)];
                // --- 最终修复 2: 调用 f64 dequantize, 在 double 精度下执行乘法 ---
                double dq_val = dequantize_u8_to_f64(q_val, (double)in_scale, in_zp);
                float_sum += dq_val; // Accumulate as double (double + double)
            }
        }

        // --- 2. Calculate float average (in double) ---
        double float_avg = float_sum / (double)pool_size; // Use double
        LOG_TRACE("  GAP Ch %d: float_sum=%.6e, float_avg=%.6e", c, float_sum, float_avg);

        // --- 3. Requantize the float average to target uint8 ---
        // (This already calls quantize_f64_to_u8)
        act_type_t out_val = quantize_f64_to_u8(float_avg, (double)out_scale, out_zp);
        LOG_TRACE("  GAP Ch %d: requantized_avg (uint8)=%u", c, out_val);

        out[c] = out_val;
    }
}


static void layer_linear_s8(
    act_type_t* out,
    const act_type_t* in,
    const weight_type_t* weight,
    const acc_type_t* bias, // Original (unfused) bias
    const LinearParams* p,
    const int32_t* out_multiplier_per_channel,
    const int32_t* out_shift_per_channel
) {
    LOG_DEBUG("Linear: In(%" PRId32 ") Out(%" PRId32 ") PerChan=%d InZp=%"PRIu8" OutZp=%"PRIu8,
        p->In, p->Out, p->is_per_channel, p->in_zp, p->out_zp);
    const act_type_t relu_limit = p->out_zp;

    for (int32_t out_c = 0; out_c < p->Out; ++out_c) {
        acc_type_t acc = 0; // Initialize acc to 0
        for (int32_t in_c = 0; in_c < p->In; ++in_c) {
            int32_t w_idx = out_c * p->In + in_c;
            acc = saturate_add_s32(acc,
                                  ((acc_type_t)in[in_c] - (acc_type_t)p->in_zp)
                                  * (acc_type_t)weight[w_idx]);
        }
        acc = saturate_add_s32(acc, bias[out_c]); // Add bias AFTER MAC loop

        int32_t current_multiplier; int32_t current_shift;
        if (p->is_per_channel) { 
            current_multiplier = out_multiplier_per_channel[out_c]; 
            current_shift = out_shift_per_channel[out_c];
        }
        else { 
            current_multiplier = *out_multiplier_per_channel; 
            current_shift = *out_shift_per_channel; 
        }
        act_type_t out_val = requantize_s32_to_u8(acc, current_multiplier, current_shift, p->out_zp);
        if (p->relu) {
            out_val = (out_val < relu_limit) ? relu_limit : out_val;
        }
        out[out_c] = out_val;
    }
}

static void layer_cat_s8( act_type_t* out, const act_type_t* in1, int32_t C1, const act_type_t* in2, int32_t C2) {
    // --- 检查 qparam。现在 C 代码的 meta_q 输入应该与 gap_out 匹配
    if (MODEL_GAP_OUT_ZERO_POINT != MODEL_INPUT_META_ZERO_POINT || MODEL_CAT_OUT_ZERO_POINT != MODEL_GAP_OUT_ZERO_POINT) {
        // --- BUG 1 修复后, 这个警告不应该再触发 ---
        // (除非 meta_q 的 scale/zp 与 cat 的 scale/zp 仍有不同)
        LOG_WARN("Cat qparams mismatch: GAP_OUT_ZP=%u, META_IN_ZP=%u, CAT_OUT_ZP=%u",
                 MODEL_GAP_OUT_ZERO_POINT, MODEL_INPUT_META_ZERO_POINT, MODEL_CAT_OUT_ZERO_POINT);
    }
    // Bug 1 修复后, 两个输入的 qparams 应该匹配, memcpy 是正确的
    memcpy(out,      in1, C1 * sizeof(act_type_t));
    memcpy(out + C1, in2, C2 * sizeof(act_type_t));
}


// =================================================================
// 3. 模型图
// =================================================================
static const ConvParams P_BLOCK1_DW = { .H = 3, .W = 7, .C = 2, .OH = 3, .OW = 7, .OC = 2, .K = 3, .S = 1, .P = 1, .G = 2, .in_zp = MODEL_INPUT_X_ZERO_POINT, .out_zp = MODEL_BLOCK1_DW_OUT_ZERO_POINT, .relu = 0, .is_per_channel = MODEL_BLOCK1_DW_IS_PER_CHANNEL };
static const ConvParams P_BLOCK1_PW = { .H = 3, .W = 7, .C = 2, .OH = 3, .OW = 7, .OC = 4, .K = 1, .S = 1, .P = 0, .G = 1, .in_zp = MODEL_BLOCK1_DW_OUT_ZERO_POINT, .out_zp = MODEL_BLOCK1_OUT_ZERO_POINT, .relu = 1, .is_per_channel = MODEL_BLOCK1_PW_IS_PER_CHANNEL };
static const ConvParams P_BLOCK2_DW = { .H = 3, .W = 7, .C = 4, .OH = 3, .OW = 7, .OC = 4, .K = 3, .S = 1, .P = 1, .G = 4, .in_zp = MODEL_BLOCK1_OUT_ZERO_POINT, .out_zp = MODEL_BLOCK2_DW_OUT_ZERO_POINT, .relu = 0, .is_per_channel = MODEL_BLOCK2_DW_IS_PER_CHANNEL };
static const ConvParams P_BLOCK2_PW = { .H = 3, .W = 7, .C = 4, .OH = 3, .OW = 7, .OC = 4, .K = 1, .S = 1, .P = 0, .G = 1, .in_zp = MODEL_BLOCK2_DW_OUT_ZERO_POINT, .out_zp = MODEL_BLOCK2_OUT_ZERO_POINT, .relu = 1, .is_per_channel = MODEL_BLOCK2_PW_IS_PER_CHANNEL };
static const LinearParams P_HEAD_0 = { .In = 6, .Out = 8, .in_zp = MODEL_CAT_OUT_ZERO_POINT, .out_zp = MODEL_HEAD0_OUT_ZERO_POINT, .relu = 1, .is_per_channel = MODEL_HEAD_0_IS_PER_CHANNEL };
static const LinearParams P_HEAD_2 = { .In = 8, .Out = 2, .in_zp = MODEL_HEAD0_OUT_ZERO_POINT, .out_zp = MODEL_OUTPUT_ZERO_POINT, .relu = 0, .is_per_channel = MODEL_HEAD_2_IS_PER_CHANNEL };

int model_forward_s8(
    const act_type_t* x_q,
    const act_type_t* meta_q,
    act_type_t* out_q,
    uint8_t* arena
) {
    LOG_DEBUG("Model Forward (Quantized - uint8 activations, Adjusted MAC)");

    // Arena buffer allocation
    act_type_t* buf_block1_dw_out = (act_type_t*)(arena);               // Size: 42
    act_type_t* buf_block1_pw_out = (act_type_t*)(arena + 42);          // Size: 84
    act_type_t* buf_block2_dw_out = (act_type_t*)(arena + 42 + 84);     // Size: 84
    act_type_t* buf_block2_pw_out = (act_type_t*)(arena + 42 + 84 + 84);// Size: 84
    act_type_t* buf_gap_out       = (act_type_t*)(arena + 42 + 84 + 84 + 84); // Size: 4
    act_type_t* buf_cat_out       = (act_type_t*)(arena + 42 + 84 + 84 + 84 + 4); // Size: 6
    act_type_t* buf_head0_out     = (act_type_t*)(arena + 42 + 84 + 84 + 84 + 4 + 6); // Size: 8

    // Check required arena size
    const size_t required_arena = 42 + 84 + 84 + 84 + 4 + 6 + 8; // Sum of buffer sizes
    #if MODEL_ARENA_SIZE < required_arena
    #error "MODEL_ARENA_SIZE is too small for activation buffers"
    #endif

    // Sizes for dumping (match buffer sizes)
    const size_t size_block1_dw = 42;
    const size_t size_block1_pw = 84;
    const size_t size_block2_dw = 84;
    const size_t size_block2_pw = 84;
    const size_t size_gap       = 4;
    const size_t size_cat       = 6;
    const size_t size_head0     = 8;
    const size_t size_head2     = 2; // Final output size

    // --- Block 1 ---
    layer_conv2d_s8(buf_block1_dw_out, x_q, g_block1_dw_weight, g_block1_dw_bias, &P_BLOCK1_DW, g_block1_dw_multiplier, g_block1_dw_shift);
    dump_intermediate_tensor("block1_dw_q", buf_block1_dw_out, size_block1_dw);

    layer_conv2d_s8(buf_block1_pw_out, buf_block1_dw_out, g_block1_pw_weight, g_block1_pw_bias, &P_BLOCK1_PW, g_block1_pw_multiplier, g_block1_pw_shift);
    dump_intermediate_tensor("block1_pw_q", buf_block1_pw_out, size_block1_pw);

    // --- Block 2 ---
    layer_conv2d_s8(buf_block2_dw_out, buf_block1_pw_out, g_block2_dw_weight, g_block2_dw_bias, &P_BLOCK2_DW, g_block2_dw_multiplier, g_block2_dw_shift);
    dump_intermediate_tensor("block2_dw_q", buf_block2_dw_out, size_block2_dw);

    layer_conv2d_s8(buf_block2_pw_out, buf_block2_dw_out, g_block2_pw_weight, g_block2_pw_bias, &P_BLOCK2_PW, g_block2_pw_multiplier, g_block2_pw_shift);
    dump_intermediate_tensor("block2_pw_q", buf_block2_pw_out, size_block2_pw);

    // --- GAP (Now using the modified version simulating Python) ---
    layer_global_avg_pool_s8(buf_gap_out, buf_block2_pw_out, 3, 7, 4,
                             MODEL_BLOCK2_OUT_ZERO_POINT, // Input ZP
                             MODEL_GAP_OUT_ZERO_POINT,    // Target Output ZP
                             NULL, // g_gap_out_multiplier (ignored)
                             NULL, // g_gap_out_shift (ignored)
                             0     // MODEL_GAP_OUT_IS_PER_CHANNEL (ignored but should be 0)
                            );
    dump_intermediate_tensor("gap_requant_q", buf_gap_out, size_gap);

    // --- Cat ---
    // meta_q 已经被 model_quantize_inputs 转换
    layer_cat_s8(buf_cat_out, buf_gap_out, 4, meta_q, 2);
    dump_intermediate_tensor("cat_q", buf_cat_out, size_cat);

    // --- Head ---
    layer_linear_s8(buf_head0_out, buf_cat_out, g_head_0_weight, g_head_0_bias, &P_HEAD_0, g_head_0_multiplier, g_head_0_shift);
    dump_intermediate_tensor("head0_q", buf_head0_out, size_head0);

    layer_linear_s8(out_q, buf_head0_out, g_head_2_weight, g_head_2_bias, &P_HEAD_2, g_head_2_multiplier, g_head_2_shift);
    dump_intermediate_tensor("head2_q", out_q, size_head2); // Dump final quantized output

    LOG_DEBUG("Model Forward Done.");
    return 0; // Ensure return 0 on success
}

// =================================================================
// 4. 浮点封装辅助函数
// =================================================================
static inline float a_tanh_f32(float x) { return tanhf(x); }

// --- 修正: quantize_f32_to_u8 -> quantize_f64_to_u8 (use C99 round) ---
static act_type_t quantize_f64_to_u8(double val, double scale, act_type_t zero_point) {
    LOG_TRACE("quantize_f64_to_u8(val=%.6e, scale=%.6e, zp=%u)", val, scale, zero_point);
    if (scale <= 0.0 || isnan(scale) || isinf(scale)) { // Check for <= 0
        LOG_WARN("Quantization scale is non-positive, NaN or Inf!");
        return zero_point;
    }
    double div_result = val / scale;
    if (isnan(div_result) || isinf(div_result)){
        LOG_WARN("Intermediate division result is NaN or Inf (val=%.6e / scale=%.6e)", val, scale);
        // Match PyTorch behavior for Inf/NaN (often saturates)
        return (val >= 0.0) ? UINT8_MAX : 0;
    }
    double q_val_f = div_result + zero_point;
    LOG_TRACE("  Intermediate q_val_f = %.6f", q_val_f);

    // --- 最终修复: 切换回 "round-half-up" (floor(x+0.5)) 以匹配 TFLite 整数 requant 逻辑 ---
    // (这假设 q_val_f 总是 >= 0, 因为它是 (val/scale) + zero_point)
    double rounded_q_val_f = floor(q_val_f + 0.5);

    LOG_TRACE("  Rounded (floor(x+0.5)/half-up) q_val_f = %.1f", rounded_q_val_f);

    // Clamp to uint8 range
    if (rounded_q_val_f > 255.0) rounded_q_val_f = 255.0;
    if (rounded_q_val_f < 0.0) rounded_q_val_f = 0.0;

    act_type_t result = (act_type_t)rounded_q_val_f;
    LOG_TRACE("  Clamped result = %u", result);
    return result;
}

// --- 最终修复 1 (续): 添加 f64 dequantize 函数 ---
static double dequantize_u8_to_f64(act_type_t val, double scale, act_type_t zero_point) {
    // 跟踪日志在 f64 中意义不大, 暂时禁用
    // LOG_TRACE("dequantize_u8_to_f64(val=%u, scale=%.6e, zp=%u)", val, scale, zero_point);
    double result = ((double)val - (double)zero_point) * scale;
    // LOG_TRACE("  Dequantized result = %.6e", result);
    // if (isnan(result) || isinf(result)) { LOG_WARN("Dequantization resulted in NaN or Inf!"); }
    return result;
}

static float dequantize_u8_to_f32(act_type_t val, float scale, act_type_t zero_point) {
    LOG_TRACE("dequantize_u8_to_f32(val=%u, scale=%.4e, zp=%u)", val, scale, zero_point);
    float result = ((float)val - (float)zero_point) * scale;
    LOG_TRACE("  Dequantized result = %.4e", result);
    if (isnan(result) || isinf(result)) { LOG_WARN("Dequantization resulted in NaN or Inf!"); }
    return result;
}

void model_quantize_inputs( const float* x_f, const float* meta_f, act_type_t* x_q, act_type_t* meta_q) {
    LOG_DEBUG("Quantizing inputs to uint8...");
    // Input X: Quantize normally
    for (int i = 0; i < MODEL_INPUT_X_SHAPE_SIZE; ++i) { 
        x_q[i] = quantize_f64_to_u8((double)x_f[i], (double)MODEL_INPUT_X_SCALE, MODEL_INPUT_X_ZERO_POINT); 
    }
    
    // --- FIX 1: Quantize META input using CAT's input scale/zp ---
    // This matches the logic in verify_with_python.py where meta is quantized to cat's params
    LOG_DEBUG("Quantizing meta input to CAT's scale/zp (SCALE=%.4e, ZP=%u)", MODEL_GAP_OUT_SCALE, MODEL_GAP_OUT_ZERO_POINT);
    for (int i = 0; i < MODEL_INPUT_META_SHAPE_SIZE; ++i) { 
        meta_q[i] = quantize_f64_to_u8((double)meta_f[i], (double)MODEL_GAP_OUT_SCALE, MODEL_GAP_OUT_ZERO_POINT); 
    }

    #ifdef DUMP_INTERMEDIATES
    dump_intermediate_tensor("input_x_q", x_q, MODEL_INPUT_X_SHAPE_SIZE);
    dump_intermediate_tensor("input_meta_q", meta_q, MODEL_INPUT_META_SHAPE_SIZE);
    #endif
}

void model_dequantize_and_postprocess( const act_type_t* out_q, float* final_out_f) {
    LOG_DEBUG("Dequantizing uint8 output and post-processing...");
    float out_f[MODEL_OUTPUT_SHAPE_SIZE];
    // (final output dequantize can safely use f32)
    out_f[0] = dequantize_u8_to_f32(out_q[0], MODEL_OUTPUT_SCALE, MODEL_OUTPUT_ZERO_POINT);
    out_f[1] = dequantize_u8_to_f32(out_q[1], MODEL_OUTPUT_SCALE, MODEL_OUTPUT_ZERO_POINT);
    float tanh_out_f[MODEL_OUTPUT_SHAPE_SIZE];
    tanh_out_f[0] = a_tanh_f32(out_f[0]);
    tanh_out_f[1] = a_tanh_f32(out_f[1]);
    final_out_f[0] = tanh_out_f[0] * MODEL_FINAL_SCALE_DX;
    final_out_f[1] = tanh_out_f[1] * MODEL_FINAL_SCALE_DY;
}

