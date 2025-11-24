#include "inference.h"

// ================= 内存池定义 =================
// 根据模型结构预分配静态内存
// 输入: 1x3x5
// L0: 4x3x5 (Conv 3x3)
// L1: 8x3x5 (Conv 3x3)
// L2: 4x3x5 (Conv 3x3)
// L3: 1x3x5 (Conv 1x1) - 输出
// 最大 buffer 需要容纳 8x3x5 = 120 字节
#define MAX_BUFFER_SIZE 256

// Ping-Pong Buffers 用于层间数据传递
int8_t buffer_A[MAX_BUFFER_SIZE];
int8_t buffer_B[MAX_BUFFER_SIZE];

// 已将 conv2d_layer 的实现移至 inference.c，并使其接受 input_zp 参数
// 这里不再保留局部重复实现，直接调用 inference.c 中的通用接口 conv2d_layer

int main() {
    printf("Starting MCU Inference Simulation...\n");

    // 1. 模拟数据获取
    printf("Data Acquisition: Sample loaded (Is Odd: %d)\n", INPUT_IS_ODD);
    
    // 2. 输入量化
    // 将 float 输入转换为 int8，存入 buffer_A
    // 输入维度: [1, 3, 5]
    Tensor t_in = { .data = buffer_A, .channels = 1, .height = 3, .width = 5 };
    quantize_input((float*)TEST_INPUT_FLOAT, t_in.data, 1*3*5, INPUT_SCALE, INPUT_ZP);
    
    printf("Input Quantized. Range example: %d, %d, %d\n", t_in.data[0], t_in.data[1], t_in.data[2]);

    // 3. 层级推理 (Pipeline)
    // 我们只演示 ODD 分支，根据 INPUT_IS_ODD 可以在实际代码中切换指针
    
    // --- Layer 0 (Conv 3x3) ---
    // Input: buffer_A -> Output: buffer_B
    // Shape: [1, 3, 5] -> [4, 3, 5] (Padding=1, Stride=1, Kernel=3, OutCh=4)
    Tensor t_l0 = { .data = buffer_B, .channels = L0_OUT_CH, .height = 3, .width = 5 };
    
    // 注意：ReLU 的 min 通常是 output_zp (对应 0.0f)，max 是 127
    // 如果使用了 ReLU6 或普通 ReLU，Quantization Aware Training 会体现在 output_zp 和 scale 上
    // 这里我们使用全范围 int8 [-128, 127] 作为 clamp 范围，除非有特定 ReLU 信息
    // 由于是 QAT，ReLU 的效果已经包含在输出的量化参数里了（负数会被截断或映射）
    // 这里的 out_min/max 用于模拟 int8 的溢出保护
    
    printf("Running Layer 0 (Conv 3x3)...\n");
    conv2d_layer(&t_in, &t_l0, 
                 L0_WEIGHTS, L0_BIAS, 
                 L0_K_H, L0_K_W, L0_PAD, 1, 
                 L0_MULT, L0_SHIFT,
                 INPUT_ZP, L0_OUT_ZP, // Input ZP, Output ZP
                 -128, 127);

    // --- Layer 1 (Conv 3x3) ---
    // Input: buffer_B -> Output: buffer_A
    // Shape: [4, 3, 5] -> [8, 3, 5]
    Tensor t_l1 = { .data = buffer_A, .channels = L1_OUT_CH, .height = 3, .width = 5 };
    printf("Running Layer 1 (Conv 3x3)...\n");
    conv2d_layer(&t_l0, &t_l1, 
                 L1_WEIGHTS, L1_BIAS, 
                 L1_K_H, L1_K_W, L1_PAD, 1, 
                 L1_MULT, L1_SHIFT,
                 L0_OUT_ZP, L1_OUT_ZP,
                 -128, 127);

    // --- Layer 2 (Conv 3x3) ---
    // Input: buffer_A -> Output: buffer_B
    // Shape: [8, 3, 5] -> [4, 3, 5]
    Tensor t_l2 = { .data = buffer_B, .channels = L2_OUT_CH, .height = 3, .width = 5 };
    printf("Running Layer 2 (Conv 3x3)...\n");
    conv2d_layer(&t_l1, &t_l2, 
                 L2_WEIGHTS, L2_BIAS, 
                 L2_K_H, L2_K_W, L2_PAD, 1, 
                 L2_MULT, L2_SHIFT,
                 L1_OUT_ZP, L2_OUT_ZP,
                 -128, 127);

    // --- Layer 3 (Conv 1x1) ---
    // Input: buffer_B -> Output: buffer_A (Final)
    // Shape: [4, 3, 5] -> [1, 3, 5]
    Tensor t_out = { .data = buffer_A, .channels = L3_OUT_CH, .height = 3, .width = 5 };
    printf("Running Layer 3 (Conv 1x1)...\n");
    conv2d_layer(&t_l2, &t_out, 
                 L3_WEIGHTS, L3_BIAS, 
                 L3_K_H, L3_K_W, L3_PAD, 1, 
                 L3_MULT, L3_SHIFT,
                 L2_OUT_ZP, L3_OUT_ZP,
                 -128, 127);

    // 4. 输出反量化
    // 最后一层的 Output Scale 通常在 weights.h 中没有直接定义宏 (我的脚本里需要补上)
    // 假设 L3_MULT 计算时用到的 effective scale 包含了 L3 Output Scale
    // 为了演示，我们假设 L3 的输出 scale 需要从 Python 脚本中额外获取
    // *修正*: 在 export_system.py 中，我会确保打印 L3 的 Output Scale 宏
    
    printf("\n=== Inference Result (1x3x5) ===\n");
    // 假设 L3_OUT_SCALE 在 weights.h 中定义了 (见 Python 脚本逻辑)
    // 如果没有定义，这里用 1.0f 占位调试
    #ifndef L3_OUT_SCALE
    #define L3_OUT_SCALE 0.00392f 
    #endif

    for (int h = 0; h < 3; h++) {
        for (int w = 0; w < 5; w++) {
            int idx = h * 5 + w;
            int8_t q_val = t_out.data[idx];
            float f_val = dequantize_output(q_val, L3_OUT_SCALE, L3_OUT_ZP);
            printf("%6.3f ", f_val);
        }
        printf("\n");
    }

    return 0;
}