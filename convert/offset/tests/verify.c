// #include "src/inference.h"
// #include "include/config.h"
// #include "src/model_meta.h" // (Auto-generated)
// #include "tests/test_helpers.h"
// #include <stdio.h>          // For printf
#include "../src/inference.h" // Use relative path from tests/ to src/
#include "../include/config.h"           // Use include path from Makefile (-Iinclude)
#include "../src/model_meta.h"// Use relative path from tests/ to src/
#include "test_helpers.h"     // Include helper functions (should be found relative to tests/)
#include <stdio.h>          // For printf
#include <stdlib.h>         // For malloc/free (only for intermediate buffers in test)
#include <stdbool.h>        // For bool
#include <string.h>         // For memset

// 从 model_meta.h 获取形状
#define X_SIZE (MODEL_INPUT_X_SHAPE_SIZE)
#define META_SIZE (MODEL_INPUT_META_SHAPE_SIZE)
#define OUT_SIZE (MODEL_OUTPUT_SHAPE_SIZE)

// 中间层大小 (number of elements)
#define BLOCK1_DW_SIZE (1*2*3*7) // Output shape [1, 2, 3, 7]
#define BLOCK1_PW_SIZE (1*4*3*7) // Output shape [1, 4, 3, 7]
#define BLOCK2_DW_SIZE (1*4*3*7) // Output shape [1, 4, 3, 7]
#define BLOCK2_PW_SIZE (1*4*3*7) // Output shape [1, 4, 3, 7]
#define GAP_REQUANT_SIZE (1*4)   // Output shape [1, 4]
#define CAT_SIZE (1*6)           // Output shape [1, 6]
#define HEAD0_SIZE (1*8)         // Output shape [1, 8]
#define HEAD2_SIZE (1*2)         // Output shape [1, 2]


// 缓冲区 (static global to avoid large stack allocation)
static float g_x_f[X_SIZE];
static float g_meta_f[META_SIZE];
static float g_c_out_f[OUT_SIZE];
static float g_golden_out_f[OUT_SIZE];

// --- 修正: Intermediate buffers are uint8_t ---
static uint8_t g_golden_block1_dw_q[BLOCK1_DW_SIZE];
static uint8_t g_golden_block1_pw_q[BLOCK1_PW_SIZE];
static uint8_t g_golden_block2_dw_q[BLOCK2_DW_SIZE];
static uint8_t g_golden_block2_pw_q[BLOCK2_PW_SIZE];
static uint8_t g_golden_gap_requant_q[GAP_REQUANT_SIZE];
static uint8_t g_golden_cat_q[CAT_SIZE];
static uint8_t g_golden_head0_q[HEAD0_SIZE];
static uint8_t g_golden_head2_q[HEAD2_SIZE];

static uint8_t g_c_block1_dw_q[BLOCK1_DW_SIZE];
static uint8_t g_c_block1_pw_q[BLOCK1_PW_SIZE];
static uint8_t g_c_block2_dw_q[BLOCK2_DW_SIZE];
static uint8_t g_c_block2_pw_q[BLOCK2_PW_SIZE];
static uint8_t g_c_gap_requant_q[GAP_REQUANT_SIZE];
static uint8_t g_c_cat_q[CAT_SIZE];
static uint8_t g_c_head0_q[HEAD0_SIZE];
static uint8_t g_c_head2_q[HEAD2_SIZE];

// --- 修正: Load all golden data (use load_bin_u8) ---
static bool load_all_golden_data(const char* data_dir) {
    char path[256];
    bool success = true;

    // Load float inputs/outputs
    snprintf(path, sizeof(path), "%s/input_x.bin", data_dir);
    if (load_bin_f32(path, g_x_f, X_SIZE) != 0) success = false;
    snprintf(path, sizeof(path), "%s/input_meta.bin", data_dir);
    if (load_bin_f32(path, g_meta_f, META_SIZE) != 0) success = false;
    snprintf(path, sizeof(path), "%s/golden_output.bin", data_dir);
    if (load_bin_f32(path, g_golden_out_f, OUT_SIZE) != 0) success = false;

    // Load integer intermediates (uint8)
    LOG_INFO("Loading golden intermediate integer data (uint8)...");
    snprintf(path, sizeof(path), "%s/golden_block1_dw_q.bin", data_dir);
    if (load_bin_u8(path, g_golden_block1_dw_q, BLOCK1_DW_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/golden_block1_pw_q.bin", data_dir);
    if (load_bin_u8(path, g_golden_block1_pw_q, BLOCK1_PW_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/golden_block2_dw_q.bin", data_dir);
    if (load_bin_u8(path, g_golden_block2_dw_q, BLOCK2_DW_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/golden_block2_pw_q.bin", data_dir);
    if (load_bin_u8(path, g_golden_block2_pw_q, BLOCK2_PW_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/golden_gap_requant_q.bin", data_dir);
    if (load_bin_u8(path, g_golden_gap_requant_q, GAP_REQUANT_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/golden_cat_q.bin", data_dir);
    if (load_bin_u8(path, g_golden_cat_q, CAT_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/golden_head0_q.bin", data_dir);
    if (load_bin_u8(path, g_golden_head0_q, HEAD0_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/golden_head2_q.bin", data_dir);
    if (load_bin_u8(path, g_golden_head2_q, HEAD2_SIZE) != 0) success = false; // Use load_bin_u8

    if (!success) {
        LOG_ERROR("Failed to load one or more golden data files from %s", data_dir);
    } else {
        LOG_INFO("Successfully loaded all golden data.");
    }
    return success;
}

// --- 修正: Load C dump data (use load_bin_u8) ---
static bool load_c_dump_data(const char* dump_dir) {
    char path[256];
    bool success = true;
    LOG_INFO("Loading C intermediate integer data dumps (uint8) from %s...", dump_dir);

    snprintf(path, sizeof(path), "%s/c_output_block1_dw_q.bin", dump_dir);
    if (load_bin_u8(path, g_c_block1_dw_q, BLOCK1_DW_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/c_output_block1_pw_q.bin", dump_dir);
    if (load_bin_u8(path, g_c_block1_pw_q, BLOCK1_PW_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/c_output_block2_dw_q.bin", dump_dir);
    if (load_bin_u8(path, g_c_block2_dw_q, BLOCK2_DW_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/c_output_block2_pw_q.bin", dump_dir);
    if (load_bin_u8(path, g_c_block2_pw_q, BLOCK2_PW_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/c_output_gap_requant_q.bin", dump_dir);
    if (load_bin_u8(path, g_c_gap_requant_q, GAP_REQUANT_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/c_output_cat_q.bin", dump_dir);
    if (load_bin_u8(path, g_c_cat_q, CAT_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/c_output_head0_q.bin", dump_dir);
    if (load_bin_u8(path, g_c_head0_q, HEAD0_SIZE) != 0) success = false; // Use load_bin_u8
    snprintf(path, sizeof(path), "%s/c_output_head2_q.bin", dump_dir);
    if (load_bin_u8(path, g_c_head2_q, HEAD2_SIZE) != 0) success = false; // Use load_bin_u8

     if (!success) {
        LOG_ERROR("Failed to load one or more C dump files from %s", dump_dir);
        LOG_ERROR("Did you compile with -DDUMP_INTERMEDIATES and run the C code successfully first?");
    } else {
        LOG_INFO("Successfully loaded C dump data.");
    }
    return success;
}


int main(int argc, char *argv[]) {
    // ... (Argument parsing remains the same) ...
    LOG_INFO("--- C-Side Verification ---");
    const char* golden_data_dir = "verify_data";
    if (argc > 1) {
        golden_data_dir = argv[1];
        LOG_INFO("Using golden data directory: %s", golden_data_dir);
    } else {
        LOG_INFO("Using default golden data directory: %s", golden_data_dir);
    }


    // 1. 初始化
    if (inference_init() != 0) {
        LOG_ERROR("Inference initialization failed!");
        return -1;
    }

    // 2. 加载所有黄金数据
    LOG_INFO("Loading golden data...");
    if (!load_all_golden_data(golden_data_dir)) {
        return -1;
    }

    // 3. 运行 C 库
    LOG_INFO("Running C inference (float-to-float)...");
    int ret = inference_run_float(g_x_f, g_meta_f, g_c_out_f);
    if (ret != 0) {
        LOG_ERROR("inference_run_float failed!");
        return -1;
    }
    LOG_INFO("C inference run completed.");

    // 4. 对比最终浮点结果
    LOG_INFO("Comparing final C float output with golden float data...");
    int final_float_failures = compare_results_f32(
        g_c_out_f,
        g_golden_out_f,
        OUT_SIZE,
        INFERENCE_C_VERIFY_TOLERANCE_ABS_FLOAT,
        INFERENCE_C_VERIFY_TOLERANCE_REL_FLOAT
    );

    // 5. --- 修正: 对比中间 uint8 结果 ---
    int intermediate_failures = 0;
#ifdef DUMP_INTERMEDIATES
    LOG_INFO("--- Intermediate Layer Comparison START ---");
    if (!load_c_dump_data(INTERMEDIATE_DUMP_DIR)) {
         LOG_ERROR("Could not load C intermediate dumps for comparison.");
         intermediate_failures = 1;
    } else {
        LOG_INFO("Comparing intermediate integer layers (uint8 C vs uint8 Golden)...");
        // --- 修正: Use compare_intermediate_u8 ---
        intermediate_failures += compare_intermediate_u8(g_c_block1_dw_q, g_golden_block1_dw_q, BLOCK1_DW_SIZE, INFERENCE_C_VERIFY_TOLERANCE_ABS_UINT8, "Block1 DW");
        intermediate_failures += compare_intermediate_u8(g_c_block1_pw_q, g_golden_block1_pw_q, BLOCK1_PW_SIZE, INFERENCE_C_VERIFY_TOLERANCE_ABS_UINT8, "Block1 PW");
        intermediate_failures += compare_intermediate_u8(g_c_block2_dw_q, g_golden_block2_dw_q, BLOCK2_DW_SIZE, INFERENCE_C_VERIFY_TOLERANCE_ABS_UINT8, "Block2 DW");
        intermediate_failures += compare_intermediate_u8(g_c_block2_pw_q, g_golden_block2_pw_q, BLOCK2_PW_SIZE, INFERENCE_C_VERIFY_TOLERANCE_ABS_UINT8, "Block2 PW");
        intermediate_failures += compare_intermediate_u8(g_c_gap_requant_q, g_golden_gap_requant_q, GAP_REQUANT_SIZE, INFERENCE_C_VERIFY_TOLERANCE_ABS_UINT8, "GAP Requant");
        intermediate_failures += compare_intermediate_u8(g_c_cat_q, g_golden_cat_q, CAT_SIZE, INFERENCE_C_VERIFY_TOLERANCE_ABS_UINT8, "Cat");
        intermediate_failures += compare_intermediate_u8(g_c_head0_q, g_golden_head0_q, HEAD0_SIZE, INFERENCE_C_VERIFY_TOLERANCE_ABS_UINT8, "Head0");
        intermediate_failures += compare_intermediate_u8(g_c_head2_q, g_golden_head2_q, HEAD2_SIZE, INFERENCE_C_VERIFY_TOLERANCE_ABS_UINT8, "Head2 (Final Int)");
    }
    LOG_INFO("--- Intermediate Layer Comparison END ---");
#else
    LOG_WARN("Intermediate layer comparison skipped. Compile with -DDUMP_INTERMEDIATES to enable.");
#endif // DUMP_INTERMEDIATES


    // 6. 最终结论
    int total_failures = final_float_failures + intermediate_failures;
    if (total_failures == 0) {
        LOG_INFO("--- VERIFICATION PASSED (Final Float & Intermediates) ---");
        printf("\n*** Verification PASSED ***\n");
    } else {
        LOG_ERROR("--- VERIFICATION FAILED ---");
        LOG_ERROR("  Final Float Mismatches: %d", final_float_failures);
        LOG_ERROR("  Intermediate Int Mismatches: %d", intermediate_failures);
        printf("\n*** Verification FAILED ***\n");
    }

    return (total_failures > 0);
}

