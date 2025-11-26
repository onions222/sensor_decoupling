#include <stdio.h>
#include <math.h>
#include <stdint.h>

#include "teacher_int8_infer.h"
#include "debug_data.h"
#include "frames_data.h"  // 由 Python 脚本导出的多帧 32x18 merging 数据

// 与 DifferentiableCoM_Patch_3x5 一致：返回 (center_x, center_y)
// center_x: 列方向 [0, DEBUG_PATCH_W-1]
// center_y: 行方向 [0, DEBUG_PATCH_H-1]
static void compute_local_com_xy(const float *patch, float *out_x, float *out_y) {
    float sum_w = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;

    for (int r = 0; r < DEBUG_PATCH_H; ++r) {
        for (int c = 0; c < DEBUG_PATCH_W; ++c) {
            int idx = r * DEBUG_PATCH_W + c;
            float v = patch[idx];
            sum_w += v;
            sum_x += v * (float)c;
            sum_y += v * (float)r;
        }
    }

    if (sum_w > 1e-12f) {
        *out_x = sum_x / sum_w;
        *out_y = sum_y / sum_w;
    } else {
        *out_x = 0.0f;
        *out_y = 0.0f;
    }
}

// 将 (local_x, local_y) + (peak_r_m, peak_c_m_10) -> 全局 18 列坐标
static void local_to_global_18(
    float local_x, float local_y,
    float peak_r_m, float peak_c_m_10,
    int is_odd,
    float *out_x18, float *out_y
) {
    const float ph_offset = (float)(DEBUG_PATCH_H / 2); // 3 -> 1
    const float pw_offset_10 = (float)(DEBUG_PATCH_W / 2); // 5 -> 2

    // 10 列坐标系下的全局坐标
    float global_x_10 = local_x + peak_c_m_10 - pw_offset_10;
    float global_y = local_y + peak_r_m - ph_offset;

    // clamp 到 [0, 9]
    if (global_x_10 < 0.0f) global_x_10 = 0.0f;
    if (global_x_10 > 9.0f) global_x_10 = 9.0f;

    float x_floor_f = floorf(global_x_10);
    float x_ceil_f  = ceilf(global_x_10);
    int x_floor = (int)x_floor_f;
    int x_ceil  = (int)x_ceil_f;

    if (x_floor < 0) x_floor = 0;
    if (x_floor > 9) x_floor = 9;
    if (x_ceil < 0) x_ceil = 0;
    if (x_ceil > 9) x_ceil = 9;

    float frac = global_x_10 - (float)x_floor;

    // 与 teacher_train 中的 odd_grid_10 / even_grid_10 一致
    static const float odd_grid_10[10]  = {0.5f, 2.5f, 4.5f, 6.5f, 8.0f, 9.0f, 10.5f, 12.5f, 14.5f, 16.5f};
    static const float even_grid_10[10] = {0.0f, 1.5f, 3.5f, 5.5f, 7.5f, 9.5f, 11.5f, 13.5f, 15.5f, 17.0f};

    const float *grid = is_odd ? odd_grid_10 : even_grid_10;

    float val_floor = grid[x_floor];
    float val_ceil  = grid[x_ceil];

    float global_x_18 = val_floor + (val_ceil - val_floor) * frac;

    *out_x18 = global_x_18;
    *out_y   = global_y;
}

static void run_debug_from_patches(void)
{
    int mismatch_cnt = 0;

    printf("Global coordinate comparison on %d samples:\n", DEBUG_NUM_SAMPLES);

    for (int i = 0; i < DEBUG_NUM_SAMPLES; ++i) {
        const float *in_patch  = debug_in_patches[i];
        int is_odd             = debug_is_odd[i];
        const float *ref_patch = debug_out_patches[i];

        float out_c_patch[DEBUG_PATCH_H * DEBUG_PATCH_W];

        // 用 C 侧 int8 网络跑一遍
        if (is_odd)
            teacher_int8_forward_odd(in_patch, out_c_patch);
        else
            teacher_int8_forward_even(in_patch, out_c_patch);

        // C 侧局部 CoM
        float local_x_c, local_y_c;
        compute_local_com_xy(out_c_patch, &local_x_c, &local_y_c);

        // C 侧全局坐标
        float gx_c, gy_c;
        local_to_global_18(
            local_x_c,
            local_y_c,
            debug_peak_r_m[i],
            debug_peak_c_m_10[i],
            is_odd,
            &gx_c,
            &gy_c
        );

        // Python 侧全局坐标（在 debug_data.h 里）
        float gx_ref = debug_global_x_18[i];
        float gy_ref = debug_global_y[i];

        float dx = fabsf(gx_c - gx_ref);
        float dy = fabsf(gy_c - gy_ref);
        float dist = sqrtf(dx * dx + dy * dy);

        printf("Sample %2d | is_odd=%d | Ref=(%.6f, %.6f), C=(%.6f, %.6f), "
               "|Δ|=(%.6f, %.6f), Dist=%.6f\n",
               i, is_odd, gx_ref, gy_ref, gx_c, gy_c, dx, dy, dist);

        if (dist > 1e-4f)
            mismatch_cnt++;
    }

    printf("\nSummary: %d / %d samples have global coord mismatch > 1e-4.\n",
           mismatch_cnt, DEBUG_NUM_SAMPLES);
}

int main(void) {
    // 示例：批量多帧推理，从 frames_data.h 导入的 32x18 merging 帧逐帧计算坐标

    printf("[BATCH] Total frames = %d\n", FRAMES_NUM);

    for (int i = 0; i < FRAMES_NUM; ++i) {
        // const float (*frame_18)[FRAME_W_18] = frames_merging_18[i];
        const int16_t (*frame_18)[FRAME_W_18] = frames_merging_18[i];

        float gx18 = 0.0f;
        float gy   = 0.0f;
        teacher_full_pipeline(frame_18, &gx18, &gy);

        printf("[FRAME %4d] x18 = %.6f, y = %.6f\n", i, gx18, gy);
    }

    // 如果需要做 C vs Python 的 patch 级对齐验证，临时打开：
    // run_debug_from_patches();

    return 0;
}