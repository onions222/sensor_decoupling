#include <stdio.h>
#include <math.h>

#include "teacher_int8_infer.h"
#include "debug_data.h"

int main(void) {
    int total_mismatch = 0;

    for (int i = 0; i < DEBUG_NUM_SAMPLES; ++i) {
        const float* in_patch = debug_in_patches[i];
        int is_odd = debug_is_odd[i];
        const float* out_ref = debug_out_patches[i];

        float out_c[DEBUG_PATCH_H * DEBUG_PATCH_W];

        if (is_odd)
            teacher_int8_forward_odd(in_patch, out_c);
        else
            teacher_int8_forward_even(in_patch, out_c);

        float max_abs_diff = 0.0f;
        int elem_mismatch = 0;

        printf("\n=== Sample %d (is_odd=%d) ===\n", i, is_odd);

        for (int j = 0; j < DEBUG_PATCH_H * DEBUG_PATCH_W; ++j) {
            float diff = fabsf(out_c[j] - out_ref[j]);
            if (diff > max_abs_diff)
                max_abs_diff = diff;

            if (diff > 1e-4f) {
                elem_mismatch++;
                printf("  idx %2d: C = %.9f, Ref = %.9f, |diff| = %.9f  <-- MISMATCH\n",
                       j, out_c[j], out_ref[j], diff);
            } else {
                printf("  idx %2d: C = %.9f, Ref = %.9f, |diff| = %.9f\n",
                       j, out_c[j], out_ref[j], diff);
            }
        }

        if (elem_mismatch > 0) {
            total_mismatch++;
            printf("[SAMPLE %d] MISMATCH: %d / %d elements differ (max |diff| = %.9f)\n",
                   i, elem_mismatch, DEBUG_PATCH_H * DEBUG_PATCH_W, max_abs_diff);
        } else {
            printf("[SAMPLE %d] OK (max |diff| = %.9f)\n", i, max_abs_diff);
        }
    }

    printf("\n=============================\n");
    if (total_mismatch == 0)
        printf("[RESULT] All %d samples match exactly within tolerance.\n", DEBUG_NUM_SAMPLES);
    else
        printf("[RESULT] %d / %d samples have mismatches.\n",
               total_mismatch, DEBUG_NUM_SAMPLES);

    return 0;
}