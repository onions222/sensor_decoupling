#include "test_helpers.h"
#include "../include/config.h" // Include for LOG_* macros
#include <stdio.h>
#include <stdlib.h> // For abs
#include <math.h>   // for fabsf
#include <string.h> // For memcmp (optional, less informative)
#include <stdbool.h>

int load_bin_f32(const char* path, float* buffer, size_t num_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) { LOG_ERROR("Failed to open file: %s", path); return -1; }
    size_t read_count = fread(buffer, sizeof(float), num_elements, f);
    fclose(f);
    if (read_count != num_elements) { LOG_ERROR("File read error: %s. Expected %zu floats, got %zu", path, num_elements, read_count); return -1; }
    LOG_INFO("Loaded %zu floats from %s", read_count, path);
    return 0;
}

// --- Load uint8 ---
int load_bin_u8(const char* path, uint8_t* buffer, size_t num_elements) {
    FILE* f = fopen(path, "rb");
    if (!f) { LOG_ERROR("Failed to open file: %s", path); return -1; }
    size_t read_count = fread(buffer, sizeof(uint8_t), num_elements, f);
    fclose(f);
    if (read_count != num_elements) { LOG_ERROR("File read error: %s. Expected %zu uint8_t, got %zu", path, num_elements, read_count); return -1; }
    // LOG_INFO("Loaded %zu uint8_t from %s", read_count, path); // Less verbose
    return 0;
}

// --- Save uint8 ---
int save_bin_u8(const char* path, const uint8_t* buffer, size_t num_elements) {
     FILE* f = fopen(path, "wb");
    if (!f) { LOG_ERROR("Failed to open file for writing: %s", path); return -1; }
    size_t write_count = fwrite(buffer, sizeof(uint8_t), num_elements, f);
    fclose(f);
    if (write_count != num_elements) { LOG_ERROR("File write error: %s. Expected %zu uint8_t, wrote %zu", path, num_elements, write_count); return -1; }
    LOG_DEBUG("Saved %zu uint8_t to %s", write_count, path);
    return 0;
}


int compare_results_f32(
    const float* c_out,
    const float* golden_out,
    size_t num_elements,
    float abs_tol,
    float rel_tol
) {
    int failures = 0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0; // Track max relative diff where applicable

    for (size_t i = 0; i < num_elements; ++i) {
        double abs_diff = fabsf(c_out[i] - golden_out[i]);
        double golden_abs = fabsf(golden_out[i]);
        double rel_diff = 0.0; // Initialize relative difference
        bool check_failed = false;

        // --- 修正: Handle near-zero golden values ---
        // Only calculate relative difference if golden value is significantly non-zero
        if (golden_abs > 1e-8f) { // Use a small threshold instead of 1e-9f divisor
            rel_diff = abs_diff / golden_abs;
            // Fail if BOTH tolerances are exceeded
            if (abs_diff > abs_tol && rel_diff > rel_tol) {
                check_failed = true;
            }
        } else {
            // If golden value is near zero, rely only on absolute tolerance
            if (abs_diff > abs_tol) {
                check_failed = true;
                rel_diff = NAN; // Indicate relative diff is not applicable
            }
        }

        // Track maximum differences
        max_abs_diff = (abs_diff > max_abs_diff) ? abs_diff : max_abs_diff;
        if (!isnan(rel_diff) && rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
        }

        if (check_failed) {
            if (failures < 10) { // Limit error prints
                LOG_ERROR("Verification FAILED (float) at index %zu:", i);
                LOG_ERROR("  C-Out:  %.6f", c_out[i]);
                LOG_ERROR("  Golden: %.6f", golden_out[i]);
                LOG_ERROR("  AbsDiff: %.6f (> Tol: %.6f)", abs_diff, abs_tol);
                if (!isnan(rel_diff)) {
                    LOG_ERROR("  RelDiff: %.6f (> Tol: %.6f)", rel_diff, rel_tol);
                } else {
                    LOG_ERROR("  RelDiff: N/A (Golden near zero)");
                }
            }
            failures++;
        } else {
             // Only log passing elements at higher log levels
             LOG_TRACE("Index %zu: C-Out=%.6f, Golden=%.6f (PASS)", i, c_out[i], golden_out[i]);
        }
    }
    if (failures > 0) {
        LOG_WARN("Final Float Comparison: %d mismatches. Max AbsDiff: %.6f, Max RelDiff: %.6f (where applicable)", failures, max_abs_diff, max_rel_diff);
    } else {
        LOG_INFO("Final Float Comparison: PASSED. Max AbsDiff: %.6f, Max RelDiff: %.6f (where applicable)", max_abs_diff, max_rel_diff);
    }
    return failures;
}

// --- Compare uint8 ---
int compare_intermediate_u8(
    const uint8_t* c_out,
    const uint8_t* golden_out,
    size_t num_elements,
    uint8_t abs_tol_lsb,
    const char* layer_name
) {
    // ... (implementation remains the same) ...
    int failures = 0;
    uint32_t max_diff = 0;

    for (size_t i = 0; i < num_elements; ++i) {
        uint32_t diff = (c_out[i] > golden_out[i]) ?
                        ((uint32_t)c_out[i] - (uint32_t)golden_out[i]) :
                        ((uint32_t)golden_out[i] - (uint32_t)c_out[i]);

        if (diff > max_diff) { max_diff = diff; }

        if (diff > abs_tol_lsb) {
            if (failures < 10) {
                LOG_ERROR("Intermediate MISMATCH [%s] at index %zu:", layer_name, i);
                LOG_ERROR("  C-Out:  %" PRIu8 " (0x%02X)", c_out[i], c_out[i]);
                LOG_ERROR("  Golden: %" PRIu8 " (0x%02X)", golden_out[i], golden_out[i]);
                LOG_ERROR("  AbsDiff: %" PRIu32 " (> Tol: %" PRIu8 ")", diff, abs_tol_lsb);
            }
            failures++;
        }
    }

    if (failures > 0) {
        LOG_WARN("Intermediate Comparison [%s]: %d mismatches. Max AbsDiff: %" PRIu32 " (Tol: %" PRIu8 ")",
                 layer_name, failures, max_diff, abs_tol_lsb);
    } else {
        LOG_INFO("Intermediate Comparison [%s]: PASSED. Max AbsDiff: %" PRIu32 " (Tol: %" PRIu8 ")",
                 layer_name, max_diff, abs_tol_lsb);
    }
    return failures;
}

