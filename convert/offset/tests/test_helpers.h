#ifndef TEST_HELPERS_H
#define TEST_HELPERS_H

#include <stddef.h>
#include <stdint.h> // For int8_t, uint8_t

/** Load float array from .bin */
int load_bin_f32(const char* path, float* buffer, size_t num_elements);

// --- 修正: Load/Save/Compare uint8 ---
/** Load uint8 array from .bin */
int load_bin_u8(const char* path, uint8_t* buffer, size_t num_elements);

/** Save uint8 array to .bin */
int save_bin_u8(const char* path, const uint8_t* buffer, size_t num_elements);

/** Compare two float arrays (final output) */
int compare_results_f32(
    const float* c_out,
    const float* golden_out,
    size_t num_elements,
    float abs_tol,
    float rel_tol
);

/** Compare two uint8 arrays (intermediate layers) */
int compare_intermediate_u8( // Renamed
    const uint8_t* c_out,
    const uint8_t* golden_out,
    size_t num_elements,
    uint8_t abs_tol_lsb, // uint8 tolerance
    const char* layer_name // For logging
);

#endif

