#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include <stddef.h> // For size_t
#include <stdbool.h>

// --- Configuration Constants (Must match Python script) ---
#define ROWS 32
#define COLS 18
#define LOCAL_PATCH_H 3
#define LOCAL_PATCH_W 6
#define K_SPARSE_POINTS 8 // K value

#define MLP_INPUT_SIZE (K_SPARSE_POINTS * 3 + 2) // 8*3 + 2 = 26
#define MLP_HIDDEN1_SIZE 32 // hidden * 2 (with width=8, hidden=16 -> 32)
#define MLP_HIDDEN2_SIZE 16 // hidden
#define MLP_OUTPUT_SIZE 2   // dx, dy

#define MAX_DX 4.0f
#define MAX_DY 0.25f // Corresponds to MAX_DY_ABS in Python
#define ALPHA_Y 0.0f // To disable Y correction

// --- Data Structures ---

// Structure to hold a sparse point's features
typedef struct {
    float value;     // Normalized value (after log1p, z-score)
    float norm_row;  // Normalized row within patch [0, 1]
    float norm_col;  // Normalized col within patch [0, 1]
} SparsePointFeature;

// Structure to hold inference results
typedef struct {
    float baseline_cog_x; // Calculated baseline CoG X (global coordinates)
    float baseline_cog_y; // Calculated baseline CoG Y (global coordinates)
    float predicted_dx;   // MLP predicted dx offset
    float predicted_dy;   // MLP predicted dy offset
    float corrected_cog_x; // Final corrected CoG X
    float corrected_cog_y; // Final corrected CoG Y
} InferenceResult;

// Structure to hold quantization parameters for a layer's activation or weight
typedef struct {
    float scale;
    int32_t zero_point;
    // Add fields for per-channel quantization if needed (e.g., float* scales, int32_t* zero_points)
    bool is_per_channel;
    float* scales;      // Array for per-channel scales
    int32_t* zero_points; // Array for per-channel zero_points
    int axis;           // Axis for per-channel quantization
} QParams;


// --- Function Prototypes ---

/**
 * @brief Finds the peak value's location in the sensor data.
 * @param sensor_data Input sensor data (ROWS x COLS).
 * @param peak_r Pointer to store the row index of the peak.
 * @param peak_c Pointer to store the column index of the peak.
 */
void find_peak(const float sensor_data[ROWS][COLS], int* peak_r, int* peak_c);

/**
 * @brief Crops a local window around the peak. Handles boundary clipping.
 * @param sensor_data Input sensor data (ROWS x COLS).
 * @param peak_r Row index of the peak.
 * @param peak_c Column index of the peak.
 * @param local_patch Output buffer for the cropped patch (LOCAL_PATCH_H x LOCAL_PATCH_W).
 * @param r0 Pointer to store the starting row index (global) of the patch.
 * @param c0 Pointer to store the starting column index (global) of the patch.
 * @return Actual height and width of the cropped patch (might be smaller at edges). Fills patch buffer.
 */
// NOTE: For simplicity, this example assumes the patch is always LOCAL_PATCH_H x LOCAL_PATCH_W.
// A more robust implementation would handle variable sizes if clipping occurs.
void crop_local_window_fixed(const float sensor_data[ROWS][COLS],
                             int peak_r, int peak_c,
                             float local_patch[LOCAL_PATCH_H][LOCAL_PATCH_W],
                             int* r0, int* c0);

/**
 * @brief Calculates the centroid (Center of Gravity) for a given 2D array.
 * @param arr Input array.
 * @param h Height of the array.
 * @param w Width of the array.
 * @param cog_x Pointer to store the calculated X centroid.
 * @param cog_y Pointer to store the calculated Y centroid.
 */
void centroid_xy_c(const float* arr, int h, int w, float* cog_x, float* cog_y);


/**
 * @brief Extracts and normalizes K sparse features from the local patch.
 * @param local_patch Input local patch (LOCAL_PATCH_H x LOCAL_PATCH_W).
 * @param sparse_features Output array to store the K sparse features. Size K_SPARSE_POINTS.
 * Features that are not found are zero-padded.
 */
void extract_sparse_features(const float local_patch[LOCAL_PATCH_H][LOCAL_PATCH_W],
                             SparsePointFeature sparse_features[K_SPARSE_POINTS]);


/**
 * @brief Performs INT8 inference using the quantized MLP model.
 * THIS IS THE MOST COMPLEX FUNCTION AND REQUIRES PLATFORM-SPECIFIC IMPLEMENTATION
 * of INT8 matrix multiplication and requantization.
 *
 * @param input_features Quantized input features (INT8).
 * @param output_offset Output buffer for the dequantized offsets (dx, dy) (float).
 */
void mlp_inference_int8(const int8_t input_features[MLP_INPUT_SIZE],
                        float output_offset[MLP_OUTPUT_SIZE]);

/**
 * @brief Top-level function to run the complete inference pipeline.
 * @param sensor_data Input raw sensor data (ROWS x COLS).
 * @param result Pointer to an InferenceResult structure to store the output.
 */
void run_inference(const float sensor_data[ROWS][COLS], InferenceResult* result);

// Small helpers exposed for main.c and for dumping intermediates
int8_t quantize_affine(float value, float scale, int32_t zero_point);
float dequantize_affine(int8_t value, float scale, int32_t zero_point);
void quantized_linear(const int8_t* input_q,
                      int input_size,
                      int output_size,
                      const int8_t* weight_q,
                      const float* bias_fp32,
                      const QParams* weight_params,
                      const QParams* bias_params,
                      const QParams* input_params,
                      const QParams* output_params,
                      int8_t* output_q,
                      const char* layer_name);


#endif // INFERENCE_H
