#include "src/inference_quantized.h" // Include the inference API
#include "src/model_meta.h"      // Include for shape size definitions
#include "include/config.h"      // Include for LOG_* macros (mainly for output)
#include <stdio.h>               // For printf
#include <string.h>              // For memset
#include <math.h>                // For floor in quantization example

// --- Example: Quantization Function (Float to uint8) ---
// This demonstrates how you would typically quantize your float inputs
// before passing them to inference_run_quantized.
act_type_t quantize_f32_to_u8(float val, float scale, act_type_t zero_point) {
    if (scale <= 0.0f) {
        // Handle invalid scale (return zero point is a safe default)
        return zero_point;
    }
    // Calculate scaled value and add zero point
    float scaled_val = val / scale + (float)zero_point;
    // Apply rounding (round-half-up: floor(x + 0.5))
    float rounded_val = floorf(scaled_val + 0.5f);
    // Clamp to uint8 range [0, 255]
    if (rounded_val > 255.0f) rounded_val = 255.0f;
    if (rounded_val < 0.0f) rounded_val = 0.0f;
    return (act_type_t)rounded_val;
}


// --- Example Input Data ---
// In a real application, you would get these float values from sensors or data processing.
// For this example, let's use some dummy float values.
const float g_example_input_x_float[MODEL_INPUT_X_SHAPE_SIZE] = {
    // Fill with example float data for X (size 42)
    // For simplicity, let's make it correspond roughly to the zero point
     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
     0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};
const float g_example_input_meta_float[MODEL_INPUT_META_SHAPE_SIZE] = {
    // Fill with example float data for Meta (size 2)
    0.5f, 0.5f // Example: representing center coordinates (normalized)
};

// --- Buffers for Quantized Inputs and Outputs ---
static act_type_t g_quantized_x[MODEL_INPUT_X_SHAPE_SIZE];
static act_type_t g_quantized_meta[MODEL_INPUT_META_SHAPE_SIZE];
static act_type_t g_quantized_output[MODEL_OUTPUT_SHAPE_SIZE];
static float g_final_float_output[MODEL_OUTPUT_SHAPE_SIZE];


int main(void) {

    printf("--- Quantized Inference Example ---\n");

    // 1. Initialize the inference library
    printf("Initializing inference engine...\n");
    if (inference_init() != 0) {
        printf("[ERROR] inference_init failed!\n");
        return 1;
    }
    printf("Initialization successful.\n");

    // 2. Prepare Quantized Inputs
    printf("Quantizing example inputs...\n");

    // Quantize input X using its specific scale and zero point
    for (int i = 0; i < MODEL_INPUT_X_SHAPE_SIZE; ++i) {
        g_quantized_x[i] = quantize_f32_to_u8(
            g_example_input_x_float[i],
            MODEL_INPUT_X_SCALE,
            MODEL_INPUT_X_ZERO_POINT
        );
    }
    printf("  Input X quantized (first value: %u)\n", g_quantized_x[0]);


    // IMPORTANT: Quantize input Meta using the GAP output scale and zero point
    // This is required for the Cat layer to function correctly in the C code.
    for (int i = 0; i < MODEL_INPUT_META_SHAPE_SIZE; ++i) {
        g_quantized_meta[i] = quantize_f32_to_u8(
            g_example_input_meta_float[i],
            MODEL_GAP_OUT_SCALE,        // Use GAP scale
            MODEL_GAP_OUT_ZERO_POINT    // Use GAP zero point
        );
    }
    printf("  Input Meta quantized (using GAP scale/zp): [%u, %u]\n",
           g_quantized_meta[0], g_quantized_meta[1]);


    // 3. Run Quantized Inference
    printf("Running quantized inference...\n");
    int status = inference_run_quantized(
        g_quantized_x,
        g_quantized_meta,
        g_quantized_output
    );

    if (status != 0) {
        printf("[ERROR] inference_run_quantized failed with status %d\n", status);
        return 2;
    }
    printf("Inference successful.\n");

    // 4. (Optional) Dequantize and Post-process the Output
    printf("Dequantizing and post-processing output...\n");
    inference_dequantize_postprocess_output(
        g_quantized_output,
        g_final_float_output
    );

    // 5. Display Results
    printf("\n--- Inference Results ---\n");
    printf("Quantized Output (out_q): [ %u, %u ]\n",
           (unsigned)g_quantized_output[0],
           (unsigned)g_quantized_output[1]);

    printf("Final Float Output (dx, dy): [ %.6f, %.6f ]\n",
           g_final_float_output[0],
           g_final_float_output[1]);

    return 0; // Indicate success
}
