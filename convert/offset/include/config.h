#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h> // Include standard types

// --- Logging ---
// 0: None, 1: Error, 2: Warn, 3: Info (Verification), 4: Debug (Layer Ops), 5: Trace (Per-element values)
#define INFERENCE_C_LOG_LEVEL 4 // Default log level
#define MODEL_ARENA_SIZE 192
// --- Intermediate Debugging ---
// Define this (e.g., via Makefile -DDUMP_INTERMEDIATES) to save C layer outputs
// #define DUMP_INTERMEDIATES
#define INTERMEDIATE_DUMP_DIR "c_intermediate_outputs" // Directory to save dumps

// --- Verification ---
// Final float output comparison tolerance
#define INFERENCE_C_VERIFY_TOLERANCE_ABS_FLOAT 8e-3f
#define INFERENCE_C_VERIFY_TOLERANCE_REL_FLOAT 0.09f // 2%
// Intermediate uint8 output comparison tolerance
#define INFERENCE_C_VERIFY_TOLERANCE_ABS_UINT8 3 // Allow 1 LSB difference

// --- Numerics ---
// Accumulator type (strictly int32_t)
typedef int32_t acc_type_t;
// Activation data type
typedef uint8_t act_type_t;
// Weight data type
typedef int8_t weight_type_t;

// Enable saturated arithmetic
#define INFERENCE_C_USE_SATURATED_ARITHMETIC 1

// --- Platform ---
#include <stdio.h>
#include <inttypes.h> // For PRIu8, PRId32 etc.
#include <math.h>     // For isnan, isinf

#define LOG_ERROR(format, ...)   if(INFERENCE_C_LOG_LEVEL >= 1) { printf("[ERROR] " format "\n", ##__VA_ARGS__); }
#define LOG_WARN(format, ...)    if(INFERENCE_C_LOG_LEVEL >= 2) { printf("[WARN] " format "\n", ##__VA_ARGS__); }
#define LOG_INFO(format, ...)    if(INFERENCE_C_LOG_LEVEL >= 3) { printf("[INFO] " format "\n", ##__VA_ARGS__); }
#define LOG_DEBUG(format, ...)   if(INFERENCE_C_LOG_LEVEL >= 4) { printf("[DEBUG] " format "\n", ##__VA_ARGS__); }
#define LOG_TRACE(format, ...)   if(INFERENCE_C_LOG_LEVEL >= 5) { printf("[TRACE] " format "\n", ##__VA_ARGS__); }


// Assertion
#include <assert.h>
#define INFERENCE_C_ASSERT(cond) assert(cond)

// --- Saturation Helpers (Updated for uint8) ---
#if INFERENCE_C_USE_SATURATED_ARITHMETIC
    static inline acc_type_t saturate_add_s32(acc_type_t a, acc_type_t b) {
        #if defined(__GNUC__) || defined(__clang__)
            acc_type_t res;
            if (__builtin_add_overflow(a, b, &res)) { return (a > 0) ? INT32_MAX : INT32_MIN; }
            return res;
        #else
            int64_t temp = (int64_t)a + (int64_t)b;
            if (temp > INT32_MAX) return INT32_MAX; if (temp < INT32_MIN) return INT32_MIN;
            return (acc_type_t)temp;
        #endif
    }
    static inline act_type_t saturate_s32_to_u8(acc_type_t val) {
    if (val > UINT8_MAX) { 
        return UINT8_MAX; 
    }
    if (val < 0) { 
        return 0; 
    }
    return (act_type_t)val;
}
#else // No saturation
    #define saturate_add_s32(a, b) ((acc_type_t)(a) + (acc_type_t)(b))
    #define saturate_s32_to_u8(val) ((act_type_t)(val > UINT8_MAX ? UINT8_MAX : (val < 0 ? 0 : val)))
#endif

#endif // CONFIG_H