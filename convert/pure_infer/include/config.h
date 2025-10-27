#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h> // Include standard types

// --- Logging ---
// 0: None, 1: Error, 2: Warn, 3: Info
// Note: Set INFERENCE_C_LOG_LEVEL to 0 or 1 for production inference to minimize overhead.
#define INFERENCE_C_LOG_LEVEL 1 // Set to 1 (Errors only) for pure inference

// --- Numerics ---
// Accumulator type (strictly int32_t)
typedef int32_t acc_type_t;
// Activation data type
typedef uint8_t act_type_t;
// Weight data type
typedef int8_t weight_type_t;

// Enable saturated arithmetic (recommended for mimicking hardware behavior)
#define INFERENCE_C_USE_SATURATED_ARITHMETIC 1

// --- Platform ---
// Include necessary headers for standard types, math, and potentially logging
#include <stdio.h>    // Required if logging is enabled (level >= 1)
#include <inttypes.h> // For PRIu8, PRId32 etc. format specifiers in logs
#include <math.h>     // For isnan, isinf, floor, tanhf (used in core and postprocess)
#include <limits.h>   // For INT32_MAX, INT32_MIN, UINT8_MAX

// --- Logging Macros ---
// Define logging macros based on the log level
#if INFERENCE_C_LOG_LEVEL >= 1
    #define LOG_ERROR(format, ...)   { printf("[ERROR] " format "\n", ##__VA_ARGS__); }
#else
    #define LOG_ERROR(format, ...)
#endif
#if INFERENCE_C_LOG_LEVEL >= 2
    #define LOG_WARN(format, ...)    { printf("[WARN] " format "\n", ##__VA_ARGS__); }
#else
    #define LOG_WARN(format, ...)
#endif
#if INFERENCE_C_LOG_LEVEL >= 3
    #define LOG_INFO(format, ...)    { printf("[INFO] " format "\n", ##__VA_ARGS__); }
#else
    #define LOG_INFO(format, ...)
#endif
#if INFERENCE_C_LOG_LEVEL >= 4
    #define LOG_DEBUG(format, ...)   { printf("[DEBUG] " format "\n", ##__VA_ARGS__); }
#else
    #define LOG_DEBUG(format, ...)
#endif
#if INFERENCE_C_LOG_LEVEL >= 5
    #define LOG_TRACE(format, ...)   { printf("[TRACE] " format "\n", ##__VA_ARGS__); }
#else
    #define LOG_TRACE(format, ...)
#endif


// --- Assertion ---
// Define assertion macro (can be disabled for release builds if needed)
#include <assert.h>
#define INFERENCE_C_ASSERT(cond) assert(cond)

// --- Saturation Helpers (for uint8 activations and int32 accumulators) ---
#if INFERENCE_C_USE_SATURATED_ARITHMETIC
    // Saturating 32-bit integer addition
    static inline acc_type_t saturate_add_s32(acc_type_t a, acc_type_t b) {
        #if defined(__GNUC__) || defined(__clang__)
            // Use compiler built-ins if available (more efficient)
            acc_type_t res;
            if (__builtin_add_overflow(a, b, &res)) {
                // If overflow occurs, return max or min based on sign of 'a'
                return (a > 0) ? INT32_MAX : INT32_MIN;
            }
            return res;
        #else
            // Portable implementation using 64-bit intermediate
            int64_t temp = (int64_t)a + (int64_t)b;
            if (temp > INT32_MAX) return INT32_MAX;
            if (temp < INT32_MIN) return INT32_MIN;
            return (acc_type_t)temp;
        #endif
    }
    // Saturate a 32-bit signed integer to the uint8 range [0, 255]
    static inline act_type_t saturate_s32_to_u8(acc_type_t val) {
        if (val > UINT8_MAX) { // Check upper bound first
            return UINT8_MAX;
        }
        if (val < 0) { // Check lower bound
            return 0;
        }
        // If within range, cast directly
        return (act_type_t)val;
    }
#else // No saturation (Standard C arithmetic, may wrap around on overflow)
    #warning "Saturated arithmetic is disabled. Results may differ from hardware/TFLite on overflow."
    #define saturate_add_s32(a, b) ((acc_type_t)((int64_t)(a) + (int64_t)(b))) // Still use 64-bit intermediate to avoid immediate overflow
    // Clamp s32 to u8 range without specific saturation intrinsics
    #define saturate_s32_to_u8(val) ((act_type_t)(val > UINT8_MAX ? UINT8_MAX : (val < 0 ? 0 : val)))
#endif

#endif // CONFIG_H
