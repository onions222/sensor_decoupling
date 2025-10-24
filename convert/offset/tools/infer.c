// Simple inference CLI supporting float and quantized modes
#include "../src/inference.h"
#include "../include/config.h"
#include "../src/model_meta.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int load_bin_f32(const char* path, float* buf, size_t n)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open %s\n", path); return -1; }
    size_t r = fread(buf, sizeof(float), n, f);
    fclose(f);
    if (r != n) { fprintf(stderr, "[ERROR] File size mismatch for %s (read %zu, expect %zu)\n", path, r, n); return -1; }
    return 0;
}

static int load_bin_u8(const char* path, uint8_t* buf, size_t n)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open %s\n", path); return -1; }
    size_t r = fread(buf, sizeof(uint8_t), n, f);
    fclose(f);
    if (r != n) { fprintf(stderr, "[ERROR] File size mismatch for %s (read %zu, expect %zu)\n", path, r, n); return -1; }
    return 0;
}

static int save_bin_f32(const char* path, const float* buf, size_t n)
{
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open %s for write\n", path); return -1; }
    size_t w = fwrite(buf, sizeof(float), n, f);
    fclose(f);
    if (w != n) { fprintf(stderr, "[ERROR] Write size mismatch for %s (wrote %zu, expect %zu)\n", path, w, n); return -1; }
    return 0;
}

static int save_bin_u8(const char* path, const uint8_t* buf, size_t n)
{
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "[ERROR] Cannot open %s for write\n", path); return -1; }
    size_t w = fwrite(buf, sizeof(uint8_t), n, f);
    fclose(f);
    if (w != n) { fprintf(stderr, "[ERROR] Write size mismatch for %s (wrote %zu, expect %zu)\n", path, w, n); return -1; }
    return 0;
}

static void usage(const char* prog)
{
    fprintf(stderr,
            "Usage: %s --x path/to/input_x.bin --meta path/to/input_meta.bin [--out out_f32.bin] [--quantized [--out-q out_u8.bin]]\n"
            "  Float mode (default):\n"
            "    - input: float32 x/meta of sizes %d / %d\n"
            "    - output: prints dx,dy; optionally saves float32 to --out\n"
            "  Quantized mode (--quantized):\n"
            "    - input: uint8 x/meta of sizes %d / %d\n"
            "    - output: prints uint8 and dequantized dx,dy; optionally saves uint8 to --out-q and float32 to --out\n",
            prog,
            (int)MODEL_INPUT_X_SHAPE_SIZE, (int)MODEL_INPUT_META_SHAPE_SIZE,
            (int)MODEL_INPUT_X_SHAPE_SIZE, (int)MODEL_INPUT_META_SHAPE_SIZE);
}

int main(int argc, char** argv)
{
    const char* x_path = NULL;
    const char* meta_path = NULL;
    const char* out_path = NULL;   // float32 output
    const char* outq_path = NULL;  // uint8 output (quantized mode)
    int quantized_mode = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--x") == 0 && i + 1 < argc) { x_path = argv[++i]; }
        else if (strcmp(argv[i], "--meta") == 0 && i + 1 < argc) { meta_path = argv[++i]; }
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) { out_path = argv[++i]; }
        else if (strcmp(argv[i], "--out-q") == 0 && i + 1 < argc) { outq_path = argv[++i]; }
        else if (strcmp(argv[i], "--quantized") == 0) { quantized_mode = 1; }
        else { usage(argv[0]); return 1; }
    }

    if (!x_path || !meta_path) { usage(argv[0]); return 1; }

    if (inference_init() != 0) {
        fprintf(stderr, "[ERROR] inference_init failed\n");
        return 3;
    }

    if (!quantized_mode) {
        // Float mode
        float x_f[MODEL_INPUT_X_SHAPE_SIZE];
        float meta_f[MODEL_INPUT_META_SHAPE_SIZE];
        float out_f[MODEL_OUTPUT_SHAPE_SIZE];

        if (load_bin_f32(x_path, x_f, MODEL_INPUT_X_SHAPE_SIZE) != 0) return 2;
        if (load_bin_f32(meta_path, meta_f, MODEL_INPUT_META_SHAPE_SIZE) != 0) return 2;

        if (inference_run_float(x_f, meta_f, out_f) != 0) {
            fprintf(stderr, "[ERROR] inference_run_float failed\n");
            return 4;
        }
        printf("Inference OK. Output (dx, dy) = [%f, %f]\n", out_f[0], out_f[1]);
        if (out_path) {
            if (save_bin_f32(out_path, out_f, MODEL_OUTPUT_SHAPE_SIZE) != 0) return 5;
            printf("Saved output to %s\n", out_path);
        }
    } else {
        // Quantized mode
        uint8_t x_q[MODEL_INPUT_X_SHAPE_SIZE];
        uint8_t meta_q[MODEL_INPUT_META_SHAPE_SIZE];
        uint8_t out_q[MODEL_OUTPUT_SHAPE_SIZE];
        float out_f[MODEL_OUTPUT_SHAPE_SIZE];

        if (load_bin_u8(x_path, x_q, MODEL_INPUT_X_SHAPE_SIZE) != 0) return 2;
        if (load_bin_u8(meta_path, meta_q, MODEL_INPUT_META_SHAPE_SIZE) != 0) return 2;

        if (inference_run_quantized(x_q, meta_q, out_q) != 0) {
            fprintf(stderr, "[ERROR] inference_run_quantized failed\n");
            return 4;
        }
        // Also produce dequantized + postprocessed float
        model_dequantize_and_postprocess(out_q, out_f);

        printf("Inference OK (quantized). Output uint8 = [%u, %u]; dequant(dx,dy) = [%f, %f]\n",
               (unsigned)out_q[0], (unsigned)out_q[1], out_f[0], out_f[1]);
        if (outq_path) {
            if (save_bin_u8(outq_path, out_q, MODEL_OUTPUT_SHAPE_SIZE) != 0) return 6;
            printf("Saved uint8 output to %s\n", outq_path);
        }
        if (out_path) {
            if (save_bin_f32(out_path, out_f, MODEL_OUTPUT_SHAPE_SIZE) != 0) return 5;
            printf("Saved float output to %s\n", out_path);
        }
    }
    return 0;
}
