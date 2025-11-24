#!/usr/bin/env python3
"""
Reference NumPy implementation of the C inference and comparison harness.
Runs the C binary `infer_demo` and a NumPy reimplementation of the conv pipeline
(using arrays in `weights.c` and `input_sample.c`), then compares results.
"""
import os
import re
import subprocess
import numpy as np

WEIGHTS_H = 'weights.h'
WEIGHTS_C = 'weights.c'
INPUT_C = 'input_sample.c'


def parse_macro(header_path, macro_name):
    pat = re.compile(rf"#define\s+{macro_name}\s+([\-0-9\.eE+f]+)")
    with open(header_path, 'r') as f:
        for line in f:
            m = pat.search(line)
            if m:
                val = m.group(1).rstrip('f')
                return float(val) if ('.' in val or 'e' in val.lower()) else int(val)
    raise RuntimeError(f"Macro {macro_name} not found in {header_path}")


def parse_c_array(filename, array_name):
    with open(filename, 'r') as f:
        txt = f.read()
    pat = re.compile(rf"{array_name}\s*\[.*?\]\s*=\s*\{{(.*?)\}}", re.S)
    m = pat.search(txt)
    if not m:
        raise RuntimeError(f"Array {array_name} not found in {filename}")
    body = m.group(1)
    nums = re.findall(r"-?\d+", body)
    return [int(x) for x in nums]


def parse_float_array(filename, array_name):
    with open(filename, 'r') as f:
        txt = f.read()
    pat = re.compile(rf"{array_name}\s*\[.*?\]\s*=\s*\{{(.*?)\}}", re.S)
    m = pat.search(txt)
    if not m:
        raise RuntimeError(f"Array {array_name} not found in {filename}")
    body = m.group(1)
    nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", body)
    return [float(x) for x in nums]


def mult_shift(x, multiplier, shift):
    # emulate multiply_by_quantized_multiplier from inference.h
    total = x.astype(np.int64) * np.int64(multiplier)
    val = total + (1 << 30)
    result = (val >> 31).astype(np.int64)
    if shift > 0:
        mask = 1 << (shift - 1)
        result = (result + mask) >> shift
    elif shift < 0:
        result = result << (-shift)
    return result.astype(np.int32)


def conv2d_numpy(input_q, weights, bias, in_ch, out_ch, k_h, k_w, pad, stride, multiplier, shift, input_zp, out_zp, out_min=-128, out_max=127):
    # input_q: numpy array shape (in_ch, H, W) dtype int32 or int8
    in_c = in_ch
    in_h = input_q.shape[1]
    in_w = input_q.shape[2]
    out_h = in_h
    out_w = in_w
    out = np.zeros((out_ch, out_h, out_w), dtype=np.int32)
    for oc in range(out_ch):
        for oy in range(out_h):
            for ox in range(out_w):
                acc = int(bias[oc])
                in_y_origin = oy * stride - pad
                in_x_origin = ox * stride - pad
                for ic in range(in_c):
                    for ky in range(k_h):
                        for kx in range(k_w):
                            iy = in_y_origin + ky
                            ix = in_x_origin + kx
                            if 0 <= iy < in_h and 0 <= ix < in_w:
                                input_val = int(input_q[ic, iy, ix])
                                w_idx = ((oc * in_c + ic) * k_h + ky) * k_w + kx
                                w_val = int(weights[w_idx])
                                adj = input_val - input_zp
                                acc += adj * w_val
                    out_val = mult_shift(np.int64(acc), multiplier, shift)
                out_val = int(out_val) + int(out_zp)
                if out_val < out_min:
                    out_val = out_min
                if out_val > out_max:
                    out_val = out_max
                out[oc, oy, ox] = out_val
    return out.astype(np.int8)


def run_numpy_pipeline():
    # read macros
    L0_IN_CH = int(parse_macro(WEIGHTS_H, 'L0_IN_CH'))
    L0_OUT_CH = int(parse_macro(WEIGHTS_H, 'L0_OUT_CH'))
    L0_K_H = int(parse_macro(WEIGHTS_H, 'L0_K_H'))
    L0_K_W = int(parse_macro(WEIGHTS_H, 'L0_K_W'))
    L0_PAD = int(parse_macro(WEIGHTS_H, 'L0_PAD'))
    L0_MULT = int(parse_macro(WEIGHTS_H, 'L0_MULT'))
    L0_SHIFT = int(parse_macro(WEIGHTS_H, 'L0_SHIFT'))
    L0_OUT_ZP = int(parse_macro(WEIGHTS_H, 'L0_OUT_ZP'))

    L1_IN_CH = int(parse_macro(WEIGHTS_H, 'L1_IN_CH'))
    L1_OUT_CH = int(parse_macro(WEIGHTS_H, 'L1_OUT_CH'))
    L1_K_H = int(parse_macro(WEIGHTS_H, 'L1_K_H'))
    L1_K_W = int(parse_macro(WEIGHTS_H, 'L1_K_W'))
    L1_PAD = int(parse_macro(WEIGHTS_H, 'L1_PAD'))
    L1_MULT = int(parse_macro(WEIGHTS_H, 'L1_MULT'))
    L1_SHIFT = int(parse_macro(WEIGHTS_H, 'L1_SHIFT'))
    L1_OUT_ZP = int(parse_macro(WEIGHTS_H, 'L1_OUT_ZP'))

    L2_IN_CH = int(parse_macro(WEIGHTS_H, 'L2_IN_CH'))
    L2_OUT_CH = int(parse_macro(WEIGHTS_H, 'L2_OUT_CH'))
    L2_K_H = int(parse_macro(WEIGHTS_H, 'L2_K_H'))
    L2_K_W = int(parse_macro(WEIGHTS_H, 'L2_K_W'))
    L2_PAD = int(parse_macro(WEIGHTS_H, 'L2_PAD'))
    L2_MULT = int(parse_macro(WEIGHTS_H, 'L2_MULT'))
    L2_SHIFT = int(parse_macro(WEIGHTS_H, 'L2_SHIFT'))
    L2_OUT_ZP = int(parse_macro(WEIGHTS_H, 'L2_OUT_ZP'))

    L3_IN_CH = int(parse_macro(WEIGHTS_H, 'L3_IN_CH'))
    L3_OUT_CH = int(parse_macro(WEIGHTS_H, 'L3_OUT_CH'))
    L3_K_H = int(parse_macro(WEIGHTS_H, 'L3_K_H'))
    L3_K_W = int(parse_macro(WEIGHTS_H, 'L3_K_W'))
    L3_PAD = int(parse_macro(WEIGHTS_H, 'L3_PAD'))
    L3_MULT = int(parse_macro(WEIGHTS_H, 'L3_MULT'))
    L3_SHIFT = int(parse_macro(WEIGHTS_H, 'L3_SHIFT'))
    L3_OUT_ZP = int(parse_macro(WEIGHTS_H, 'L3_OUT_ZP'))

    # input zp
    with open('input_sample.h', 'r') as f:
        txt = f.read()
    m = re.search(r"#define\s+INPUT_ZP\s+(-?\d+)", txt)
    INPUT_ZP = int(m.group(1)) if m else -128

    # parse arrays from weights.c and input_sample.c
    L0_W = parse_c_array(WEIGHTS_C, 'L0_WEIGHTS')
    L0_B = parse_c_array(WEIGHTS_C, 'L0_BIAS')
    L1_W = parse_c_array(WEIGHTS_C, 'L1_WEIGHTS')
    L1_B = parse_c_array(WEIGHTS_C, 'L1_BIAS')
    L2_W = parse_c_array(WEIGHTS_C, 'L2_WEIGHTS')
    L2_B = parse_c_array(WEIGHTS_C, 'L2_BIAS')
    L3_W = parse_c_array(WEIGHTS_C, 'L3_WEIGHTS')
    L3_B = parse_c_array(WEIGHTS_C, 'L3_BIAS')

    in_q = parse_c_array(INPUT_C, 'TEST_INPUT_INT8')
    # reshape input to (in_ch, H, W)
    inp = np.array(in_q, dtype=np.int32).reshape((1, 3, 5))

    out0 = conv2d_numpy(inp, L0_W, L0_B, L0_IN_CH, L0_OUT_CH, L0_K_H, L0_K_W, L0_PAD, 1, L0_MULT, L0_SHIFT, INPUT_ZP, L0_OUT_ZP)
    out1 = conv2d_numpy(out0.astype(np.int8), L1_W, L1_B, L1_IN_CH, L1_OUT_CH, L1_K_H, L1_K_W, L1_PAD, 1, L1_MULT, L1_SHIFT, L0_OUT_ZP, L1_OUT_ZP)
    out2 = conv2d_numpy(out1.astype(np.int8), L2_W, L2_B, L2_IN_CH, L2_OUT_CH, L2_K_H, L2_K_W, L2_PAD, 1, L2_MULT, L2_SHIFT, L1_OUT_ZP, L2_OUT_ZP)
    out3 = conv2d_numpy(out2.astype(np.int8), L3_W, L3_B, L3_IN_CH, L3_OUT_CH, L3_K_H, L3_K_W, L3_PAD, 1, L3_MULT, L3_SHIFT, L2_OUT_ZP, L3_OUT_ZP)

    L3_OUT_SCALE = float(re.search(r"#define\s+L3_OUT_SCALE\s+([0-9\.eE\-+f]+)", open(WEIGHTS_H).read()).group(1).rstrip('f'))
    py_deq = (out3.astype(np.int32).squeeze().astype(np.float32) - L3_OUT_ZP) * L3_OUT_SCALE
    return py_deq


def run_c_inference():
    if not os.path.exists('./infer_demo'):
        subprocess.check_call(['make', 'all'])
    p = subprocess.run(['./infer_demo'], capture_output=True, text=True)
    if p.returncode != 0:
        print('C inference failed:', p.stderr)
        raise RuntimeError('C inference failed')
    lines = p.stdout.splitlines()
    idx = 0
    for i, l in enumerate(lines):
        if '=== Inference Result' in l:
            idx = i + 1
            break
    mat = []
    for r in range(3):
        line = lines[idx + r].strip()
        vals = [float(x) for x in line.split()]
        mat.append(vals)
    return np.array(mat)


def main():
    py = run_numpy_pipeline()
    c = run_c_inference()
    print('Python dequantized output:')
    print(np.array2string(py, formatter={'float_kind': lambda x: f"{x:6.3f}"}))
    print('C dequantized output:')
    print(np.array2string(c, formatter={'float_kind': lambda x: f"{x:6.3f}"}))
    diff = np.abs(py - c)
    print('Max abs difference:', diff.max())
    tol = 1e-2
    if diff.max() <= tol:
        print('Outputs match within tolerance')
        return 0
    else:
        print('Outputs differ')
        return 2


if __name__ == '__main__':
    exit(main())
