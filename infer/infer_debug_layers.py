#!/usr/bin/env python3
"""
Diagnostic: run NumPy conv pipeline for a single JSON sample and print per-layer internals.
Usage: python3 infer_debug_layers.py /path/to/json [sample_key]
"""
import sys, os, re, json
import numpy as np

ROOT = os.path.dirname(__file__)
WEIGHTS_H = os.path.join(ROOT, 'weights.h')
WEIGHTS_C = os.path.join(ROOT, 'weights.c')
INPUT_C = os.path.join(ROOT, 'input_sample.c')


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


def mult_shift_scalar(x, multiplier, shift):
    total = int(x) * int(multiplier)
    val = total + (1 << 30)
    res31 = val >> 31
    if shift > 0:
        mask = 1 << (shift - 1)
        tmp = res31 + mask
        after = tmp >> shift
    elif shift < 0:
        after = res31 << (-shift)
    else:
        after = res31
    return int(after)


def conv2d_numpy_debug(input_q, weights, bias, in_ch, out_ch, k_h, k_w, pad, stride, multiplier, shift, input_zp, out_zp):
    in_c = in_ch
    in_h = input_q.shape[1]
    in_w = input_q.shape[2]
    out_h = in_h
    out_w = in_w
    out = np.zeros((out_ch, out_h, out_w), dtype=np.int32)
    stats = {'acc_min': None, 'acc_max': None}
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
                # stats
                stats['acc_min'] = acc if stats['acc_min'] is None else min(stats['acc_min'], acc)
                stats['acc_max'] = acc if stats['acc_max'] is None else max(stats['acc_max'], acc)
                out_val = mult_shift_scalar(acc, multiplier, shift)
                out_val = int(out_val) + int(out_zp)
                # saturate
                if out_val < -128: out_val = -128
                if out_val > 127: out_val = 127
                out[oc, oy, ox] = out_val
    return out.astype(np.int8), stats


def compress_to_10_col(matrix_18_col):
    matrix_10_col = np.zeros((32, 10), dtype=np.float32)
    odd_pairs  = [(0,1),(2,3),(4,5),(6,7),(10,11),(12,13),(14,15),(16,17)]
    odd_single = [(8,4), (9,5)]
    odd_map_18_to_10 = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2, 6:3, 7:3, 8:4, 9:5, 10:6, 11:6, 12:7, 13:7, 14:8, 15:8, 16:9, 17:9}
    even_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)]
    even_single = [(0,0), (17,9)]
    even_map_18_to_10 = {0:0, 1:1, 2:1, 3:2, 4:2, 5:3, 6:3, 7:4, 8:4, 9:5, 10:5, 11:6, 12:6, 13:7, 14:7, 15:8, 16:8, 17:9}
    for r in range(32):
        is_odd = (r % 2 == 1)
        pairs = odd_pairs if is_odd else even_pairs
        singles = odd_single if is_odd else even_single
        map_18_to_10 = odd_map_18_to_10 if is_odd else even_map_18_to_10
        for (c_left, c_right) in pairs:
            col_10_idx = map_18_to_10[c_left]
            avg = (matrix_18_col[r, c_left] + matrix_18_col[r, c_right]) / 2.0
            matrix_10_col[r, col_10_idx] = avg
        for (c_18, col_10_idx) in singles:
            matrix_10_col[r, col_10_idx] = matrix_18_col[r, c_18]
    return matrix_10_col


def main():
    if len(sys.argv) < 2:
        print('Usage: infer_debug_layers.py /path/to/json [sample_key]')
        return 2
    path = sys.argv[1]
    with open(path, 'r') as f:
        data = json.load(f)
    keys = list(data.keys())
    key = sys.argv[2] if len(sys.argv) > 2 else keys[0]
    entry = data[key]
    merging_18 = np.array(entry['merging']['normalized_matrix'], dtype=np.float32)
    effective_merging_10 = compress_to_10_col(merging_18)
    peak_r_m, peak_c_m_10 = np.unravel_index(np.argmax(effective_merging_10), effective_merging_10.shape)
    is_odd = int(peak_r_m % 2 == 1)
    print('Sample key:', key, 'peak_r_m, peak_c_m_10, is_odd=', peak_r_m, peak_c_m_10, is_odd)

    # extract 3x5 patch (pad and normalize same as dataset loader)
    from math import floor
    pad2d = (2,2,1,1)
    # create padded 10-col
    padded = np.pad(effective_merging_10, ((0,0),(pad2d[0],pad2d[1])), mode='constant')
    r0 = peak_r_m; c0 = peak_c_m_10
    patch = padded[r0:r0+3, c0:c0+5]
    # normalize
    patch = np.maximum(patch, 0.0)
    s = patch.sum()
    if s > 1e-8:
        patch = patch / s

    # quantize input using input_sample.h macros
    # read input_sample.h to get INPUT_SCALE and INPUT_ZP
    ih = open(os.path.join(ROOT, 'input_sample.h')).read()
    m = re.search(r"#define\s+INPUT_SCALE\s+([0-9\.eE\-+f]+)", ih)
    INPUT_SCALE = float(m.group(1).rstrip('f')) if m else 1.0/255.0
    m2 = re.search(r"#define\s+INPUT_ZP\s+(-?\d+)", ih)
    INPUT_ZP = int(m2.group(1)) if m2 else -128
    print('INPUT_SCALE, INPUT_ZP=', INPUT_SCALE, INPUT_ZP)

    # prepare weights
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
    L3_OUT_SCALE = float(re.search(r"#define\s+L3_OUT_SCALE\s+([0-9\.eE\-+f]+)", open(WEIGHTS_H).read()).group(1).rstrip('f'))

    # parse arrays
    L0_W = parse_c_array(WEIGHTS_C, 'L0_WEIGHTS')
    L0_B = parse_c_array(WEIGHTS_C, 'L0_BIAS')
    L1_W = parse_c_array(WEIGHTS_C, 'L1_WEIGHTS')
    L1_B = parse_c_array(WEIGHTS_C, 'L1_BIAS')
    L2_W = parse_c_array(WEIGHTS_C, 'L2_WEIGHTS')
    L2_B = parse_c_array(WEIGHTS_C, 'L2_BIAS')
    L3_W = parse_c_array(WEIGHTS_C, 'L3_WEIGHTS')
    L3_B = parse_c_array(WEIGHTS_C, 'L3_BIAS')

    # quantize input
    q = np.round(patch / INPUT_SCALE) + INPUT_ZP
    q = np.clip(q, -128, 127).astype(np.int32)
    print('Input patch (float)')
    print(patch)
    print('Quantized input (int)')
    print(q.reshape((1,3,5)))

    # run layers with debug stats
    out0, s0 = conv2d_numpy_debug(q.reshape((1,3,5)), L0_W, L0_B, L0_IN_CH, L0_OUT_CH, L0_K_H, L0_K_W, L0_PAD, 1, L0_MULT, L0_SHIFT, INPUT_ZP, L0_OUT_ZP)
    print('\nLayer0 q out (shape) =', out0.shape, 'min/max =', out0.min(), out0.max(), 'acc min/max =', s0['acc_min'], s0['acc_max'])
    print(out0)

    out1, s1 = conv2d_numpy_debug(out0.astype(np.int32), L1_W, L1_B, L1_IN_CH, L1_OUT_CH, L1_K_H, L1_K_W, L1_PAD, 1, L1_MULT, L1_SHIFT, L0_OUT_ZP, L1_OUT_ZP)
    print('\nLayer1 q out (shape) =', out1.shape, 'min/max =', out1.min(), out1.max(), 'acc min/max =', s1['acc_min'], s1['acc_max'])
    print(out1)

    out2, s2 = conv2d_numpy_debug(out1.astype(np.int32), L2_W, L2_B, L2_IN_CH, L2_OUT_CH, L2_K_H, L2_K_W, L2_PAD, 1, L2_MULT, L2_SHIFT, L1_OUT_ZP, L2_OUT_ZP)
    print('\nLayer2 q out (shape) =', out2.shape, 'min/max =', out2.min(), out2.max(), 'acc min/max =', s2['acc_min'], s2['acc_max'])
    print(out2)

    out3, s3 = conv2d_numpy_debug(out2.astype(np.int32), L3_W, L3_B, L3_IN_CH, L3_OUT_CH, L3_K_H, L3_K_W, L3_PAD, 1, L3_MULT, L3_SHIFT, L2_OUT_ZP, L3_OUT_ZP)
    print('\nLayer3 q out (shape) =', out3.shape, 'min/max =', out3.min(), out3.max(), 'acc min/max =', s3['acc_min'], s3['acc_max'])
    print(out3)

    print('\nLayer3 dequantized (float):')
    print((out3.astype(np.int32) - L3_OUT_ZP) * L3_OUT_SCALE)

    return 0

if __name__ == '__main__':
    sys.exit(main())
