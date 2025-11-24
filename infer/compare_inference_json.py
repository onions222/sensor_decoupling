#!/usr/bin/env python3
"""
Run comparison over all samples in a JSON file.
For each sample:
 - extract 3x5 patch like the exporter did
 - normalize and quantize using `INPUT_SCALE` / `INPUT_ZP`
 - write `input_sample.c` for that sample
 - rebuild `infer_demo` and run it
 - compute NumPy pipeline output and compare

Usage: python3 compare_inference_json.py /path/to/aligned_g1.json [--limit N]
"""
import sys
import os
import re
import json
import subprocess
import numpy as np
from time import time

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


def conv2d_numpy(input_q, weights, bias, in_ch, out_ch, k_h, k_w, pad, stride, multiplier, shift, input_zp, out_zp, out_min=-128, out_max=127):
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
                out_val = mult_shift_scalar(acc, multiplier, shift)
                out_val = int(out_val) + int(out_zp)
                if out_val < out_min:
                    out_val = out_min
                if out_val > out_max:
                    out_val = out_max
                out[oc, oy, ox] = out_val
    return out.astype(np.int8)


def write_input_c(float_patch, int8_flat, input_scale, input_zp, peak_r_m, peak_c_m_10, is_odd):
    # overwrite input_sample.c with the sample-specific arrays
    txt = []
    txt.append('/* Generated input sample for validation */')
    txt.append('#include <stdint.h>')
    txt.append('#include "input_sample.h"')
    txt.append('')
    txt.append('const float TEST_INPUT_FLOAT[3][5] = {')
    for r in range(3):
        row = float_patch[r]
        vals = ', '.join([f"{v:.6f}f" for v in row])
        txt.append('    { ' + vals + ' },')
    txt.append('};')
    txt.append('')
    txt.append('const int8_t TEST_INPUT_INT8[3 * 5] = {')
    # write as rows flattened
    flat = ', '.join([str(int(x)) for x in int8_flat])
    txt.append('    ' + flat + ',')
    txt.append('};')
    with open(INPUT_C, 'w') as f:
        f.write('\n'.join(txt) + '\n')
    # Also overwrite input_sample.h to include peak and odd info for this sample
    header = []
    header.append('/* Generated input sample header for validation */')
    header.append('#ifndef INPUT_SAMPLE_H')
    header.append('#define INPUT_SAMPLE_H')
    header.append('')
    header.append('// Sample Input Parameters')
    header.append(f'#define INPUT_SCALE {input_scale}f')
    header.append(f'#define INPUT_ZP {int(input_zp)}')
    header.append(f'#define INPUT_IS_ODD {int(is_odd)}')
    header.append('')
    header.append('// Peak info (to be filled per-sample)')
    header.append('// Use PEAK_R_M (row index in 0..31) and PEAK_C_M_10 (col index in 0..9)')
    header.append(f'#define PEAK_R_M {int(peak_r_m)}')
    header.append(f'#define PEAK_C_M_10 {int(peak_c_m_10)}')
    header.append('')
    header.append('// Normalized Float Patch (3x5)')
    header.append('extern const float TEST_INPUT_FLOAT[3][5];')
    header.append('')
    header.append('// Pre-quantized Int8 Input (for debugging)')
    header.append('extern const int8_t TEST_INPUT_INT8[3 * 5];')
    header.append('')
    header.append('#endif // INPUT_SAMPLE_H')
    with open(os.path.join(ROOT, 'input_sample.h'), 'w') as f:
        f.write('\n'.join(header) + '\n')


def run_c_and_parse():
    # rebuild and run
    subprocess.check_call(['make', 'all'])
    p = subprocess.run(['./infer_demo'], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError('C inference failed:\n' + p.stderr)
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


def extract_patch_from_matrix(mat):
    raw = np.array(mat, dtype=np.float32)
    peak_r, peak_c = np.unravel_index(np.argmax(raw), raw.shape)
    r_start = max(0, peak_r - 1)
    c_start = max(0, peak_c - 2)
    patch = raw[r_start:r_start+3, c_start:c_start+5]
    if patch.shape != (3,5):
        tmp = np.zeros((3,5), dtype=np.float32)
        tmp[:patch.shape[0], :patch.shape[1]] = patch
        patch = tmp
    s = patch.sum()
    if s > 1e-6:
        patch = patch / s
    return patch


def main():
    if len(sys.argv) < 2:
        print('Usage: compare_inference_json.py /path/to/aligned_g1.json [--limit N]')
        return 2
    json_path = sys.argv[1]
    limit = None
    if '--limit' in sys.argv:
        idx = sys.argv.index('--limit')
        limit = int(sys.argv[idx+1])

    with open(json_path, 'r') as f:
        data = json.load(f)
    keys = list(data.keys())
    if limit:
        keys = keys[:limit]

    # read params and weights arrays once
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

    # input params
    ih = open(os.path.join(ROOT, 'input_sample.h')).read()
    m = re.search(r"#define\s+INPUT_SCALE\s+([0-9\.eE\-+f]+)", ih)
    INPUT_SCALE = float(m.group(1).rstrip('f')) if m else 1.0/255.0
    m2 = re.search(r"#define\s+INPUT_ZP\s+(-?\d+)", ih)
    INPUT_ZP = int(m2.group(1)) if m2 else -128

    # parse weights arrays
    L0_W = parse_c_array(WEIGHTS_C, 'L0_WEIGHTS')
    L0_B = parse_c_array(WEIGHTS_C, 'L0_BIAS')
    L1_W = parse_c_array(WEIGHTS_C, 'L1_WEIGHTS')
    L1_B = parse_c_array(WEIGHTS_C, 'L1_BIAS')
    L2_W = parse_c_array(WEIGHTS_C, 'L2_WEIGHTS')
    L2_B = parse_c_array(WEIGHTS_C, 'L2_BIAS')
    L3_W = parse_c_array(WEIGHTS_C, 'L3_WEIGHTS')
    L3_B = parse_c_array(WEIGHTS_C, 'L3_BIAS')

    stats = []
    start = time()
    for i, k in enumerate(keys):
        entry = data[k]
        # extract merging 18-col matrix and compute peak (r, c10)
        merging_18 = np.array(entry['merging']['normalized_matrix'], dtype=np.float32)
        # compress 18->10 as in dataset
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

        effective_merging_10 = compress_to_10_col(merging_18)
        peak_r_m, peak_c_m_10 = np.unravel_index(np.argmax(effective_merging_10), effective_merging_10.shape)
        is_odd = float(peak_r_m % 2 == 1)

        mat = effective_merging_10
        patch = extract_patch_from_matrix(mat)
        # quantize
        q = np.round(patch / INPUT_SCALE) + INPUT_ZP
        q = np.clip(q, -128, 127).astype(np.int8)
        write_input_c(patch, q.flatten(), INPUT_SCALE, INPUT_ZP, peak_r_m, peak_c_m_10, is_odd)
        # run C
        try:
            c_out = run_c_and_parse()
        except Exception as e:
            print('C run failed for', k, 'error', e)
            stats.append((k, float('nan')))
            continue
        # run numpy pipeline
        inp = q.reshape((1,3,5)).astype(np.int32)
        out0 = conv2d_numpy(inp, L0_W, L0_B, L0_IN_CH, L0_OUT_CH, L0_K_H, L0_K_W, L0_PAD, 1, L0_MULT, L0_SHIFT, INPUT_ZP, L0_OUT_ZP)
        out1 = conv2d_numpy(out0.astype(np.int8), L1_W, L1_B, L1_IN_CH, L1_OUT_CH, L1_K_H, L1_K_W, L1_PAD, 1, L1_MULT, L1_SHIFT, L0_OUT_ZP, L1_OUT_ZP)
        out2 = conv2d_numpy(out1.astype(np.int8), L2_W, L2_B, L2_IN_CH, L2_OUT_CH, L2_K_H, L2_K_W, L2_PAD, 1, L2_MULT, L2_SHIFT, L1_OUT_ZP, L2_OUT_ZP)
        out3 = conv2d_numpy(out2.astype(np.int8), L3_W, L3_B, L3_IN_CH, L3_OUT_CH, L3_K_H, L3_K_W, L3_PAD, 1, L3_MULT, L3_SHIFT, L2_OUT_ZP, L3_OUT_ZP)
        L3_OUT_SCALE = float(re.search(r"#define\s+L3_OUT_SCALE\s+([0-9\.eE\-+f]+)", open(WEIGHTS_H).read()).group(1).rstrip('f'))
        py_deq = (out3.astype(np.int32).squeeze().astype(np.float32) - L3_OUT_ZP) * L3_OUT_SCALE
        c_deq = c_out
        diff = np.abs(py_deq - c_deq)
        maxd = float(diff.max())
        stats.append((k, maxd))
        # compute centroids for python and C dequantized outputs
        def local_com_from_patch(mat):
            eps = 1e-8
            m = np.maximum(mat, 0.0)
            mass = m.sum() + eps
            ys = np.arange(m.shape[0])[:, None]
            xs = np.arange(m.shape[1])[None, :]
            xw = (m * xs).sum() / mass
            yw = (m * ys).sum() / mass
            return xw, yw

        def global_from_patch(mat, peak_r_m, peak_c_m_10, is_odd_flag):
            local_x, local_y = local_com_from_patch(mat)
            pw_offset = 5 // 2
            ph_offset = 3 // 2
            global_x_10 = local_x + peak_c_m_10 - pw_offset
            global_y = local_y + peak_r_m - ph_offset
            x_clamped = min(max(global_x_10, 0.0), 9.0)
            x_floor = int(np.floor(x_clamped))
            x_ceil = int(np.ceil(x_clamped))
            frac = x_clamped - float(x_floor)
            odd_grid = np.array([0.5, 2.5, 4.5, 6.5, 8.0, 9.0, 10.5, 12.5, 14.5, 16.5], dtype=np.float32)
            even_grid = np.array([0.0, 1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.0], dtype=np.float32)
            grid = odd_grid if int(is_odd) else even_grid
            val_floor = float(grid[x_floor])
            val_ceil = float(grid[x_ceil])
            global_x_18 = val_floor + (val_ceil - val_floor) * frac
            return global_x_18, global_y

        py_cx, py_cy = global_from_patch(py_deq, peak_r_m, peak_c_m_10, is_odd)
        c_cx, c_cy = global_from_patch(c_deq, peak_r_m, peak_c_m_10, is_odd)
        cent_dist = float(np.hypot(py_cx - c_cx, py_cy - c_cy))
        print(f"{i+1}/{len(keys)} {k} centroid_py=({py_cx:.3f},{py_cy:.3f}) centroid_c=({c_cx:.3f},{c_cy:.3f}) dist={cent_dist:.6f} maxdiff={maxd:.6f}")
        if (i+1) % 10 == 0:
            print(f'Processed {i+1}/{len(keys)} samples (elapsed {time()-start:.1f}s)')
    # summary
    vals = [v for (_, v) in stats if not np.isnan(v)]
    if vals:
        print('Samples:', len(vals))
        print('Max diff overall:', max(vals))
        print('Median diff:', float(np.median(vals)))
        bad = sorted(stats, key=lambda x: -x[1])[:10]
        print('Top 10 worst samples:')
        for k, v in bad:
            print(k, v)
    else:
        print('No successful runs')
    return 0


if __name__ == '__main__':
    sys.exit(main())
