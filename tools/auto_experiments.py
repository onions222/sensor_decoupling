#!/usr/bin/env python3
import subprocess
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
INFER_DIR = ROOT / 'infer'
VALIDATE = INFER_DIR / 'validate_c_inference.py'
COMPARE = INFER_DIR / 'tools' / 'compare_jsons.py'

variants = [
    ('default','-DDEBUG_INT8_CHECK'),
    ('ties_to_even','-DDEBUG_INT8_CHECK -DTIES_TO_EVEN'),
    ('use_fp_requant','-DDEBUG_INT8_CHECK -DUSE_FP_REQUANT'),
    ('force_float_requant','-DDEBUG_INT8_CHECK -DFORCE_FLOAT_REQUANT'),
    ('force_full_float','-DDEBUG_INT8_CHECK -DFORCE_FULL_FLOAT_MLP'),
]

results = []

for name, cflags in variants:
    print(f'\n=== Variant: {name} CFLAGS="{cflags}" ===')
    # build
    cmd = ['make','clean']
    subprocess.run(cmd, cwd=str(INFER_DIR), check=True)
    cmd = ['make', f"CFLAGS={cflags} -Wall -Wextra -O2 -I."]
    subprocess.run(cmd, cwd=str(INFER_DIR), check=True)
    # run validation (limit 1) and produce dumps
    c_exe = INFER_DIR / 'inference_app'
    proc = subprocess.run(['python3', str(VALIDATE), '--limit','1','--dump-intermediates','--c_exe', str(c_exe)], cwd=str(INFER_DIR), capture_output=True, text=True)
    print(proc.stdout)
    # compare produced files
    pyfile = INFER_DIR / 'intermediates' / 'intermediates_aligned_g26.json_id0_py.json'
    cfile = INFER_DIR / 'intermediates' / 'intermediates_aligned_g26.json_id0_c.json'
    if pyfile.exists() and cfile.exists():
        cmp = subprocess.run(['python3', str(COMPARE), str(pyfile), str(cfile)], capture_output=True, text=True)
        print(cmp.stdout)
        # capture final diff line
        # naive parse: look for "Final output comparison" block
        out = cmp.stdout
        results.append((name, out))
    else:
        print('Missing intermediates for comparison')

# Summarize
print('\n=== Summary ===')
for name, out in results:
    # find line starting with 'final:' or 'Final output comparison' and next lines
    print('---', name)
    for L in out.splitlines():
        if 'final:' in L or 'Final output comparison' in L or L.strip().startswith('  python final'):
            print(L)
    print()

