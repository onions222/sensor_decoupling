#!/usr/bin/env python3
"""
Exhaustive auto experiment runner for integer rounding variants.

This script compiles the C infer code with combinations of flags and runs the
existing `validate_c_inference.py` to produce intermediates, then compares
Python vs C integer dumps and reports mismatches.

Combinations tested (8 total):
 - USE_BANKERS_MULT: 0/1 (switch MultiplyByQuantizedMultiplier implementation)
 - BIAS_TIES_TO_EVEN: 0/1 (bias rounding)
 - TIES_TO_EVEN: 0/1 (affects right_shift_round behavior when bankers mult is used)

Produces: tools/auto_experiments_results.json in the repo.
"""
import itertools
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INFER = ROOT / 'infer'
VALIDATE = INFER / 'validate_c_inference.py'

combs = list(itertools.product([0,1],[0,1],[0,1]))
results = []

for use_bankers, bias_tie, ties_to_even in combs:
    label = f"BANKERS={use_bankers} BIAS_TEE={bias_tie} TEE={ties_to_even}"
    print('\n---', label)
    flags = []
    if use_bankers:
        flags.append('-DUSE_BANKERS_MULT')
    if bias_tie:
        flags.append('-DBIAS_TIES_TO_EVEN')
    if ties_to_even:
        flags.append('-DTIES_TO_EVEN')
    cflags = ' '.join(flags) + ' -Wall -Wextra -O2 -I.'

    # build
    try:
        subprocess.run(['make','clean'], cwd=str(INFER), check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['make', f'CFLAGS={cflags}'], cwd=str(INFER), check=True)
    except subprocess.CalledProcessError as e:
        print('Build failed:', e)
        results.append({'label':label,'status':'build-fail','error':str(e)})
        continue

    # run validation for 1 sample
    try:
        proc = subprocess.run(['python3', str(VALIDATE), '--limit','1','--dump-intermediates','--c_exe','./inference_app'], cwd=str(INFER), check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout.splitlines()[-5:])
    except subprocess.CalledProcessError as e:
        print('Validation failed:', e)
        results.append({'label':label,'status':'run-fail','error':str(e)})
        continue

    # locate dumps
    py = None
    c = None
    INTERMED = INFER / 'intermediates'
    if INTERMED.exists():
        dumps = sorted(INTERMED.iterdir(), key=lambda p: p.stat().st_mtime)
        for p in reversed(dumps):
            if p.name.endswith('_py.json'):
                py = p
                break
        for p in reversed(dumps):
            if p.name.startswith('inter_') and p.name.endswith('_c_int.json'):
                c = p
                break
    if not py or not c:
        print('Dumps missing:', py, c)
        results.append({'label':label,'status':'no-dumps'})
        continue

    pj = json.load(open(py))
    cj = json.load(open(c))

    def arr(d,k):
        v = d.get(k)
        if not v: return []
        if isinstance(v, list) and len(v) and isinstance(v[0], list):
            return v[0]
        return v

    diffs = {}
    total = 0
    for key in ['net_0_q_int','net_2_q_int','net_4_q_int']:
        pa = arr(pj,key)
        ca = arr(cj,key)
        mism = []
        L = max(len(pa), len(ca))
        for i in range(L):
            pv = pa[i] if i < len(pa) else None
            cv = ca[i] if i < len(ca) else None
            if pv != cv:
                mism.append({'idx':i,'py':pv,'c':cv})
        diffs[key] = mism
        total += len(mism)

    results.append({'label':label,'status':'ok','py_dump':py.name,'c_dump':c.name,'diffs':diffs,'total':total})
    print('Total mismatches:', total)

# write results
with open(ROOT / 'infer' / 'tools' / 'auto_experiments_results.json','w') as f:
    json.dump(results, f, indent=2)

print('\nWrote results to infer/tools/auto_experiments_results.json')
