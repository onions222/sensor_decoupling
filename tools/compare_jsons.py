#!/usr/bin/env python3
import json
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print('Usage: compare_jsons.py <python_intermediate.json> <c_intermediate.json>')
    sys.exit(2)

pyf = Path(sys.argv[1])
cf = Path(sys.argv[2])
if not pyf.exists() or not cf.exists():
    print('Files not found')
    sys.exit(2)

py = json.loads(pyf.read_text())
c = json.loads(cf.read_text())

layers = ['after_quant_in','net_0','net_1','net_2','net_3','net_4','after_dequant_out','final']

for layer in layers:
    if layer in py and layer in c:
        pa = py[layer]
        ca = c[layer]
        # assume shapes: batch x N
        if not pa or not ca:
            print(f'{layer}: empty')
            continue
        pa0 = pa[0]
        ca0 = ca[0]
        n = min(len(pa0), len(ca0))
        diffs = [abs(float(pa0[i]) - float(ca0[i])) for i in range(n)]
        maxd = max(diffs) if diffs else 0.0
        avgd = sum(diffs)/len(diffs) if diffs else 0.0
        print(f'{layer}: n={n} max_diff={maxd:.12g} avg_diff={avgd:.12g}')
        # print top up to 5 diffs
        idxs = sorted(range(n), key=lambda i: diffs[i], reverse=True)[:5]
        for i in idxs:
            print(f'  idx={i} py={pa0[i]:.12g} c={ca0[i]:.12g} diff={diffs[i]:.12g}')
    else:
        print(f'{layer}: missing in one of the files')

# Compare final scalar outputs (first element of final batch)
if 'final' in py and 'final' in c:
    p = py['final'][0]
    q = c['final'][0]
    print('\nFinal output comparison:')
    print(f'  python final = {p}')
    print(f'  c final     = {q}')
    diffs = [abs(float(p[i]) - float(q[i])) for i in range(min(len(p), len(q)))]
    print(f'  diffs = {diffs}')
