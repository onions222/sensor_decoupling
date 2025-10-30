import json
import re
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: compare_intermediates.py <intermediate.json> <debug.txt>")
    sys.exit(2)

ij = Path(sys.argv[1])
dj = Path(sys.argv[2])

with open(ij,'r') as f:
    act = json.load(f)
with open(dj,'r') as f:
    dbg = f.read()

# extract net_4 python value
py_net4 = None
if 'net_4' in act:
    py_net4 = act['net_4'][0]
else:
    # net_4 may be 'final' or 'after_dequant_out'
    if 'after_dequant_out' in act:
        py_net4 = act['after_dequant_out'][0]
    elif 'final' in act:
        py_net4 = act['final'][0]

print('Python net outputs (pre/post dequant):')
print(py_net4)

# parse debug lines
lines = dbg.splitlines()
layer3_dbg = []
for L in lines:
    m = re.match(r"\[DBG3\] out=(\d+) acc32=(\-?\d+) acc_fp\(requant\)=(\-?[0-9.eE+-]+) fp_ref=(\-?[0-9.eE+-]+)", L)
    if m:
        o = int(m.group(1))
        acc32 = int(m.group(2))
        acc_fp = float(m.group(3))
        fp_ref = float(m.group(4))
        layer3_dbg.append({'out':o,'acc32':acc32,'acc_fp':acc_fp,'fp_ref':fp_ref})

print('\nC debug layer3 entries:')
for e in layer3_dbg:
    print(e)

# Compare python net_4 first element to C derived dequant value
# Python pre-tanh value appears in act['net_4'][0][0]
if py_net4 is not None and layer3_dbg:
    py_val = float(py_net4[0])
    # C final dequant we can extract from DBG_FINAL line
    m = re.search(r"\[DBG_FINAL\] layer3_q=\[(\-?\d+),(\-?\d+)\] deq=\[([0-9eE+\-\.]+),([0-9eE+\-\.]+)\] out=\[([0-9eE+\-\.]+),([0-9eE+\-\.]+)\]", dbg)
    if m:
        c_deq0 = float(m.group(3))
        print('\nC final dequant[0] =', c_deq0)
        print('Python net_4[0] =', py_val)
        print('Diff (py - c) =', py_val - c_deq0)
    else:
        print('\nNo DBG_FINAL dequant found in debug file.')
else:
    print('\nInsufficient data to compare.')
