"""Minimal collision reproduction and production-size contraction across two seeds."""
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE/'source/proj'))
from mh.v2_math import CountSketch
from contract_method import contract_patch

torch.set_num_threads(1)
small = CountSketch(4,1,1,1701,'cpu')
a = torch.ones(1)
_, old = small.patch(a,3,0)
original_ratio = float((a-small.encode(old)).norm()/a.norm())
assert original_ratio == 2.
_, corrected, residual, gamma = contract_patch(small,a,3,0)
assert residual.norm() < a.norm()*.001 and gamma > 0
records=[]
for seed in (42,43):
    for depth,width,k in ((3,256,128),(5,2048,2048)):
        sketch=CountSketch(395024,depth,width,1701,'cpu')
        generator=torch.Generator().manual_seed(seed)
        residual=torch.zeros(depth*width)
        for r in range(1,21):
            mean=sketch.encode(torch.randn(395024,generator=generator)*.001)
            accumulated=mean+residual
            wire,delta,residual,gamma=contract_patch(sketch,accumulated,k,r-1)
            assert residual.double().square().sum() <= accumulated.double().square().sum()
            records.append(dict(seed=seed,width=width,round=r,gamma=gamma,
                contraction=float(residual.norm()/accumulated.norm()),bytes=len(wire)))
proof=dict(original_minimal_ratio=original_ratio,repaired_minimal_ratio=float(
    (a-small.encode(corrected)).norm()/a.norm()),production_size_probes=records,
    scope='codec algebra on generated updates; not real-update fidelity, epoch stability, teacher or IDS utility')
(BASE/'probes/repair-algebra.json').write_text(json.dumps(proof,indent=2)+'\n')
print('Original collision residual x2; decoded line search contracts; 80 production-size cases PASS')
