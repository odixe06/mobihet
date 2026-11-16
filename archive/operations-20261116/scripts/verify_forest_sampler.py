"""Train/val-only fidelity gate for the torch port, before full-network T4 runtime runs."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from methods.mobihet_forest_quantile import Sampler
from scripts.marginal_direct_generator import Generator
from scripts.probe_pair_generator import PAIRS,evaluate,load


def main():
    out=ROOT/"runs/forest-production"
    report_path=out/"sampler-gate.json"
    assert not report_path.exists(),"Preserve completed fidelity evidence"
    sources={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
             ("methods/mobihet_forest_quantile.py","scripts/direct_generator.py","scripts/marginal_direct_generator.py",
              "scripts/probe_pair_generator.py","scripts/verify_forest_sampler.py")}
    order,((xt,yt),(xv,yv)),files=load()
    digest=hashlib.sha256()
    for a in (xt,yt,xv,yv):digest.update(a.tobytes())
    pairs=[(order.index(a),order.index(b)) for a,b in PAIRS]
    rows=[];started=time.monotonic()
    for c in range(16):
        m=Generator.fit(xt[yt==c],pairs)
        sampler=Sampler([m],"cuda")
        gen=torch.Generator(device="cuda").manual_seed(8100+c)
        x=sampler.sample(16384,gen)[0].cpu().numpy()
        assert np.isfinite(x).all()
        baseline=m.sample(np.random.default_rng(8100+c),16384)
        a,_=evaluate(xv[yv==c],baseline,pairs);b,_=evaluate(xv[yv==c],x,pairs)
        row=dict(class_id=c,baseline=a,torch_forest=b)
        rows.append(row);print("FOREST_SAMPLER_CLASS",json.dumps(row),flush=True)
    before=float(np.mean([r["baseline"]["pair_tv"] for r in rows]))
    after=float(np.mean([r["torch_forest"]["pair_tv"] for r in rows]))
    gates=dict(pair_improvement=bool(after<=.8*before),singleton=all(r["torch_forest"]["singleton_tv"]<=r["baseline"]["singleton_tv"]+.02 for r in rows))
    assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in sources.items())
    report_path.write_text(json.dumps(dict(sources=sources,classes=rows,gates=gates,pair_before=before,pair_after=after,
        sample_sha256=digest.hexdigest(),files=files,seconds=time.monotonic()-started,
        scope="bounded reused train->val panel, GPU torch sampler port; no official-test/FL utility proof"),indent=2)+"\n")
    assert all(gates.values()),"Torch port did not pass locked statistical gate; do not launch full-network run"
    print("FOREST_SAMPLER_GATE_PASS",flush=True)


if __name__=="__main__":main()
