"""Exact driver/compile/crash-resume preflight for the new forest sampler."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import preflight_standard_copula as standard
from scripts.preflight_f50 import history,same_tree
from scripts.f50_common import hp_for

OUT=ROOT/"runs/forest-production"
SOURCES=("methods/mobihet_forest_quantile.py","scripts/direct_generator.py","scripts/marginal_direct_generator.py",
         "scripts/probe_pair_generator.py","scripts/preflight_forest_quantile.py","methods/mobihet.py",
         "methods/mobihet_wstudy.py","methods/mobihet_next.py","scripts/run_fl.py","core/local.py")


def drive(path,until,seed):
    cmd=[sys.executable,"scripts/run_q20.py","--method","mobihet_forest_quantile","--out",str(path),"--width","1",
         "--rounds","50","--until",str(until),"--save-every","1","--compile","--clients","613,2029",
         "--cache","runs/n-study/smoke-cache","--set",*hp_for("core",seed),*standard.FIXTURE]
    p=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    (path.parent/(path.name+f"-u{until}.log")).write_text(p.stdout+p.stderr)
    assert p.returncode==0 and "compile gate rank0: PASS" in p.stdout,str(path)
    return p.stdout


def main():
    OUT.mkdir(exist_ok=True)
    before={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in SOURCES}
    subprocess.run([sys.executable,"-m","pytest","-q","tests/test_forest_quantile_torch.py",
                    "tests/test_direct_generator.py","tests/test_marginal_direct_generator.py"],cwd=ROOT,check=True)
    reports=[]
    for seed in (42,43):
        parent=OUT/f"s{seed}"
        assert not parent.exists(),"Preserve old evidence; diagnose and use distinct directory on retry"
        parent.mkdir();a,b=parent/"straight",parent/"resumed"
        drive(a,4,seed);drive(b,2,seed)
        checkpoint=parent/"round2.pt";shutil.copyfile(b/"state.pt",checkpoint)
        assert "resumed rank0 at round 3" in drive(b,3,seed)
        shutil.copyfile(checkpoint,b/"state.pt")
        assert "dropped 1 rows past round 2" in drive(b,4,seed)
        assert history(a)==history(b)
        for r in range(1,5):
            with np.load(a/f"cms_r{r:03d}.npz") as x,np.load(b/f"cms_r{r:03d}.npz") as y:
                assert x.files==y.files and all(np.array_equal(x[k],y[k]) for k in x.files)
            assert same_tree(torch.load(a/f"models/r{r:03d}.pt",weights_only=True),torch.load(b/f"models/r{r:03d}.pt",weights_only=True))
        assert same_tree(torch.load(a/"state.pt",weights_only=True),torch.load(b/"state.pt",weights_only=True))
        assert all(row["bytes_up"]==row["bytes_down"]==544 for row in history(a))
        for folder in (a,b):
            record=json.loads((folder/"recovery/forest_rank0.json").read_text())
            assert record["counts_consistent"] and sum(record["bootstrap_bytes"].values())==1704951
        reports.append(dict(seed=seed,compile_pass=True,state_history_cm_snapshot_crash_resume_bitexact=True))
        print("FOREST_PREFLIGHT_SEED_PASS",seed,flush=True)
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in before.items())
    (OUT/"preflight.json").write_text(json.dumps(dict(sources=before,reports=reports,
        scope="local sm86/world1/2vehicles, not T4/2rank/fullnetwork or utilityPASS"),indent=2)+"\n")
    print("FOREST_PRODUCTION_PREFLIGHT_DONE",flush=True)


if __name__=="__main__":main()
