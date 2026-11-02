"""Both-seed CE recovery gate and metadata preserve real-fixture training exactly."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from core import fl, grouped, local
from core.dagsnet import width, DAGSNet
from methods.mobihet_study import Method as Basic
from methods.mobihet_vrecover import Method
from scripts.run_v20 import DiagnosticEnv
from scripts.run_fl import PROTOCOL
from scripts.preflight_q20 import equal_state

OUT = ROOT / "runs/v-study/recovery-ce"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    parents = json.loads((ROOT / "runs/v-study/batch.json").read_text())[:2]
    names = list(parents[0]["overlay_sha256"]) + ["scripts/probe_v20_skips.py", "methods/mobihet_vrecover.py",
        "scripts/preflight_v20_recovery.py", "tests/test_v20_recovery.py"]
    before = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    assert all(before[n] == h for n, h in parents[0]["overlay_sha256"].items())
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_v20_recovery.py", "tests/test_v20_skips.py",
        "tests/test_v20_data.py", "tests/test_v20_audit.py", "--junitxml=" + str(OUT / "tests.xml")], cwd=ROOT, check=True)
    os.environ["V4_NPY"] = str(ROOT / "runs/v-study/fixture-cache"); reports = []
    for parent in parents:
        hp = dict(PROTOCOL)
        for item in parent["hp"]:
            k, v = item.split("=", 1)
            try: hp[k] = json.loads(v)
            except json.JSONDecodeError: hp[k] = v
        hp.update(group_rows=128, diagnostic_ids=[613, 2029], diagnostic_rounds=[1, 2, 3])
        env = DiagnosticEnv(width(1), hp, clients=[613, 2029])
        grouped.compile_all()
        ok, _ = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
        assert ok
        initial = env.gen.get_state().clone(); expected = []; model = Basic(env); torch.manual_seed(hp["seed"])
        for r in (1, 2, 3):
            info = model.round(r, fl.cosine_lr(r, 50, .001, .00001))
            expected.append(({k: getattr(model, k).clone() for k in ("priv_p", "priv_b")}, info,
                env.gen.get_state().clone(), torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()))
        del model; gc.collect(); torch.cuda.empty_cache()
        env.gen.set_state(initial); model = Method(env); torch.manual_seed(hp["seed"])
        folder = OUT / f"smoke-s{hp['seed']}"; os.environ["MOBIHET_RECOVERY_DIR"] = str(folder / "recovery")
        for r in (1, 2, 3):
            info = model.round(r, fl.cosine_lr(r, 50, .001, .00001)); weights, old_info, gen, cpu, cuda = expected[r - 1]
            for k, v in weights.items(): assert torch.equal(getattr(model, k), v)
            assert all(info[k] == v for k, v in old_info.items())
            assert info["applied_steps"] > 0 and info["skipped_steps"] == 0
            assert torch.equal(env.gen.get_state(), gen) and torch.equal(torch.get_rng_state(), cpu) and torch.equal(torch.cuda.get_rng_state(), cuda)
        before_state = model.state(); rng = (env.gen.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
        model.diagnostics(3, folder); equal_state(before_state, model.state())
        assert all(torch.equal(a, b) for a, b in zip(rng, (env.gen.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())))
        restored = Method(env); restored.load_state(before_state); equal_state(before_state, restored.state())
        single = DAGSNet(env.cfg); single.load_state_dict(env.lay.state_dict(model.priv_p[0], model.priv_b[0]), strict=True)
        assert sum(p.numel() for p in single.parameters()) == 395408
        reports.append({"seed": hp["seed"], "rounds": 3, "weights_bn_rng_stats_bitexact": True,
            "diagnostic_state_rng_unchanged": True, "strict_rebuild": True, "state_roundtrip": True})
        del model, restored, single, before_state, expected; gc.collect(); torch.cuda.empty_cache()
    assert before == {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    (OUT / "preflight.json").write_text(json.dumps({"source_sha256": before, "cells": reports,
        "scope": "Guard-only CE recovery; localsm86/150train23val, not full20roundT4 proof"}, indent=2) + "\n")
    print("V20_CE_RECOVERY_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
