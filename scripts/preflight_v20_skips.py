"""Prove step observation preserves real-fixture weights, BN, RNG and returned stats."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from core import grouped, local, fl
from core.dagsnet import width
from methods.mobihet_study import Method as Basic
from scripts.probe_v20_skips import ObservedBasic
from scripts.run_v20 import DiagnosticEnv
from scripts.run_fl import PROTOCOL

OUT = ROOT / "runs/v-study/numerics-guard"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names = ["scripts/probe_v20_skips.py", "scripts/preflight_v20_skips.py"]
    before = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    os.environ["V4_NPY"] = str(ROOT / "runs/v-study/fixture-cache")
    reports = []
    for spec in json.loads((ROOT / "runs/v-study/batch.json").read_text())[:2]:
        hp = dict(PROTOCOL)
        for item in spec["hp"]:
            k, v = item.split("=", 1)
            try: hp[k] = json.loads(v)
            except json.JSONDecodeError: hp[k] = v
        hp.update(group_rows=128, diagnostic_ids=[613, 2029])
        env = DiagnosticEnv(width(1), hp, clients=[613, 2029])
        grouped.compile_all()
        ok, _ = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
        assert ok
        initial = env.gen.get_state().clone(); expected = []
        reference = Basic(env); torch.manual_seed(hp["seed"])
        for r in (1, 2, 3):
            info = reference.round(r, fl.cosine_lr(r, 50, .001, .00001))
            expected.append(({k: getattr(reference, k).clone() for k in ("priv_p", "priv_b")}, info,
                env.gen.get_state().clone(), torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()))
        del reference; gc.collect(); torch.cuda.empty_cache()
        env.gen.set_state(initial); model = ObservedBasic(env); torch.manual_seed(hp["seed"])
        for r in (1, 2, 3):
            info = model.round(r, fl.cosine_lr(r, 50, .001, .00001)); weights, old_info, gen, cpu, cuda = expected[r - 1]
            for k, v in weights.items(): assert torch.equal(getattr(model, k), v)
            assert info == old_info and torch.equal(env.gen.get_state(), gen)
            assert torch.equal(torch.get_rng_state(), cpu) and torch.equal(torch.cuda.get_rng_state(), cuda)
            assert model.observation["all_finite"] and not model.observation["zero_applied_client_ids"]
        reports.append({"seed": hp["seed"], "rounds": 3, "weights_bn_rng_stats_bitexact": True})
        del model, expected; gc.collect(); torch.cuda.empty_cache()
    assert before == {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    (OUT / "preflight.json").write_text(json.dumps({"source_sha256": before, "cells": reports,
        "scope": "Real150train/23val sm86, observation equivalence only; no T4 skip frequency or numerical cause proof"}, indent=2) + "\n")
    print("V20_NUMERICS_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
