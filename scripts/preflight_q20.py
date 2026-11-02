"""Real v4 Q20: both seeds, matched controls, full W1, compiled AMP, artifacts and guard checks."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from core import fl, grouped, local
from core.dagsnet import DAGSNet, width
from methods.mobihet_next import Method
from methods.mobihet_study import Method as Parent, decode_probability
from scripts.q20_common import CELLS, settings
from scripts.run_fl import PROTOCOL

OUT = ROOT / "runs/q-study"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def equal_state(a, b):
    assert a.keys() == b.keys()
    for k, value in a.items():
        if torch.is_tensor(value): assert torch.equal(value, b[k]), k
        else: assert value == b[k], k


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = json.loads((ROOT / "runs/p-study/batch.json").read_text())[4]
    names = list(original["overlay_sha256"]) + ["scripts/run_p21_recovery.py", "methods/mobihet_next.py",
        "scripts/q20_common.py", "scripts/run_q20.py", "scripts/preflight_q20.py", "tests/test_mobihet_next.py"]
    before = {name: sha(ROOT / name) for name in names}
    for name, h in original["overlay_sha256"].items(): assert before[name] == h
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_mobihet_next.py", "tests/test_mobihet_study.py",
        "tests/test_p20_audit.py", "--junitxml=" + str(OUT / "tests.xml")], cwd=ROOT, check=True)
    reports = []
    for seed in (42, 43):
        hp = dict(PROTOCOL)
        for item in original["hp"]:
            k, v = item.split("=", 1)
            try: hp[k] = json.loads(v)
            except json.JSONDecodeError: hp[k] = v
        hp.update(settings("stream"), group_rows=128, seed=seed, diagnostic_ids=[613, 2029],
                  diagnostic_rounds=[1, 2, 3], external_diagnostics=False)
        env = fl.Env(width(1), hp, clients=[613, 2029], cache=str(ROOT / "runs/n-study/smoke-cache"))
        assert len(env.train.Y) == 173 and len(env.test.Y) == 46
        grouped.compile_all()
        ok, gate = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
        assert ok, gate
        initial_gen = env.gen.get_state().clone()
        env.gen.set_state(initial_gen); reference = Parent(env); torch.manual_seed(seed)
        refs = []
        for r in (1, 2, 3):
            reference.round(r, fl.cosine_lr(r, 50, .001, .00001))
            refs.append(({k: getattr(reference, k).clone() for k in ("priv_p", "priv_b", "Q")},
                         env.gen.get_state().clone(), reference.gen_g.get_state().clone(), torch.get_rng_state().clone(),
                         torch.cuda.get_rng_state().clone(), reference.evaluate()["private"].copy()))
        del reference; gc.collect(); torch.cuda.empty_cache()
        mixed_first = None
        for mode in CELLS:
            env.hp.update(settings(mode)); env.gen.set_state(initial_gen)
            model = Method(env); torch.manual_seed(seed)
            folder = OUT / f"smoke-{mode}-s{seed}"; folder.mkdir(exist_ok=True)
            os.environ["MOBIHET_RECOVERY_DIR"] = str(folder / "recovery")
            started = time.monotonic(); rows = []
            for r in (1, 2, 3):
                row = model.round(r, fl.cosine_lr(r, 50, .001, .00001)); rows.append(row)
                assert row["skipped_frac"] == 0 and np.isfinite(row["train_loss"])
                assert torch.isfinite(model.priv_p).all() and torch.isfinite(model.priv_b).all()
                cm = model.evaluate()["private"]; assert cm.sum() == 46
                if mode in {"stream", "query"}:
                    values, gen, synth_gen, cpu_rng, cuda_rng, expected = refs[r - 1]
                    for key, value in values.items(): assert torch.equal(value, getattr(model, key)), (mode, seed, r, key)
                    assert torch.equal(gen, env.gen.get_state()) and torch.equal(synth_gen, model.gen_g.get_state())
                    assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
                    assert np.array_equal(expected, cm)
                elif r == 1:
                    values = {k: getattr(model, k).clone() for k in ("priv_p", "priv_b", "Q")}
                    if mixed_first is None: mixed_first = values
                    else:
                        for key, value in values.items(): assert torch.equal(value, mixed_first[key]), (mode, seed, key)
                if mode not in {"stream", "query"}:
                    assert torch.equal(model.Q, decode_probability((model.class_sums / model.class_totals[:, None]).float().half()))
                if mode != "stream":
                    assert torch.equal(model.bank_target, decode_probability(model.bank_body))
                    assert torch.equal(model.bank_body, (model.aggregate_sums / model.aggregate_weights[:, None]).float().half())
                    old = model.gen_g.get_state().clone(); x, y = model.make_bank(r)
                    assert torch.equal(x, model.bank_X) and torch.equal(y, model.bank_y) and torch.equal(old, model.gen_g.get_state())
            state = copy.deepcopy(model.state()); gen = env.gen.get_state().clone()
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            model.diagnostics(3, folder)
            equal_state(state, model.state())
            assert torch.equal(gen, env.gen.get_state()) and torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(cuda_rng, torch.cuda.get_rng_state())
            file = folder / "state.pt"; torch.save(state, file)
            restored = Method(env); restored.load_state(torch.load(file, weights_only=True, map_location=env.dev))
            equal_state(state, restored.state()); assert np.array_equal(cm, restored.evaluate()["private"])
            single = DAGSNet(env.cfg); single.load_state_dict(env.lay.state_dict(restored.priv_p[0], restored.priv_b[0]), strict=True)
            assert sum(p.numel() for p in single.parameters()) == 395408
            reports.append({"seed": seed, "mode": mode, "rounds": rows, "seconds": time.monotonic() - started,
                "strict_rebuild": True, "state_roundtrip": True, "diagnostic_state_rng_unchanged": True})
            print("Q20_REAL_CELL_PASS", seed, mode, flush=True)
            del model, restored, single, state
            gc.collect(); torch.cuda.empty_cache()
    assert {name: sha(ROOT / name) for name in names} == before, "Source changed during preflight"
    (OUT / "preflight.json").write_text(json.dumps({"source_sha256": before, "cells": reports,
        "source_unchanged": True, "reference_and_query_3round_bitexact_both_seeds": True,
        "mixed_first_round_bitexact_both_seeds": True, "train_rows": 173, "test_rows": 46,
        "scope": "Local sm86 realv4 W1 compiledAMP, not full5343/T4/utility/fullresume proof"}, indent=2) + "\n")
    print("Q20_REAL_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
