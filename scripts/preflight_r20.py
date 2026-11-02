"""Both-seed real W1: unchanged controls and fixed-history/manual-injection equivalence."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from core import fl, grouped, local
from core.dagsnet import DAGSNet, width
from methods import mobihet_rstudy as replay
from methods import mobihet_rstudy_data as frozen
from methods.mobihet_next import Method as Live
from scripts.q20_common import settings
from scripts.run_fl import PROTOCOL
from scripts.preflight_q20 import equal_state

OUT = ROOT / "runs/r-study"


def capture(model):
    return ({k: getattr(model, k).clone() for k in ("priv_p", "priv_b", "Q")},
            model.env.gen.get_state().clone(), model.gen_g.get_state().clone(),
            torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone(), model.evaluate()["private"].copy())


def same(model, expected):
    values, gen, synth_gen, cpu, cuda, cm = expected
    for key, value in values.items(): assert torch.equal(value, getattr(model, key)), key
    assert torch.equal(gen, model.env.gen.get_state()) and torch.equal(synth_gen, model.gen_g.get_state())
    assert torch.equal(cpu, torch.get_rng_state()) and torch.equal(cuda, torch.cuda.get_rng_state())
    assert np.array_equal(cm, model.evaluate()["private"])


def main():
    OUT.mkdir(exist_ok=True)
    parent = json.loads((ROOT / "runs/q-study/batch.json").read_text())[0]
    names = list(parent["overlay_sha256"]) + ["methods/mobihet_rstudy.py", "methods/mobihet_rstudy_data.py",
        "scripts/preflight_r20.py", "tests/test_r20.py"]
    before = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    for n, h in parent["overlay_sha256"].items(): assert before[n] == h, n
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_r20.py", "tests/test_mobihet_next.py",
        "tests/test_mobihet_study.py", "tests/test_q20_targets.py", "tests/test_q20_audit.py",
        "--junitxml=" + str(OUT / "tests.xml")], cwd=ROOT, check=True)
    reports = []
    for seed in (42, 43):
        remote = replay.load_schedule(seed, "cuda", frozen.SHA256)
        del remote
        hp = dict(PROTOCOL)
        for item in parent["hp"]:
            key, value = item.split("=", 1)
            try: hp[key] = json.loads(value)
            except json.JSONDecodeError: hp[key] = value
        hp.update(settings("stream"), group_rows=128, seed=seed, diagnostic_ids=[613, 2029],
                  diagnostic_rounds=[1, 2, 3], external_diagnostics=False)
        env = fl.Env(width(1), hp, clients=[613, 2029], cache=str(ROOT / "runs/n-study/smoke-cache"))
        assert len(env.train.Y) == 173 and len(env.test.Y) == 46
        grouped.compile_all()
        ok, gate = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
        assert ok, gate
        initial = env.gen.get_state().clone()
        os.environ["MOBIHET_RECOVERY_DIR"] = str(OUT / f"smoke-seed{seed}/recovery")
        reference = Live(env); torch.manual_seed(seed); expected = []
        for r in (1, 2, 3):
            reference.round(r, fl.cosine_lr(r, 50, .001, .00001)); expected.append(capture(reference))
        del reference; gc.collect(); torch.cuda.empty_cache()
        for repeat in ("A", "B"):
            env.hp["r_kind"] = "repeat"; env.gen.set_state(initial)
            model = replay.Method(env); torch.manual_seed(seed)
            for r in (1, 2, 3):
                row = model.round(r, fl.cosine_lr(r, 50, .001, .00001)); assert row["skipped_frac"] == 0
                same(model, expected[r - 1])
            reports.append({"kind": "repeat", "repeat": repeat, "seed": seed, "bitexact": True})
            del model; gc.collect(); torch.cuda.empty_cache()
        del expected
        # Teacher source here is a real fixture, independent of reception. Full history decode was checked above.
        env.hp.update(settings("mixed_input")); env.gen.set_state(initial)
        reference = Live(env); torch.manual_seed(seed)
        history = {key: [] for key in ("Q", "target", "body", "banks")}
        for r in (1, 2, 3):
            reference.round(r, fl.cosine_lr(r, 50, .001, .00001))
            for key, value in (("Q", reference.Q), ("target", reference.bank_target), ("body", reference.bank_body)):
                history[key].append(value.clone())
            history["banks"].append(replay.bank_sha(reference.bank_X))
        for key in ("Q", "target", "body"): history[key] = torch.stack(history[key])
        del reference; gc.collect(); torch.cuda.empty_cache()
        first = None
        for mode in ("mixed_input", "real", "margin", "strength"):
            env.hp.update(settings(mode), r_kind="replay", r_source_sha256=frozen.SHA256)
            env.gen.set_state(initial); manual = Live(env); torch.manual_seed(seed); expected = []
            for r in (1, 2, 3):
                manual.round(r, fl.cosine_lr(r, 50, .001, .00001))
                manual.Q = history["Q"][r - 1].clone(); manual.bank_target = history["target"][r - 1].clone()
                manual.bank_body = history["body"][r - 1].clone()
                expected.append(capture(manual))
            del manual; gc.collect(); torch.cuda.empty_cache()
            env.gen.set_state(initial)
            with patch.object(replay, "load_schedule", return_value=history): model = replay.Method(env)
            torch.manual_seed(seed)
            for r in (1, 2, 3):
                row = model.round(r, fl.cosine_lr(r, 50, .001, .00001)); assert row["skipped_frac"] == 0
                same(model, expected[r - 1])
                if r == 1:
                    value = {k: getattr(model, k).clone() for k in ("priv_p", "priv_b")}
                    if first is None: first = value
                    else:
                        for k in value: assert torch.equal(value[k], first[k])
            state = copy.deepcopy(model.state()); rng = (env.gen.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
            folder = OUT / f"smoke-{mode}-s{seed}"; folder.mkdir(exist_ok=True)
            model.diagnostics(3, folder); equal_state(state, model.state())
            assert all(torch.equal(a, b) for a, b in zip(rng, (env.gen.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())))
            with patch.object(replay, "load_schedule", return_value=history): restored = replay.Method(env)
            restored.load_state(state); equal_state(state, restored.state())
            single = DAGSNet(env.cfg); single.load_state_dict(env.lay.state_dict(restored.priv_p[0], restored.priv_b[0]), strict=True)
            assert sum(p.numel() for p in single.parameters()) == 395408
            reports.append({"kind": "replay", "mode": mode, "seed": seed, "manual_injection_bitexact": True,
                "rounds": 3, "state_roundtrip": True, "diagnostic_state_rng_unchanged": True, "strict_rebuild": True})
            print("R20_REAL_CELL_PASS", seed, mode, flush=True)
            del model, restored, state, single, expected; gc.collect(); torch.cuda.empty_cache()
    assert before == {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    (OUT / "preflight.json").write_text(json.dumps({"source_sha256": before, "source_unchanged": True, "cells": reports,
        "cuda_frozen_decode_exact_both_seeds": True, "replay_first_round_bitexact": True,
        "scope": "Realv4 W1 local sm86 173train/46test; not T4/full5343/utility/fullresume proof"}, indent=2) + "\n")
    print("R20_REAL_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
