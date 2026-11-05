"""Clean split, both-seed real W1 objectives and isolated bootstrap before any V20 launch."""
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from core import data, fl, grouped, local
from core.dagsnet import DAGSNet, width
from methods.mobihet_vstudy import Method
from scripts.p20_common import settings as basic_settings
from scripts.q20_common import settings as mixed_settings
from scripts.preflight_q20 import equal_state
from scripts.run_fl import PROTOCOL
from scripts.run_v20 import DiagnosticEnv

OUT = ROOT / "runs/v-study"
CELLS = ("real", "augment", "mixed_input", "margin")


def settings(mode):
    return (basic_settings(mode) if mode in {"real", "augment"} else mixed_settings(mode)) | {
        "validation_diagnostic": True, "external_diagnostics": False}


def main():
    OUT.mkdir(exist_ok=True)
    split = json.loads((OUT / "split-audit.json").read_text())
    assert split["train_plus_val_counts_exact"] and split["all_clients_temporal_train_before_val"]
    parent = json.loads((ROOT / "runs/q-study/batch.json").read_text())[0]
    names = list(parent["overlay_sha256"]) + ["methods/mobihet_vstudy.py", "scripts/run_v20.py",
        "scripts/preflight_v20.py", "tests/test_v20_data.py"]
    before = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    for n, h in parent["overlay_sha256"].items(): assert before[n] == h
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_v20_data.py", "--junitxml=" + str(OUT / "tests.xml")],
        cwd=ROOT, check=True)
    cache = OUT / "fixture-cache"; cache.mkdir(exist_ok=True)
    if not (cache / "done").exists():
        s = data.load(data.find_root(), "train", clients=[613, 2029], with_fold=True)
        for key in ("X", "Y", "off", "ids", "F"): np.save(cache / f"train_{key}.npy", getattr(s, key))
        (cache / "done").write_text("diagnostic fixture train-only source")
    os.environ["V4_NPY"] = str(cache)
    train_counts, val_counts = (np.load(OUT / f"{f}_counts.npy") for f in ("train", "val"))
    reports = []
    for seed in (42, 43):
        hp = dict(PROTOCOL)
        for item in parent["hp"]:
            k, v = item.split("=", 1)
            try: hp[k] = json.loads(v)
            except json.JSONDecodeError: hp[k] = v
        hp.update(settings("mixed_input"), group_rows=128, seed=seed, diagnostic_ids=[613, 2029], diagnostic_rounds=[1, 2, 3])
        env = DiagnosticEnv(width(1), hp, clients=[613, 2029])
        assert len(env.train.Y) == train_counts[[613, 2029]].sum()
        assert len(env.test.Y) == val_counts[[613, 2029]].sum()
        grouped.compile_all()
        ok, gate = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
        assert ok, gate
        initial = env.gen.get_state().clone(); mixed_first = None
        for mode in CELLS:
            env.hp.update(settings(mode)); env.gen.set_state(initial)
            folder = OUT / f"smoke-{mode}-s{seed}"; folder.mkdir(exist_ok=True)
            os.environ["MOBIHET_RECOVERY_DIR"] = str(folder / "recovery")
            model = Method(env); torch.manual_seed(seed); rows = []
            assert torch.equal(model.cnt.cpu().long(), torch.from_numpy(train_counts[env.ids]))
            for r in (1, 2, 3):
                row = model.round(r, fl.cosine_lr(r, 50, .001, .00001)); rows.append(row)
                assert row["skipped_frac"] == 0 and np.isfinite(row["train_loss"])
                assert torch.isfinite(model.priv_p).all() and torch.isfinite(model.priv_b).all()
                cm = model.evaluate()["private"]
                assert np.array_equal(cm.sum(2), val_counts[env.ids])
                if mode in {"mixed_input", "margin"} and r == 1:
                    state = (model.priv_p.clone(), model.priv_b.clone())
                    if mixed_first is None: mixed_first = state
                    else: assert all(torch.equal(a, b) for a, b in zip(state, mixed_first))
            state = model.state(); before_rng = (env.gen.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
            model.diagnostics(3, folder); equal_state(state, model.state())
            assert all(torch.equal(a, b) for a, b in zip(before_rng, (env.gen.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())))
            restored = Method(env); restored.load_state(state); equal_state(state, restored.state())
            assert np.array_equal(cm, restored.evaluate()["private"])
            single = DAGSNet(env.cfg); single.load_state_dict(env.lay.state_dict(model.priv_p[0], model.priv_b[0]), strict=True)
            assert sum(p.numel() for p in single.parameters()) == 395408
            reports.append({"mode": mode, "seed": seed, "train_rows": len(env.train.Y), "val_rows": len(env.test.Y),
                "rounds": rows, "state_roundtrip": True, "strict_rebuild": True, "train_only_class_counts_exact": True})
            print("V20_REAL_CELL_PASS", seed, mode, flush=True)
            del model, restored, state, single; gc.collect(); torch.cuda.empty_cache()
    assert before == {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in names}
    (OUT / "preflight.json").write_text(json.dumps({"source_sha256": before, "source_unchanged": True, "cells": reports,
        "scope": "Originalfoldtrain/val realW1/2seed/3round sm86; not full5343/T4/utility proof"}, indent=2) + "\n")
    print("V20_REAL_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
