"""P20 real v4 fixture, full W1, compiled AMP, production server steps and decoded carriers."""
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from core import fl, grouped, local
from core.dagsnet import DAGSNet, width
from methods.mobihet import Method as Base
from methods.mobihet_study import Method
from scripts.p20_common import CELLS, settings
from scripts.run_fl import PROTOCOL

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "runs/p-study"
    sources = ["methods/mobihet_study.py", "methods/mobihet_study_wire.py", "methods/mobihet_study_diagnostics.py",
               "scripts/run_fl.py", "scripts/p20_common.py", "scripts/preflight_p20.py"]
    source_before = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy")
    ids = [613, 2029]  # Same train-only all-class fixture already audited N/O.
    hp = dict(PROTOCOL, **settings("class"), group_rows=128, seed=42)
    env = fl.Env(width(1), hp, clients=ids, cache=str(ROOT / "runs/n-study/smoke-cache"))
    assert len(env.train.Y) == counts[ids].sum() == 173
    grouped.compile_all()
    ok, compile_report = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
    assert ok, compile_report
    initial_gen = env.gen.get_state()
    records, first_bank, first_stream, proxy_reference = [], None, None, {}
    for mode in CELLS:
        env.hp.update(settings(mode), diagnostic_ids=ids, diagnostic_rounds=[1, 2, 3], external_diagnostics=False)
        env.gen.set_state(initial_gen)
        m = Method(env)
        torch.manual_seed(42)
        t0 = time.monotonic(); rows = []
        for r in (1, 2, 3):
            info = m.round(r, fl.cosine_lr(r, 50, .001, .00001))
            assert info["skipped_frac"] == 0 and np.isfinite(info["train_loss"])
            assert torch.isfinite(m.priv_p).all() and torch.isfinite(m.priv_b).all()
            cm = m.evaluate()["private"]
            assert cm.sum() == len(env.test.Y) == 46
            if mode in {"proxy_bank", "proxy_weights_bank"}:
                digest = hashlib.sha256()
                for tensor in (m.priv_p, m.priv_b, m.bank_target, m.server_p, m.server_b):
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
                if mode == "proxy_bank": proxy_reference[r] = digest.hexdigest()
                else: assert proxy_reference[r] == digest.hexdigest(), (mode, r, "carrier changed training function")
            if r == 1:
                if mode in {"bank_class", "bank_input", "proxy_bank", "proxy_hybrid", "bank_hard", "proxy_weights_bank"}:
                    if first_bank is None: first_bank = (m.priv_p.clone(), m.priv_b.clone())
                    else:
                        assert torch.equal(first_bank[0], m.priv_p) and torch.equal(first_bank[1], m.priv_b), mode
                if mode in {"augment", "class", "gate"}:
                    if first_stream is None: first_stream = (m.priv_p.clone(), m.priv_b.clone())
                    else:
                        assert torch.equal(first_stream[0], m.priv_p) and torch.equal(first_stream[1], m.priv_b), mode
            rows.append(info)
        before = m.priv_p.clone(), m.priv_b.clone(), m.gen_g.get_state().clone(), env.gen.get_state().clone()
        m.diagnostics(3, out)
        for a, b in zip(before, (m.priv_p, m.priv_b, m.gen_g.get_state(), env.gen.get_state())):
            assert torch.equal(a, b), (mode, "diagnostic mutated training state")
        file = out / f"smoke-{mode}.pt"
        torch.save(m.state(), file)
        restored = Method(env)
        restored.load_state(torch.load(file, weights_only=True, map_location=env.dev))
        assert np.array_equal(restored.evaluate()["private"], cm)
        model = DAGSNet(env.cfg)
        model.load_state_dict(env.lay.state_dict(restored.priv_p[0], restored.priv_b[0]), strict=True)
        assert sum(p.numel() for p in model.parameters()) == 395408
        record = {"mode": mode, "seconds": time.monotonic() - t0, "rounds": rows, "state_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                  "strict_rebuild": True, "diagnostic_state_unchanged": True, "eval_state_roundtrip": True}
        # Bounded synthetic evidence of the specific v4 constraint, not classifier utility.
        if mode in {"class", "project_class"}:
            from types import SimpleNamespace
            ctx = SimpleNamespace(Gfix=1, x=env.train.X[:1][None], clients=torch.zeros(1, dtype=torch.long, device=env.dev))
            x, _ = m._synth(ctx)
            record["nonbinary_nan_flag_fraction"] = float(((x[..., 71:] != 0) & (x[..., 71:] != 1)).float().mean())
        records.append(record)
        print("P20_REAL_CELL_PASS", mode, round(record["seconds"], 1), flush=True)
        del m, restored, model
        gc.collect(); torch.cuda.empty_cache()
    source_after = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    assert source_after == source_before, "Source changed during preflight; outputs are not a gate for these bytes"
    report = {"device": str(env.dev), "ids": ids, "train_rows": 173, "test_rows": 46,
              "compile_gate": compile_report, "production_teacher_rows": 1024, "production_proxy_steps": 200,
              "same_bank_r1_bitexact": True, "stream_controls_r1_bitexact": True, "cells": records,
              "source_sha256": source_before, "source_unchanged_during_preflight": True, "proxy_carrier_3round_bitexact": True,
              "scope": "Local sm86 full W1 compiledAMP integration, not T4/full5343/exactresume/utility proof"}
    (out / "smoke-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print("P20_REAL_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
