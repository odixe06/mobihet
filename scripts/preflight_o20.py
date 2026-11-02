"""Real-data compile/replay integration gate; no utility or full resume claim."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from core import fl, grouped, local, metrics
from core.dagsnet import width, DAGSNet
from methods.mobihet import Method as LiveMethod
from methods.mobihet_replay import Method, frozen, source_schedule
from scripts.run_fl import PROTOCOL


def main():
    out = Path("runs/o-study"); out.mkdir(exist_ok=True)
    counts = np.load("evidence/research-data/train_counts.npy")
    ids = [613, 2029]  # Established N train-only fixture:173trainrows, all16classes.
    cache = out / "smoke-cache"; cache.mkdir(exist_ok=True)
    np.save(cache / "train_sizes.npy", counts.sum(1))
    hp = dict(PROTOCOL, arm="none", synth="copula", synth_rows=64, synth_target="soft", kd_weight="fixed",
              q_codec="mean_fp16", q_collect="online", q_agg="count", kd_support="off", group_rows=128,
              real_kd=False, real_kd_gate=False, replay_mode="full", replay_source_sha256=frozen.SHA256)
    env = fl.Env(width(1), hp, clients=ids, cache=str(cache))
    assert len(env.train.Y) == 173 and len(env.test.Y) == 46
    assert set(env.train.Y.cpu().tolist()) == set(range(16))
    assert env.dev.type == "cuda", "GPU compile/backward preflight required"
    grouped.compile_all()
    ok, report = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
    assert ok, report
    for seed in (42, 43): source_schedule(seed, env.dev, frozen.SHA256)
    rng, global_rng = env.gen.get_state(), torch.cuda.get_rng_state()
    r1reference, cells = None, []
    for real, mode in ((False, "full"), (True, "full"), (False, "uniform"), (False, "permuted")):
        hp.update(real_kd=real, real_kd_gate=real, replay_mode=mode)
        env.gen.set_state(rng); torch.cuda.set_rng_state(global_rng)
        m = Method(env); rows = []
        for r in (1, 2, 3):
            before_p, before_b = m.priv_p.clone(), m.priv_b.clone()
            before_g, before_s = env.gen.get_state(), m.gen_g.get_state()
            before_cuda = torch.cuda.get_rng_state()
            # Manual teacher injection through original training path, using identical data/sampler/dropout RNG.
            control = LiveMethod(env)
            control.priv_p, control.priv_b = before_p, before_b
            control.version = r - 1
            control.Q = None if r == 1 else m.replay_target[r - 2].clone()
            control.gen_g.set_state(before_s)
            env.gen.set_state(before_g); torch.cuda.set_rng_state(before_cuda)
            control.round(r, fl.cosine_lr(r, 50, 1e-3, 1e-5))
            env.gen.set_state(before_g); torch.cuda.set_rng_state(before_cuda)
            info = m.round(r, fl.cosine_lr(r, 50, 1e-3, 1e-5))
            assert torch.equal(control.priv_p, m.priv_p) and torch.equal(control.priv_b, m.priv_b)
            assert torch.equal(control.Q, m.Q_live)
            assert info["skipped_frac"] == 0 and np.isfinite(info["train_loss"])
            assert torch.isfinite(m.priv_p).all() and torch.isfinite(m.priv_b).all()
            cm = m.evaluate()["private"]
            assert cm.sum() == 46 and info["bytes_up"] == info["bytes_down"] == 544
            rows.append({"round": r, "metrics": metrics.ten(cm.sum(0)), **info})
            if r == 1:
                if r1reference is None: r1reference = (m.priv_p.clone(), m.priv_b.clone())
                else:
                    assert all(torch.equal(got, want) for got, want in zip((m.priv_p, m.priv_b), r1reference))
            del control
        path = out / f"smoke-{real}-{mode}.pt"; torch.save(m.state(), path)
        restored = Method(env); restored.load_state(torch.load(path, map_location=env.dev, weights_only=True))
        model = DAGSNet(env.cfg)
        model.load_state_dict(env.lay.state_dict(restored.priv_p[0].cpu(), restored.priv_b[0].cpu()), strict=True)
        assert sum(p.numel() for p in model.parameters()) == 395408
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        assert np.array_equal(restored.evaluate()["private"], cm)
        cells.append({"real_kd": real, "replay_mode": mode, "rows": rows,
                      "state_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        del m, restored
    (out / "smoke-audit.json").write_text(json.dumps({"compile_gate": report, "ids": ids, "train_rows": 173,
        "test_rows": 46, "r1_four_cells_bitexact": True, "manual_injection_bitexact": True,
        "source_sha256": frozen.SHA256, "cells": cells,
        "replay_code_sha256": hashlib.sha256(Path("methods/mobihet_replay.py").read_bytes()).hexdigest(),
        "scope": "Real2client compiledAMPsm86 integration; not T4/utility/fullresume proof"}, indent=2) + "\n")
    print("O20_REAL_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
