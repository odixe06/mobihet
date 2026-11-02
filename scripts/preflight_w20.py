"""Both-seed real W1 fixture: W knobs at default are bit-exact with the audited Q/R code; every W cell runs clean.

Local sm86, 2 clients/173 train rows, 3 rounds: not T4/full5343/utility proof.
"""
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
import torch
from core import fl, grouped, local
from core.dagsnet import DAGSNet, width
from methods import mobihet_rstudy as replay
from methods import mobihet_rstudy_data as frozen
from methods import mobihet_wstudy as wstudy
from methods.mobihet_next import Method as Live
from scripts.preflight_q20 import equal_state
from scripts.preflight_r20 import capture, same
from scripts.q20_common import settings
from scripts.run_fl import PROTOCOL
from scripts.w20_common import CELLS, SOURCES

OUT = ROOT / "runs/w-study"


def lr(r):
    return fl.cosine_lr(r, 50, .001, .00001)


def run(model, seed, rounds=3):
    torch.manual_seed(seed)
    rows = [model.round(r, lr(r)) for r in range(1, rounds + 1)]
    assert all(row["skipped_frac"] == 0 for row in rows)
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    before = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in SOURCES}
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_wstudy.py", "tests/test_r20.py", "tests/test_mobihet_next.py",
        "tests/test_units.py", "--junitxml=" + str(OUT / "tests.xml")], cwd=ROOT, check=True)
    parent = json.loads((ROOT / "runs/q-study/batch.json").read_text())[0]
    reports = []
    for seed in (42, 43):
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
        os.environ["MOBIHET_RECOVERY_DIR"] = str(OUT / f"smoke-s{seed}/recovery")

        def fresh(**knobs):
            for key in ("w_real", "w_synth", "w_boot", "synth_step_floor", "r_source_sha256"): env.hp.pop(key, None)
            env.hp.update(knobs); env.gen.set_state(initial)

        # 1. defaults reproduce the audited live code, stream and real-KD paths
        for mode in ("stream", "real"):
            fresh(**settings(mode), r_kind="live"); reference = Live(env); torch.manual_seed(seed); expected = []
            for r in (1, 2, 3):
                reference.round(r, lr(r)); expected.append(capture(reference))
            del reference; gc.collect(); torch.cuda.empty_cache()
            fresh(**settings(mode), r_kind="live"); model = wstudy.Method(env); torch.manual_seed(seed)
            for r in (1, 2, 3):
                model.round(r, lr(r)); same(model, expected[r - 1])
            reports.append({"check": "default_equals_live", "mode": mode, "seed": seed, "bitexact": True})
            del model, expected; gc.collect(); torch.cuda.empty_cache()
        # fixture teacher history for replay (real fixture, not the frozen Q6/Q7 schedule)
        fresh(**settings("mixed_input"), r_kind="live"); reference = Live(env); torch.manual_seed(seed)
        history = {key: [] for key in ("Q", "target", "body", "banks")}
        for r in (1, 2, 3):
            reference.round(r, lr(r))
            for key, value in (("Q", reference.Q), ("target", reference.bank_target), ("body", reference.bank_body)):
                history[key].append(value.clone())
            history["banks"].append(replay.bank_sha(reference.bank_X))
        for key in ("Q", "target", "body"): history[key] = torch.stack(history[key])
        del reference; gc.collect(); torch.cuda.empty_cache()
        fresh(**settings("real"), r_kind="replay", r_source_sha256=frozen.SHA256)
        with patch.object(replay, "load_schedule", return_value=history): reference = replay.Method(env)
        torch.manual_seed(seed); expected = []
        for r in (1, 2, 3):
            reference.round(r, lr(r)); expected.append(capture(reference))
        del reference; gc.collect(); torch.cuda.empty_cache()
        fresh(**settings("real"), r_kind="replay", r_source_sha256=frozen.SHA256)
        with patch.object(replay, "load_schedule", return_value=history): model = wstudy.Method(env)
        torch.manual_seed(seed)
        for r in (1, 2, 3):
            model.round(r, lr(r)); same(model, expected[r - 1])
        reports.append({"check": "default_equals_replay", "mode": "real", "seed": seed, "bitexact": True})
        del model, expected; gc.collect(); torch.cuda.empty_cache()

        # 2. every W cell: finite, applied, state roundtrip, strict rebuild, diagnostics leave state/RNG unchanged
        for cell, spec in CELLS.items():
            knobs = dict(settings(spec["mode"]), r_kind=spec["r_kind"], **spec["w"])
            if spec["r_kind"] == "replay": knobs["r_source_sha256"] = frozen.SHA256
            fresh(**knobs)
            with patch.object(replay, "load_schedule", return_value=history): model = wstudy.Method(env)
            applied, epoch = [], local.local_epoch
            def spy(*args, **kwargs):
                stats = epoch(*args, **kwargs); applied.append(stats["applied"].clone()); return stats
            with patch.object(local, "local_epoch", spy): rows = run(model, seed)
            for key in ("priv_p", "priv_b", "Q"): assert torch.isfinite(getattr(model, key)).all(), (cell, key)
            floor = spec["w"].get("synth_step_floor", 0)
            # fixture clients have 1-3 real steps: with a floor every client takes exactly `floor` steps per round
            assert all(bool((a == floor).all()) if floor else bool((a <= 3).all()) for a in applied), (cell, applied)
            if spec["w"].get("w_boot") == "compact":
                assert model.bootstrap_bytes["up"] < 300_000 and model.bootstrap_bytes["down"] < 800_000
                assert (model.cop_mass >= 0).all() and torch.all(model.cop_cdf[..., -1] <= 1)
            state = copy.deepcopy(model.state()); rng = (env.gen.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())
            folder = OUT / f"smoke-{cell}-s{seed}"; folder.mkdir(exist_ok=True)
            model.diagnostics(3, folder); equal_state(state, model.state())
            assert all(torch.equal(a, b) for a, b in zip(rng, (env.gen.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state())))
            with patch.object(replay, "load_schedule", return_value=history): restored = wstudy.Method(env)
            restored.load_state(state); equal_state(state, restored.state())
            single = DAGSNet(env.cfg); single.load_state_dict(env.lay.state_dict(restored.priv_p[0], restored.priv_b[0]), strict=True)
            assert sum(p.numel() for p in single.parameters()) == 395408
            reports.append({"check": "cell", "cell": cell, "seed": seed, "rounds": 3, "finite": True, "skip0": True,
                "bootstrap_bytes": model.bootstrap_bytes, "train_loss_r3": rows[-1]["train_loss"],
                "state_roundtrip": True, "diagnostic_state_rng_unchanged": True, "strict_rebuild": True})
            print("W20_CELL_PASS", seed, cell, rows[-1]["train_loss"], flush=True)
            del model, restored, state, single; gc.collect(); torch.cuda.empty_cache()
    assert before == {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in SOURCES}
    (OUT / "preflight.json").write_text(json.dumps({"source_sha256": before, "source_unchanged": True, "reports": reports,
        "scope": "Real v4 W1 local sm86 173train/46test, 3 rounds; not T4/full5343/utility/fullresume proof"}, indent=2) + "\n")
    print("W20_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
