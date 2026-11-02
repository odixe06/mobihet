"""Guard regression and real v4 seed43 three-round finite-path equivalence, no HP changes."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from core import fl, grouped, local
from core.dagsnet import width
from methods.mobihet_study import Method
from scripts.run_fl import PROTOCOL
from scripts.run_p21_recovery import GuardedMethod

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p-study/recovery-p21"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = next(s for s in json.loads((ROOT / "runs/p-study/batch.json").read_text()) if s["arm"] == "P21")
    names = list(spec["overlay_sha256"]) + ["scripts/run_p21_recovery.py", "scripts/preflight_p21_recovery.py"]
    before = {name: digest(ROOT / name) for name in names}
    assert all(before[name] == h for name, h in spec["overlay_sha256"].items())
    hp = dict(PROTOCOL)
    for item in spec["hp"]:
        k, v = item.split("=", 1)
        try: hp[k] = json.loads(v)
        except json.JSONDecodeError: hp[k] = v
    hp.update(group_rows=128, diagnostic_ids=[613, 2029], diagnostic_rounds=[1, 2, 3], external_diagnostics=False)
    env = fl.Env(width(1), hp, clients=[613, 2029], cache=str(ROOT / "runs/n-study/smoke-cache"))
    assert len(env.train.Y) == 173 and len(env.test.Y) == 46
    grouped.compile_all()
    ok, report = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
    assert ok, report
    os.environ["MOBIHET_RECOVERY_DIR"] = str(OUT / "local-guard")
    initial_gen = env.gen.get_state().clone()
    references = []
    for cls in (Method, GuardedMethod):
        env.gen.set_state(initial_gen)
        model = cls(env)
        torch.manual_seed(43)
        for r in (1, 2, 3):
            info = model.round(r, fl.cosine_lr(r, 50, .001, .00001))
            assert info["skipped_frac"] == 0 and np.isfinite(info["train_loss"])
            state = copy.deepcopy(model.state())
            if cls is Method:
                references.append((state, env.gen.get_state().clone(), model.evaluate()["private"].copy(), info))
            else:
                reference, gen, cm, old_info = references[r - 1]
                assert reference.keys() == state.keys()
                for key, value in state.items():
                    if torch.is_tensor(value): assert torch.equal(value, reference[key]), (r, key)
                    else: assert value == reference[key], (r, key)
                assert torch.equal(gen, env.gen.get_state())
                assert np.array_equal(cm, model.evaluate()["private"])
                for key in info:
                    if "seconds" not in key: assert info[key] == old_info[key], (r, key)
            print("P21_EQUIVALENCE_ROUND", cls.__name__, r, flush=True)
        if cls is GuardedMethod:
            with patch.object(grouped, "TRAIN", grouped.forward):
                try: model.round(1, .001)
                except AssertionError as exc: assert "every rank compile gate" in str(exc)
                else: raise AssertionError("Eager backend accepted")
            with patch.object(env, "allreduce", return_value=torch.tensor(0, device=env.dev)):
                try: model.round(1, .001)
                except AssertionError as exc: assert "every rank compile gate" in str(exc)
                else: raise AssertionError("Peer compile failure accepted")
            model.guard_active, model.guard_round, model.guard_calls = True, 1, 0
            index = env.groups[0]
            with patch.object(Method, "_probs", return_value=torch.full((len(index), 1024, 16), float("nan"), device=env.dev)):
                try: model._probs(model.priv_p[index], model.priv_b[index], model.bank_X, len(index), chunk=128)
                except FloatingPointError: pass
                else: raise AssertionError("Nonfinite packet accepted")
        del model
        gc.collect(); torch.cuda.empty_cache()
    records = [json.loads(line) for line in (OUT / "local-guard/rank0.jsonl").read_text().splitlines()]
    assert sum(v["event"] == "round_finite" for v in records) == 3
    bad = [v for v in records if v["event"] == "nonfinite_probability"]
    assert len(bad) == 1 and bad[0]["fp32_eager_probability"]["nonfinite"] == 0
    assert {name: digest(ROOT / name) for name in names} == before
    (OUT / "preflight.json").write_text(json.dumps({"source_sha256": before, "seed": 43, "rounds": 3,
        "train_rows": 173, "test_rows": 46, "compile_gate": report, "finite_training_bitexact": True,
        "rng_and_metrics_bitexact": True, "eager_or_peer_failure_rejected": True, "nonfinite_packet_rejected": True,
        "scope": "Local sm86 real v4 fixture; T4 probe1round passed previously, original failure not reproduced; no numerical repair proof"}, indent=2) + "\n")
    print("P21_RECOVERY_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
