"""Local codec and 2-client training smoke; not a full-network utility experiment."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from core import fl, grouped, local
from core.dagsnet import width
from methods.mobihet_compression import Method
from scripts.q20_common import settings
from scripts.run_fl import PROTOCOL


def q8(q):
    body = torch.round(q * 255).to(torch.uint8)
    decoded = body.float()
    return body, decoded / decoded.sum(-1, keepdim=True).clamp_min(1)


def main():
    out = ROOT / "runs/core-compression"
    out.mkdir(parents=True, exist_ok=True)
    sources = ["methods/mobihet_compression.py", "methods/mobihet.py", "methods/mobihet_wstudy.py", "methods/mobihet_next.py"]
    before = {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources}
    # Existing core trajectories: downlink target only, not reconstructed uplink packets.
    records = []
    for slug in ("mh-x0-core-s42", "mh-x1-core-s43"):
        for p in sorted((ROOT / "artifacts/x20" / slug).rglob("models/r*.pt")):
            q = torch.load(p, map_location="cpu", weights_only=False)["Q"]
            if q is None:
                continue
            raw = q.half().numpy().astype("<f2").tobytes()
            compressed = zlib.compress(raw, 6)
            assert zlib.decompress(compressed) == raw
            body, decoded = q8(q)
            assert torch.isfinite(decoded).all() and (decoded >= 0).all()
            records.append(dict(path=str(p.relative_to(ROOT)), sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                fp16_body=len(raw), zlib_body=len(compressed), uint8_body=body.numel(),
                                max_abs=float((q-decoded).abs().max()),
                                row_l1_max=float((q-decoded).abs().sum(-1).max()),
                                argmax_changes=int((q.argmax(-1) != decoded.argmax(-1)).sum())))
    assert len(records) == 40, len(records)
    (out / "q-codec.json").write_text(json.dumps(records, indent=2) + "\n")

    class Probe(Method):
        def _copula_bootstrap(self, client, NB=256):
            super()._copula_bootstrap(client, NB)
            if self.env.hp["probe"] == "lossless":
                raw = self.cop_cdf.cpu().numpy().tobytes()
                restored = np.frombuffer(zlib.decompress(zlib.compress(raw)), dtype=np.float32).reshape(self.cop_cdf.shape).copy()
                self.cop_cdf = torch.from_numpy(restored).to(self.env.dev)

    reports = []
    for seed in (42,):
        hp = dict(PROTOCOL, **{})
        hp.update(settings("stream"), seed=seed, group_rows=128, r_kind="live", w_boot="compact",
                  synth_step_floor=16, external_diagnostics=False, diagnostic_rounds=[])
        env = fl.Env(width(1), hp, clients=[613, 2029], cache=str(ROOT / "runs/n-study/smoke-cache"))
        grouped.compile_all()
        ok, gate = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
        assert ok, gate
        initial = env.gen.get_state().clone()
        control = None
        for arm in ("control", "lossless", "bins64", "q8", "combo"):
            env.hp.update(probe=arm, compression_bins=64 if arm in {"bins64", "combo"} else 256, compression_q8=arm in {"q8", "combo"})
            os.environ["MOBIHET_RECOVERY_DIR"] = str(out / f"smoke-{arm}" / "recovery")
            env.gen.set_state(initial)
            torch.manual_seed(seed)
            model = Probe(env)
            rows = [model.round(r, fl.cosine_lr(r, 50, .001, .00001)) for r in (1, 2, 3)]
            assert all(row["skipped_frac"] == 0 for row in rows)
            assert all(torch.isfinite(getattr(model, k)).all() for k in ("priv_p", "priv_b", "Q"))
            state = [getattr(model, k).cpu().clone() for k in ("priv_p", "priv_b", "Q")]
            if arm == "control":
                control = state
            if arm == "lossless":
                assert all(torch.equal(a, b) for a, b in zip(control, state))
            reports.append(dict(seed=seed, arm=arm, rounds=3, train_rows=len(env.train.Y),
                                finite=True, skipped_frac=0, 
                                bitexact_control=all(torch.equal(a, b) for a, b in zip(control, state))))
            del model, state
            gc.collect()
            torch.cuda.empty_cache()
        del env
    assert before == {n: hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in sources}
    (out / "source.json").write_text(json.dumps(before, indent=2) + "\n")
    (out / "smoke.json").write_text(json.dumps(reports, indent=2) + "\n")
    print(json.dumps(dict(q_snapshots=len(records), q8_max_abs=max(r["max_abs"] for r in records),
                          q8_argmax_changes=sum(r["argmax_changes"] for r in records), smoke=reports), indent=2))


if __name__ == "__main__":
    main()
