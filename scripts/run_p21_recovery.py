"""P21 fresh recovery: unchanged finite training, strict compiled gate, failure metadata only."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from core import grouped
from methods import mobihet_study


def summary(value):
    finite = torch.isfinite(value)
    return {"shape": list(value.shape), "nonfinite": int((~finite).sum()),
            "absmax_finite": float(torch.where(finite, value.abs(), 0).max())}


class GuardedMethod(mobihet_study.Method):
    def record(self, value):
        folder = Path(os.environ["MOBIHET_RECOVERY_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / f"rank{self.env.rank}.jsonl").open("a") as file:
            file.write(json.dumps({"rank": self.env.rank, **value}) + "\n")
            file.flush()

    def round(self, r, lr):
        env = self.env
        if r == 1:
            compiled = torch.tensor(int(grouped.TRAIN is not grouped.forward), device=env.dev)
            count = int(env.allreduce(compiled))
            self.record({"event": "compile_guard", "compiled_ranks": count, "world": env.world})
            assert count == env.world, "P21 recovery requires every rank compile gate PASS; refuse eager fallback"
        self.guard_round, self.guard_calls = r, 0
        self.guard_active = True
        try:
            info = super().round(r, lr)
        finally:
            self.guard_active = False
        assert self.guard_calls == len(env.groups), "Unexpected number of private scoring groups"
        self.record({"event": "round_finite", "round": r, "groups": self.guard_calls, "clients": env.K})
        return info

    @torch.no_grad()
    def _probs(self, flat, bufs, xs, G, chunk=512):
        probability = super()._probs(flat, bufs, xs, G, chunk)
        if not getattr(self, "guard_active", False):
            return probability
        index = self.env.groups[self.guard_calls]
        self.guard_calls += 1
        if bool(torch.isfinite(probability).all()):
            return probability
        env = self.env
        row = {"event": "nonfinite_probability", "round": self.guard_round, "group": self.guard_calls,
               "client_ids": env.ids[index].tolist(), "probability": summary(probability),
               "parameters": summary(flat), "buffers": summary(bufs), "synthetic": summary(xs)}
        bad = (~torch.isfinite(probability)).any(2).any(1).cpu().numpy()
        row["affected_client_ids"] = env.ids[index][bad].tolist()
        # Counterfactual inference only after failure, at the same state and synthetic inputs.
        try:
            results = []
            with torch.autocast("cuda", enabled=False):
                for begin in range(0, len(xs), chunk):
                    x = xs[begin:begin + chunk][None].expand(G, -1, -1).float()
                    logits, _ = grouped.forward(env.lay.params(flat), env.lay.bufs(bufs), x, env.cfg, train=False)
                    results.append(logits.softmax(-1))
            row["fp32_eager_probability"] = summary(torch.cat(results, 1))
        except Exception as exc:
            row["fp32_diagnostic_error"] = type(exc).__name__
        self.record(row)
        raise FloatingPointError("P21 nonfinite private probabilities; metadata saved; no packet admitted")


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", required=True)
    args, _ = parser.parse_known_args()
    os.environ["MOBIHET_RECOVERY_DIR"] = str(Path(args.out) / "recovery")
    mobihet_study.Method = GuardedMethod
    from scripts import run_fl
    run_fl.main()


if __name__ == "__main__":
    main()
