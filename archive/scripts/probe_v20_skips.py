"""One-round numerical diagnosis only: unchanged CE training, per-client step counters.

No guard threshold is relaxed for an accepted V20 result. This probe records the
first epoch that the old zero-skip guard discarded; it exports no private weights.
"""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from core import fl, grouped, local
from core.dagsnet import width
from methods.mobihet_study import Method as Basic
from scripts.run_fl import PROTOCOL
from scripts.run_v20 import DiagnosticEnv


class ObservedBasic(Basic):
    def round(self, r, lr):
        groups = []
        original = local.local_epoch

        def observed(lay, flat, bufs, data, clients, *args, **kwargs):
            stat = original(lay, flat, bufs, data, clients, *args, **kwargs)
            indices = clients.detach().cpu().tolist()
            groups.append({"client_ids": self.env.ids[indices].tolist(),
                "applied": stat["applied"].detach().cpu().long().tolist(),
                "skipped": stat["skipped"].detach().cpu().long().tolist(),
                "weights_finite": bool(torch.isfinite(flat).all()),
                "buffers_finite": bool(torch.isfinite(bufs).all()),
                "loss_finite": bool(torch.isfinite(stat["loss"]).all())})
            return stat

        with patch.object(local, "local_epoch", observed): info = super().round(r, lr)
        ids = [i for g in groups for i in g["client_ids"]]
        assert sorted(ids) == sorted(self.env.ids.tolist()) and len(ids) == self.env.K
        clients = [{"client_id": cid, "applied": applied, "skipped": skipped}
            for g in groups for cid, applied, skipped in zip(g["client_ids"], g["applied"], g["skipped"])]
        self.observation = {"rank": self.env.rank, "round": r, "clients": clients,
            "applied": sum(c["applied"] for c in clients), "skipped": sum(c["skipped"] for c in clients),
            "zero_applied_client_ids": [c["client_id"] for c in clients if c["applied"] == 0],
            "groups": groups, "all_finite": all(g[k] for g in groups for k in ("weights_finite", "buffers_finite", "loss_finite")),
            "global_info": info, "loss_scale": self.env.hp.get("loss_scale", 2. ** 12),
            "scope": "One-epoch diagnostic, no evaluation or teacher selection; not accepted V20 training or dynamic GradScaler warm-up proof"}
        return info


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, choices=(42, 43), required=True)
    ap.add_argument("--max-hours", type=float, required=True); ap.add_argument("--cache", default="")
    ap.add_argument("--parent-arm", choices=("V0", "V1"), required=True)
    ap.add_argument("--set", nargs="*", required=True)
    a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    hp = dict(PROTOCOL)
    for item in a.set:
        key, value = item.split("=", 1)
        try: hp[key] = json.loads(value)
        except json.JSONDecodeError: hp[key] = value
    assert hp["seed"] == a.seed and hp["study_mode"] == "real" and hp["synth"] == "off" and hp["no_kd"]
    assert hp["group_rows"] == 4096 and hp["batch"] == 64 and hp["amp"]
    env = DiagnosticEnv(width(1), hp, cache=a.cache or None)
    model = ObservedBasic(env)
    grouped.compile_all()
    ok, gate = local.compile_gate(env.lay, env.train, env.cfg, hp, torch.Generator(device=env.dev).manual_seed(0))
    print(f"compile gate rank{env.rank}: {'PASS' if ok else 'FAIL'} {gate}", flush=True)
    compiled_ranks = int(env.allreduce(torch.tensor(int(ok), device=env.dev)))
    assert compiled_ranks == env.world == 2
    (a.out / f"fold_rank{env.rank}.json").write_text(json.dumps(env.fold_receipt, indent=2) + "\n")
    run = None
    if env.rank == 0:
        import wandb
        run = wandb.init(project="mobihet-v4", name=a.out.name, id=a.out.name,
            config={"diagnostic_only": True, "hp": hp, "clients": len(env.all_ids), "world": env.world,
                    "backend": "compiled", "parent_arm": a.parent_arm, "n_params": env.lay.P})
    info = model.round(1, fl.cosine_lr(1, 50, .001, .00001))
    model.observation.update(compile_gate=gate, compiled_ranks=compiled_ranks, world=env.world,
        n_params=env.lay.P, hp=hp, parent_arm=a.parent_arm)
    (a.out / f"numerics_rank{env.rank}.json").write_text(json.dumps(model.observation, indent=2) + "\n")
    if run:
        run.log({"round": 1, **{k: v for k, v in info.items() if isinstance(v, (int, float))}})
        run.finish()
    torch.distributed.barrier()
    print("V20_NUMERICS_EPOCH_RECORDED", env.rank, flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__": main()
