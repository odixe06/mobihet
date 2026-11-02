"""CE-only V20 recovery: unchanged optimizer, explicit per-client numerical evidence."""
import math
import os
from pathlib import Path
import json

import torch
from core import grouped
from scripts.probe_v20_skips import ObservedBasic
from scripts.run_p21_recovery import GuardedMethod


def admissible(observation, info):
    """Match common audit's 1% ceiling, and refuse untrained clients/nonfinite state."""
    fraction = info["skipped_frac"]
    return (math.isfinite(fraction) and 0 <= fraction <= .01 and math.isfinite(info["train_loss"])
            and observation["all_finite"] and not observation["zero_applied_client_ids"])


class Method(ObservedBasic):
    def __init__(self, env):
        assert env.hp["validation_diagnostic"] is True and env.hp["external_diagnostics"] is False
        assert env.hp["study_mode"] == "real" and env.hp["no_kd"] and env.hp["synth"] == "off"
        super().__init__(env)

    def round(self, r, lr):
        env = self.env
        if r == 1:
            count = int(env.allreduce(torch.tensor(int(grouped.TRAIN is not grouped.forward), device=env.dev)))
            GuardedMethod.record(self, {"event": "compile_guard", "compiled_ranks": count, "world": env.world})
            assert count == env.world, "Every recovery rank must pass compiled gate"
        info = super().round(r, lr)
        totals = env.allreduce(torch.tensor([self.observation["applied"], self.observation["skipped"]],
            dtype=torch.int64, device=env.dev))
        info.update(applied_steps=int(totals[0]), skipped_steps=int(totals[1]))
        accepted = int(env.allreduce(torch.tensor(int(admissible(self.observation, info)), device=env.dev)))
        self.observation.update(guard_policy="common_global_skip_le_0.01_finite_positive_applied_per_client",
                                accepted_ranks=accepted, world=env.world)
        folder = Path(os.environ["MOBIHET_RECOVERY_DIR"]).parent
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"numerics_rank{env.rank}_r{r:03}.json").write_text(json.dumps(self.observation, indent=2) + "\n")
        assert accepted == env.world, "Numerical guard failed; per-client evidence saved on each rank"
        GuardedMethod.record(self, {"event": "round_finite", "round": r, "groups": 0, "clients": env.K,
            "skipped_frac": info["skipped_frac"], "applied_steps_global": info["applied_steps"],
            "skipped_steps_global": info["skipped_steps"]})
        return info
