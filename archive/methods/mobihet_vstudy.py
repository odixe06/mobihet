"""Same existing objectives on the common clean diagnostic foldtrain/val branch."""
import torch
from core import grouped
from methods.mobihet_study import Method as Basic
from methods.mobihet_next import Method as Mixed
from scripts.run_p21_recovery import GuardedMethod


class GuardedBasic(Basic):
    def round(self, r, lr):
        if r == 1:
            compiled = self.env.allreduce(torch.tensor(int(grouped.TRAIN is not grouped.forward), device=self.env.dev))
            assert int(compiled) == self.env.world, "Every validation control requires compiled backend"
            GuardedMethod.record(self, {"event": "compile_guard", "compiled_ranks": int(compiled), "world": self.env.world})
        info = super().round(r, lr)
        assert info["skipped_frac"] == 0
        GuardedMethod.record(self, {"event": "round_finite", "round": r, "groups": 0, "clients": self.env.K})
        return info


def Method(env):
    assert env.hp["validation_diagnostic"] is True and env.hp["external_diagnostics"] is False
    if env.hp["study_mode"] in {"real", "augment"}: return GuardedBasic(env)
    assert env.hp["next_mode"] in {"mixed_input", "margin"}
    return Mixed(env)
