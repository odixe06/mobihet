"""Reception replay must preserve source targets, lag and immutable-history identity."""
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from methods import mobihet_rstudy as replay
from methods.mobihet_next import signature
from methods.mobihet_study import reception


def test_frozen_history_independent_source_and_tamper():
    from methods import mobihet_rstudy_data as frozen
    batch = json.loads(Path("runs/q-study/batch.json").read_text())
    for seed, spec in zip((42, 43), batch[6:8]):
        history = replay.load_schedule(seed, "cpu", frozen.SHA256)
        for r in (1, 10, 20):
            path = Path("artifacts/q20") / spec["slug"] / spec["slug"] / "models" / f"r{r:03}.pt"
            source = torch.load(path, weights_only=True, map_location="cpu")
            assert torch.equal(source["Q"], history["Q"][r - 1])
            assert torch.equal(source["bank_target"], history["target"][r - 1])
            assert replay.bank_sha(source["bank_X"]) == history["banks"][r - 1]
    with pytest.raises(AssertionError): replay.load_schedule(42, "cpu", "0" * 64)


def test_replay_round_preserves_live_and_replaces_only_downlink():
    model = object.__new__(replay.Replay)
    q = torch.eye(16) * .8 + .2 / 16
    x = torch.arange(32 * 77).reshape(32, 77).half()
    target = q[torch.arange(32) % 16]
    model.variant = "margin"; model.version = 0; model.Q = None; model.source_sha = "fixture"
    model.history = {"Q": q[None], "target": target[None], "body": target.half()[None], "banks": [replay.bank_sha(x)]}
    model.env = type("Env", (), {"world": 1})()
    def live(self, r, lr):
        self.version = r; self.Q = q.flip(0); self.bank_X = x; self.bank_y = torch.arange(32) % 16
        self.bank_target = target.flip(0); self.bank_body = self.bank_target.half()
        return {"teacher_produced_sha256": "live", "teacher_received_agreement": .125}
    with patch.object(replay.Live, "round", live):
        info = model.round(1, .001)
        assert torch.equal(model.Q, q) and torch.equal(model.Q_live, q.flip(0))
        assert torch.equal(model.bank_target, target) and torch.equal(model.bank_target_live, target.flip(0))
        assert info["live_teacher_produced_sha256"] == "live"
        assert info["teacher_produced_sha256"] == signature(q, x, target, "margin")
        bad = copy.deepcopy(model); bad.version = 0; bad.history["banks"] = ["bad"]
        with pytest.raises(AssertionError): bad.round(1, .001)
    # Wrong teacher at r2 must fail before training.
    model.history["Q"] = q[None].expand(2, -1, -1)
    model.Q = q.flip(0)
    with pytest.raises(AssertionError): model.round(2, .001)


def test_strength_uses_same_source_mass_as_margin():
    gen = torch.Generator().manual_seed(23)
    q = torch.rand(16, 16, generator=gen); q += torch.eye(16)
    q /= q.sum(1, keepdim=True)
    counts = torch.randint(0, 12, (11, 16), generator=gen).float()
    a, b = reception(q, counts, "margin"), reception(q, counts, "strength")
    torch.testing.assert_close((counts * a).sum(1), (counts * b).sum(1))
