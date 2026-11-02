"""Observation must expose skipped/all-skipped clients, not classify them as healthy."""
from types import SimpleNamespace
import numpy as np
import torch
from core import local
from methods.mobihet_study import Method as Basic
from scripts.probe_v20_skips import ObservedBasic


def test_nonzero_skip_and_no_applied_client_are_reported(monkeypatch):
    model = object.__new__(ObservedBasic)
    model.env = SimpleNamespace(ids=np.array([3, 7]), K=2, rank=0, hp={})
    stat = {"applied": torch.tensor([2., 0.]), "skipped": torch.tensor([1., 1.]), "loss": torch.ones(2)}
    original = lambda *args, **kwargs: stat
    monkeypatch.setattr(local, "local_epoch", original)
    def round_(self, r, lr):
        result = local.local_epoch(None, torch.ones(2, 2), torch.ones(2, 2), None, torch.tensor([0, 1]))
        assert result is stat
        return {"skipped_frac": .5}
    monkeypatch.setattr(Basic, "round", round_)
    assert model.round(1, .001) == {"skipped_frac": .5}
    assert local.local_epoch is original
    assert model.observation["skipped"] == 2 and model.observation["zero_applied_client_ids"] == [7]
    assert model.observation["loss_scale"] == 4096


def test_observer_is_restored_when_training_raises(monkeypatch):
    import pytest
    original = local.local_epoch
    model = object.__new__(ObservedBasic)
    monkeypatch.setattr(Basic, "round", lambda *args: (_ for _ in ()).throw(FloatingPointError("probe")))
    with pytest.raises(FloatingPointError): model.round(1, .001)
    assert local.local_epoch is original
