"""Independent target-routing checks; do not use method-produced targets as the expected loss."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from core import grouped
from methods.mobihet_next import Method
from scripts.q20_common import settings
from test_methods import ToyEnv


@pytest.mark.parametrize("mode", ["mixed_class", "mixed_input"])
def test_bank_half_really_uses_input_identity(mode, monkeypatch):
    env = ToyEnv(); env.hp.update(settings(mode)); model = Method(env)
    model.Q = .8 * torch.eye(16) + .2 / 16
    model.bank_target = torch.roll(model.Q[torch.arange(1024) % 16], 1, 1)
    model.bank_index = torch.arange(32)[None]
    labels = (torch.arange(64) % 16)[None]
    x = torch.zeros(1, 64, 77)
    monkeypatch.setattr(model, "_synth", lambda ctx: (x, labels))
    z = torch.linspace(-2, 2, 72 * 16).reshape(1, 72, 16).requires_grad_(True)
    monkeypatch.setattr(grouped, "TRAIN", lambda *a, **kw: (z, {}))
    model._shift = torch.zeros(1, 16); model._csum = torch.zeros(1, 16, 16)
    y = torch.arange(8)[None]
    ctx = SimpleNamespace(Gfix=1, x=torch.zeros(1, 8, 77), y=y, mask=torch.ones(1, 8, dtype=torch.bool),
                          clients=torch.tensor([0]), P=None, B=None)
    actual, _ = model._loss_private(ctx)
    target = model.Q[labels].clone()
    if mode == "mixed_input": target[:, 32:] = torch.roll(model.Q[labels[:, 32:]], 1, 2)
    expected = F.cross_entropy(z[:, :8].reshape(-1, 16), y.flatten())
    expected += (target * (target.log() - z[:, 8:].log_softmax(-1))).sum(-1).mean()
    torch.testing.assert_close(actual.sum(), expected, rtol=1e-6, atol=1e-6)
    grad = torch.autograd.grad(actual.sum(), z)[0]
    expected_grad = torch.zeros_like(z)
    expected_grad[:, :8] = (z[:, :8].softmax(-1) - F.one_hot(y, 16)) / 8
    expected_grad[:, 8:] = (z[:, 8:].softmax(-1) - target) / 64
    torch.testing.assert_close(grad, expected_grad, rtol=1e-5, atol=1e-7)


def test_sqrt_changes_only_input_aggregation_at_first_round(monkeypatch, tmp_path):
    monkeypatch.setenv("MOBIHET_RECOVERY_DIR", str(tmp_path))
    monkeypatch.setattr(grouped, "TRAIN", lambda *a, **kw: grouped.forward(*a, **kw))
    values = []
    for mode in ["mixed_input", "input_sqrt"]:
        env = ToyEnv(); env.hp.update(settings(mode)); model = Method(env)
        torch.manual_seed(120); model.round(1, .001); values.append(model)
    a, b = values
    assert torch.equal(a.priv_p, b.priv_p) and torch.equal(a.priv_b, b.priv_b) and torch.equal(a.Q, b.Q)
    assert torch.equal(a.class_sums, b.class_sums)
    assert torch.equal(a.aggregate_weights, a.cnt.double().sum(0)[a.bank_y])
    torch.testing.assert_close(b.aggregate_weights, b.cnt.double().sqrt().sum(0)[b.bank_y], rtol=1e-12, atol=1e-10)
    assert not torch.equal(a.bank_target, b.bank_target)
