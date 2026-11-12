"""Module isolation, divergent-loss gradients, real-row Q and actual Q8 aggregation."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from core import grouped, local
from methods import mobihet_refine, mobihet_sim
from test_methods import ToyEnv


def environment(**extra):
    env = ToyEnv()
    for item in (Path(__file__).resolve().parents[1] / "runs/sim/core_hp.txt").read_text().split():
        import json
        key, val = item.split("=", 1)
        try:
            val = json.loads(val)
        except json.JSONDecodeError:
            pass
        env.hp[key] = val
    env.hp.update(batch=8, group_rows=32, amp=False, q_bits=8, synth_step_floor=0, **extra)
    return env


@pytest.mark.parametrize("kind", ["forward", "reverse", "symmetric", "js4"])
def test_loss_gradient_against_distribution_reference(kind):
    z = torch.randn(3, 16, dtype=torch.float64, requires_grad=True)
    q = torch.rand_like(z)
    q[:, 2] = 0  # Q8 has zero entries; reverse losses must stay finite.
    q /= q.sum(-1, keepdim=True)
    logp = z.log_softmax(-1)
    got = mobihet_refine.divergence(logp, q, kind)
    q = q.clamp_min(1e-6)
    if kind != "forward":
        q = q / q.sum(-1, keepdim=True)
    p = logp.exp()
    fwd = F.kl_div(logp, q, reduction="none").sum(-1)
    rev = F.kl_div(q.log(), p, reduction="none").sum(-1)
    m = (p + q) / 2
    js4 = 2 * (F.kl_div(m.log(), q, reduction="none").sum(-1) +
               F.kl_div(m.log(), p, reduction="none").sum(-1))
    expected = {"forward": fwd, "reverse": rev, "symmetric": (fwd + rev) / 2, "js4": js4}[kind]
    torch.testing.assert_close(got, expected)
    g1 = torch.autograd.grad(got.sum(), z, retain_graph=True)[0]
    g2 = torch.autograd.grad(expected.sum(), z)[0]
    torch.testing.assert_close(g1, g2)
    assert torch.isfinite(g1).all()


def test_defaults_preserve_cbb_loss_gradient_q_and_rng(monkeypatch):
    old = mobihet_sim.Method(environment())
    new = mobihet_refine.Method(environment())
    G, b, a = 2, 8, 64
    ctx = SimpleNamespace(Gfix=G, x=torch.zeros(G, b, 77), y=torch.arange(G * b).view(G, b),
                          clients=torch.arange(G), mask=torch.ones(G, b, dtype=torch.bool), P={}, B={})
    logits = torch.randn(G, b + a, 16, requires_grad=True)
    monkeypatch.setattr(grouped, "TRAIN", lambda *args, **kw: (logits, {}))
    lab = torch.arange(a).remainder(16)[None].expand(G, -1)
    for m in (old, new):
        m._synth = lambda ctx: (torch.zeros(G, a, 77), lab)
        m._shift = torch.zeros(G, 16)
        m._csum = torch.zeros(G, 16, 16)
        m.Q = torch.eye(16) * .9 + .1 / 16
    rng = torch.get_rng_state().clone()
    lo, ln = old._loss_stream(ctx)[0], new._loss_stream(ctx)[0]
    assert torch.equal(lo, ln) and torch.equal(old._csum, new._csum)
    assert torch.equal(torch.get_rng_state(), rng)
    assert torch.equal(torch.autograd.grad(lo.sum(), logits, retain_graph=True)[0],
                       torch.autograd.grad(ln.sum(), logits)[0])


@pytest.mark.parametrize("kind,eps", [("reverse", 0), ("symmetric", 0), ("js4", 0), ("forward", .05)])
def test_changed_loss_preserves_real_q_and_masks(monkeypatch, kind, eps):
    m = mobihet_refine.Method(environment(refine_loss=kind, real_smoothing=eps))
    G, b, a = 3, 8, 64
    mask = torch.ones(G, b, dtype=torch.bool)
    mask[1] = False  # active synthetic-only floor step
    mask[2] = False  # padded client
    ctx = SimpleNamespace(Gfix=G, x=torch.zeros(G, b, 77), y=torch.arange(G * b).view(G, b) % 16,
                          clients=torch.arange(G), mask=mask, active=torch.tensor([True, True, False]), P={}, B={})
    z = torch.randn(G, b + a, 16, requires_grad=True)
    m._synth = lambda ctx: (torch.zeros(G, a, 77), torch.arange(a).remainder(16)[None].expand(G, -1))
    monkeypatch.setattr(grouped, "TRAIN", lambda *args, **kw: (z, {}))
    m._shift, m._csum = torch.zeros(G, 16), torch.zeros(G, 16, 16)
    m.Q = torch.eye(16)
    loss = m._loss_stream(ctx)[0]
    expected = torch.zeros_like(m._csum)
    expected[0].index_add_(0, ctx.y[0], z[0, :b].detach().softmax(-1))
    torch.testing.assert_close(m._csum, expected)
    assert torch.isfinite(loss).all()
    grad = torch.autograd.grad(loss[:2].sum(), z)[0]
    assert torch.count_nonzero(grad[1, :b]) == 0 and torch.count_nonzero(grad[1, b:]) > 0


def test_sqrt_aggregation_uses_actual_q8_packets_and_is_resumable(monkeypatch):
    m = mobihet_refine.Method(environment(refine_agg="sqrt_count"))
    q = torch.rand(m.env.K, 16, 16)
    q /= q.sum(-1, keepdim=True)
    sums = q * m.cnt[..., None]

    def epoch(lay, flat, bufs, data, idx, *args, **kwargs):
        m._csum[:len(idx)] = sums[idx]
        return {"loss": torch.zeros(len(idx)), "applied": torch.ones(len(idx)), "skipped": torch.zeros(len(idx))}

    monkeypatch.setattr(local, "local_epoch", epoch)
    info = m._round_private(1, .001)
    counts = m.cnt[..., None]
    decoded = (((sums / counts.clamp_min(1)).clamp(0, 1) * 255).round() / 255).half().double()
    w = counts.double().sqrt()
    expected = ((decoded * w).sum(0) / w.sum(0).clamp_min(1)).float().half().float()
    expected /= expected.sum(-1, keepdim=True).clamp_min(1e-12)
    codes = (expected.clamp(0, 1) * 255).round()
    expected = codes / codes.sum(-1, keepdim=True).clamp_min(1)
    assert torch.equal(m.Q, expected)
    assert info["bytes_up"] > 34 and info["bytes_down"] > 32
    restored = mobihet_refine.Method(environment(refine_agg="sqrt_count"))
    restored.load_state(copy.deepcopy(m.state()))
    assert torch.equal(restored.Q, m.Q) and restored.q_agg == "sqrt_count"


def test_no_step_client_is_rejected_and_epoch_hook_restored(monkeypatch):
    m = mobihet_refine.Method(environment())

    def epoch(lay, flat, bufs, data, idx, *args, **kwargs):
        m._csum.zero_()
        return {"loss": torch.zeros(len(idx)), "applied": torch.zeros(len(idx)), "skipped": torch.ones(len(idx))}

    monkeypatch.setattr(local, "local_epoch", epoch)
    with pytest.raises(AssertionError, match="no optimizer update"):
        m._round_private(1, .001)
    assert local.local_epoch is epoch
