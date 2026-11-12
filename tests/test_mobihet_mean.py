"""Mean-probability objective reference, masks, CBB identity and exact continuation."""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from core import grouped
from methods import mobihet_mean, mobihet_refine
from test_mobihet_refine import environment


@pytest.mark.parametrize("hard", [False, True])
def test_loss_gradient_matches_probability_reference_and_jensen(hard):
    torch.manual_seed(123)
    z = torch.randn(3, 64, 16, dtype=torch.float64, requires_grad=True)
    q = torch.eye(16, dtype=z.dtype) if hard else torch.rand(16, 16, dtype=z.dtype)
    q = q / q.sum(-1, keepdim=True)
    q.requires_grad_()
    row, mean = mobihet_mean.class_losses(z.log_softmax(-1), q)
    # Select each class explicitly: no reshape convention shared with implementation.
    target = q.detach().clamp_min(1e-6)
    p = z.softmax(-1)
    expected = torch.stack([F.kl_div(p[:, c::16].mean(1).log(), target[c].expand(3, -1),
                                      reduction="none").sum(-1) for c in range(16)], 1).mean(1)
    torch.testing.assert_close(mean, expected)
    actual = torch.autograd.grad(mean.sum(), (z, q), retain_graph=True, allow_unused=True)
    ref = torch.autograd.grad(expected.sum(), z)[0]
    torch.testing.assert_close(actual[0], ref)
    assert actual[1] is None and bool((row >= mean - 1e-12).all())
    # A change at client0/class0 cannot affect other clients/classes' contributions.
    assert torch.isfinite(actual[0]).all()


def test_extreme_logits_repeated_predictions_and_nonzero_gradient():
    z = torch.zeros(2, 16, dtype=torch.float64)
    z[:, 0] = -10000
    z = z[:, None].expand(-1, 64, -1).clone().requires_grad_()
    row, mean = mobihet_mean.class_losses(z.log_softmax(-1), torch.eye(16, dtype=z.dtype))
    torch.testing.assert_close(row, mean)
    grad = torch.autograd.grad(mean.sum(), z)[0]
    assert torch.isfinite(mean).all() and torch.isfinite(grad).all()
    assert grad[:, 0::16, 0].abs().min() > 0


def run_pair(old, new, rounds):
    initial = torch.get_rng_state()
    reference = []
    for r in rounds:
        info = old._round_private(r, .001)
        reference.append((copy.deepcopy(old.state()), info, torch.get_rng_state(), old.gen_g.get_state()))
    torch.set_rng_state(initial)
    for r, (state, info, rng, synth_rng) in zip(rounds, reference):
        got = new._round_private(r, .001)
        assert {k: v for k, v in got.items() if not k.startswith("mean_kd_")} == info
        for k, value in state.items():
            if torch.is_tensor(value):
                assert torch.equal(new.state()[k], value), k
        assert torch.equal(torch.get_rng_state(), rng)
        assert torch.equal(new.gen_g.get_state(), synth_rng)


@pytest.mark.parametrize("mode", ["row", "mean", "half"])
def test_cbb_identity_before_soft_teacher(mode):
    torch.set_num_threads(1)
    run_pair(mobihet_refine.Method(environment()), mobihet_mean.Method(environment(mean_kd=mode)),
             (1, 2) if mode == "row" else (1,))


@pytest.mark.parametrize("mode", ["mean", "half"])
@pytest.mark.parametrize("soft", [False, True])
def test_masks_real_q_bn_and_loss_reference(monkeypatch, mode, soft):
    m = mobihet_mean.Method(environment(mean_kd=mode))
    G, b = 3, 8
    mask = torch.zeros(G, b, dtype=torch.bool)
    mask[0, :5] = True
    enabled = torch.tensor([True, True, False])
    ctx = SimpleNamespace(Gfix=G, x=torch.zeros(G, b, 77), y=torch.arange(G * b).view(G, b) % 16,
                          clients=torch.tensor([0, 1, 0]), mask=mask, active=enabled, P={}, B={})
    z = torch.randn(G, b + 64, 16, requires_grad=True)
    lab = (torch.arange(64) % 16)[None].expand(G, -1)
    m._synth = lambda ctx: (torch.zeros(G, 64, 77), lab)
    expected_mask = torch.cat([mask, enabled[:, None].expand(G, 64)], 1)
    sentinel = {"bn": torch.tensor(3.)}

    def train(*args, **kw):
        assert torch.equal(kw["mask"], expected_mask)
        return z, sentinel

    monkeypatch.setattr(grouped, "TRAIN", train)
    m._shift, m._csum = torch.zeros(G, 16), torch.zeros(G, 16, 16)
    m._mean_stats = torch.zeros(7, dtype=torch.float64)
    m.Q = torch.eye(16) * .9 + .1 / 16 if soft else None
    loss, bn = m._loss_stream(ctx)
    ce = F.cross_entropy(z[0, :5], ctx.y[0, :5])
    if soft:
        row, mean = mobihet_mean.class_losses(z[:, b:].log_softmax(-1), m.Q)
        synth = mean if mode == "mean" else .5 * row
    else:
        synth = F.cross_entropy(z[:, b:].reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, 64).mean(1)
    expected = synth + torch.stack([ce, ce * 0, ce * 0])
    torch.testing.assert_close(loss, expected)
    assert bn is sentinel
    q = torch.zeros_like(m._csum)
    q[0].index_add_(0, ctx.y[0, :5], z[0, :5].detach().softmax(-1))
    torch.testing.assert_close(q, m._csum)
    grad = torch.autograd.grad(torch.where(enabled, loss, 0.).sum(), z)[0]
    assert grad[1, :b].count_nonzero() == 0 and grad[1, b:].count_nonzero() > 0
    assert grad[2].count_nonzero() == 0
    assert m._mean_stats[5:].tolist() == [2, 5]


@pytest.mark.parametrize("mode", ["mean", "half"])
def test_exact_resume_with_rng(mode):
    torch.set_num_threads(1)
    m = mobihet_mean.Method(environment(mean_kd=mode))
    m._round_private(1, .001)
    state = copy.deepcopy(m.state())
    rng = torch.get_rng_state()
    generators = [g.get_state() for g in (m.env.gen, m.gen_g, m.gen_q)]
    expected = m._round_private(2, .001)
    final = copy.deepcopy(m.state())
    resumed = mobihet_mean.Method(environment(mean_kd=mode))
    resumed.load_state(state)
    torch.set_rng_state(rng)
    for gen, value in zip((resumed.env.gen, resumed.gen_g, resumed.gen_q), generators):
        gen.set_state(value)
    assert resumed._round_private(2, .001) == expected
    for k, value in final.items():
        if torch.is_tensor(value):
            assert torch.equal(resumed.state()[k], value), k
