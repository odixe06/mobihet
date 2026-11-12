"""Prior/Q coordinate reference, real-NCKD reception and CBB/resume equivalence."""
import copy
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from core import grouped, local
from methods import mobihet_macro, mobihet_refine
from test_mobihet_refine import environment


def tensor_state_equal(a, b):
    assert a.keys() == b.keys()
    for k, v in a.items():
        if torch.is_tensor(v):
            assert torch.equal(v, b[k]), k


@pytest.mark.parametrize("mode,rounds", [("cbb", (1, 2)), ("real_nckd", (1,))])
def test_cbb_identity(mode, rounds):
    torch.set_num_threads(1)
    old = mobihet_refine.Method(environment())
    new = mobihet_macro.Method(environment(macro_mode=mode))
    rng = torch.get_rng_state()
    expected = []
    for r in rounds:
        info = old._round_private(r, .001)
        expected.append((copy.deepcopy(old.state()), info, torch.get_rng_state(), old.gen_g.get_state()))
    torch.set_rng_state(rng)
    for r, (state, info, after, synth_rng) in zip(rounds, expected):
        got = new._round_private(r, .001)
        assert {k: v for k, v in got.items() if not k.startswith("macro_")} == info
        tensor_state_equal(state, new.state())
        assert torch.equal(torch.get_rng_state(), after) and torch.equal(new.gen_g.get_state(), synth_rng)


def test_prior_uses_pooled_count_laplace_convention():
    m = mobihet_macro.Method(environment(macro_mode="prior"))
    total = m.cnt.sum(0).double()
    expected = (-math.log(16) - ((total + 1) / (total.sum() + 16)).log()).float()
    assert torch.equal(m.prior_shift, expected)


@pytest.mark.parametrize("mode", ["prior", "real_nckd"])
@pytest.mark.parametrize("soft,valid", [(False, True), (True, True), (True, False)])
def test_explicit_loss_gradient_q_masks_and_bn(monkeypatch, mode, soft, valid):
    m = mobihet_macro.Method(environment(macro_mode=mode))
    G, b = 3, 8
    mask = torch.zeros(G, b, dtype=torch.bool)
    mask[0, :5] = True  # real tail, floor-only client, inactive padded client
    enabled = torch.tensor([True, True, False])
    y = torch.arange(G * b).view(G, b) % 16
    ctx = SimpleNamespace(Gfix=G, x=torch.zeros(G, b, 77), y=y, clients=torch.tensor([0, 1, 0]),
                          mask=mask, active=enabled, P={}, B={})
    z = torch.randn(G, b + 64, 16, requires_grad=True)
    lab = (torch.arange(64) % 16)[None].expand(G, -1)
    m._synth = lambda ctx: (torch.zeros(G, 64, 77), lab)
    sentinel = {"bn": torch.tensor(3.)}

    def train(*args, **kw):
        assert torch.equal(kw["mask"], torch.cat([mask, enabled[:, None].expand(G, 64)], 1))
        return z, sentinel

    monkeypatch.setattr(grouped, "TRAIN", train)
    m._shift, m._csum = torch.zeros(G, 16), torch.zeros(G, 16, 16)
    m._macro_stats = torch.zeros(5, dtype=torch.float64)
    m.Q = torch.eye(16) * .9 + .1 / 16 if soft else None
    if soft:
        if not valid:
            m.Q[0] = m.Q[1]
        m._real_keep = m.Q.argmax(1) == torch.arange(16)
    loss, bn = m._loss_stream(ctx)
    shift = m.prior_shift if mode == "prior" else torch.zeros(16)
    zg = z[:, b:] + shift
    ce = local.ce_loss(z[:, :b], y, mask)
    if soft:
        q = m.Q[lab].clamp_min(1e-6)
        synth = F.kl_div(zg.log_softmax(-1), q, reduction="none").sum(-1).mean(1)
        if mode == "real_nckd":
            # Explicitly remove target column, rather than implementation's -1000 mask.
            terms = []
            for i in range(5):
                true = int(y[0, i])
                other = torch.arange(16) != true
                mass = 1 - m.Q[true, true]
                target = m.Q[true, other] / mass
                terms.append(m._real_keep[true] * mass * F.kl_div(z[0, i, other].log_softmax(-1), target, reduction="sum"))
            ce = ce + torch.stack([torch.stack(terms).mean(), ce[1] * 0, ce[2] * 0])
    else:
        synth = F.cross_entropy(zg.reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, 64).mean(1)
    expected = ce + synth
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-6)
    grad = torch.autograd.grad(torch.where(enabled, loss, 0.).sum(), z, retain_graph=True)[0]
    ref = torch.autograd.grad(torch.where(enabled, expected, 0.).sum(), z)[0]
    torch.testing.assert_close(grad, ref, atol=1e-6, rtol=1e-5)
    assert grad[1, :b].count_nonzero() == 0 and grad[1, b:].count_nonzero() > 0
    assert grad[2].count_nonzero() == 0 and bn is sentinel
    actual_q = torch.zeros_like(m._csum)
    actual_q[0].index_add_(0, y[0, :5], (z[0, :5].detach() + shift).softmax(-1))
    torch.testing.assert_close(m._csum, actual_q)
    assert m._macro_stats[:2].tolist() == [2, 5]
    assert m._macro_stats[2] == (5 if soft and mode == "real_nckd" else 0)
    assert m._macro_stats[4] == ((5 if valid else 4) if soft and mode == "real_nckd" else 0)


def test_nckd_has_zero_true_class_gradient_and_handles_hard_q():
    y = torch.tensor([[0, 1]])
    z = torch.randn(1, 2, 16, requires_grad=True)
    for q in (torch.eye(16)[y], torch.eye(16)[y] * .9 + .1 / 16):
        loss = mobihet_macro.nckd(q, z, y)
        grad = torch.autograd.grad(loss.sum(), z, retain_graph=True)[0]
        assert torch.isfinite(loss).all() and torch.isfinite(grad).all()
        assert grad.gather(-1, y[..., None]).count_nonzero() == 0


@pytest.mark.parametrize("mode", ["prior", "real_nckd"])
def test_exact_resume_with_all_rngs(mode):
    torch.set_num_threads(1)
    m = mobihet_macro.Method(environment(macro_mode=mode))
    m._round_private(1, .001)
    state = copy.deepcopy(m.state())
    rngs = [torch.get_rng_state(), m.env.gen.get_state(), m.gen_g.get_state(), m.gen_q.get_state()]
    info = m._round_private(2, .001)
    final = copy.deepcopy(m.state())
    resumed = mobihet_macro.Method(environment(macro_mode=mode))
    resumed.load_state(state)
    torch.set_rng_state(rngs[0])
    for gen, value in zip((resumed.env.gen, resumed.gen_g, resumed.gen_q), rngs[1:]):
        gen.set_state(value)
    assert resumed._round_private(2, .001) == info
    tensor_state_equal(final, resumed.state())
