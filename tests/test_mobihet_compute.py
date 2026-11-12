"""B20 compute screen: CBB identity, head-only synthetic gradient isolation, finite floor steps, late-row schedule."""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from core import grouped, local
from methods import mobihet_compute, mobihet_refine
from test_mobihet_macro import tensor_state_equal
from test_mobihet_refine import environment


def test_full_is_cbb_bit_exact():
    torch.set_num_threads(1)
    old = mobihet_refine.Method(environment())
    new = mobihet_compute.Method(environment(synth_grad="full"))
    rng = torch.get_rng_state()
    expected = []
    for r in (1, 2):
        info = old._round_private(r, .001)
        expected.append((copy.deepcopy(old.state()), info, torch.get_rng_state(), old.gen_g.get_state()))
    torch.set_rng_state(rng)
    for r, (state, info, after, synth_rng) in zip((1, 2), expected):
        got = new._round_private(r, .001)
        assert got.pop("synth_rows") == 64 and got == info
        tensor_state_equal(state, new.state())
        assert torch.equal(torch.get_rng_state(), after) and torch.equal(new.gen_g.get_state(), synth_rng)


def _ctx(m, G=3, b=8):
    env = m.env
    P, B = grouped.init_state(env.cfg, 0, G)
    P = {k: v.detach().requires_grad_(True) for k, v in P.items()}
    mask = torch.zeros(G, b, dtype=torch.bool)
    mask[0] = True
    mask[1, :3] = True                                                  # client 2: floor-only step (no real row)
    ctx = SimpleNamespace(Gfix=G, x=torch.randn(G, b, 77), y=torch.arange(G * b).view(G, b) % 16,
                          clients=torch.tensor([0, 1, 2]), mask=mask, active=torch.tensor([True, True, True]),
                          P=P, B=B)
    m._shift, m._csum = torch.zeros(G, 16), torch.zeros(G, 16, 16)
    g = torch.Generator().manual_seed(1)
    m._synth = lambda ctx: (torch.randn(G, m.aux, 77, generator=g), (torch.arange(m.aux) % 16)[None].expand(G, -1))
    return ctx


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("mode", ["head", "mix16"])
def test_synthetic_gradient_path(mode, soft):
    env = environment(synth_grad=mode)
    env.cfg = dict(env.cfg, dropout=0.0)
    m = mobihet_compute.Method(env)
    m.Q = torch.eye(16) * .9 + .1 / 16 if soft else None
    ctx = _ctx(m)
    loss, new_b = m._loss_stream(ctx)
    assert torch.isfinite(loss).all() and new_b
    loss.sum().backward()
    got = {k: v.grad.clone() for k, v in ctx.P.items()}
    # reference: same real/full pass, synthetic head rows through a detached backbone
    for v in ctx.P.values():
        v.grad = None
    G, b, nf = 3, 8, 16 if mode == "mix16" else 0
    g = torch.Generator().manual_seed(1)
    xg = torch.randn(G, 64, 77, generator=g)
    lab = (torch.arange(64) % 16)[None].expand(G, -1)
    mf = torch.cat([ctx.mask, torch.ones(G, nf, dtype=torch.bool)], 1)
    bn = torch.where(mf.any(1)[:, None], mf, torch.ones_like(mf))
    zf, _ = grouped.forward(ctx.P, ctx.B, torch.cat([ctx.x, xg[:, :nf]], 1), env.cfg, mask=bn, train=True)
    detached = {k: (v if k.startswith("head.") else v.detach()) for k, v in ctx.P.items()}
    zh, _ = grouped.forward(detached, ctx.B, xg[:, nf:], env.cfg, mask=torch.ones(G, 64 - nf, dtype=torch.bool),
                            train=True)
    zs = torch.cat([zf[:, b:], zh], 1)
    ce = local.masked_mean(F.cross_entropy(zf[:, :b].reshape(-1, 16), ctx.y.reshape(-1), reduction="none").view(G, b),
                           ctx.mask)
    if soft:
        lt = m.Q.clamp_min(1e-6).log()[lab]
        syn = (lt.exp() * (lt - F.log_softmax(zs, -1))).sum(-1).mean(1)
    else:
        syn = F.cross_entropy(zs.reshape(-1, 16), lab.reshape(-1), reduction="none").view(G, 64).mean(1)
    ref = ce + syn
    torch.testing.assert_close(loss, ref)
    ref.sum().backward(retain_graph=True)
    for k, v in ctx.P.items():
        torch.testing.assert_close(got[k], v.grad, msg=k)
    if mode == "head":                                                  # backbone sees only the real rows
        for v in ctx.P.values():
            v.grad = None
        ce.sum().backward(inputs=[v for k, v in ctx.P.items() if not k.startswith("head.")])
        for k, v in ctx.P.items():
            if not k.startswith("head."):
                torch.testing.assert_close(got[k], v.grad, msg=k)
    # Q from real rows only, floor-only client contributes nothing
    assert m._csum[2].sum() == 0 and torch.isclose(m._csum[0].sum(), torch.tensor(8.)) and torch.isclose(
        m._csum[1].sum(), torch.tensor(3.))


def test_late_rows_schedule():
    m = mobihet_compute.Method(environment(synth_rows_late=32, synth_late_from=1))
    assert m._rows_for(1) == 64 and m._rows_for(2) == 32
    info = m._round_private(1, .001)
    assert info["synth_rows"] == 64
    info = m._round_private(2, .001)
    assert info["synth_rows"] == 32 and m.aux == 32 and m.labels.tolist() == [i % 16 for i in range(32)]


@pytest.mark.parametrize("mode", ["head", "mix16"])
def test_rounds_finite_with_floor(mode):
    torch.set_num_threads(1)
    env = environment(synth_grad=mode)
    env.hp["synth_step_floor"] = 4
    m = mobihet_compute.Method(env)
    for r in (1, 2):
        info = m._round_private(r, .001)
        assert info["clients_without_applied"] == 0 and torch.isfinite(torch.tensor(info["train_loss"]))
    assert all(torch.isfinite(v).all() for v in m.state().values() if torch.is_tensor(v))
