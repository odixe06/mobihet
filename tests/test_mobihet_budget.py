"""Fixed row budget: knobs off = Lite2 bit-exact, capped real rows leave the step, class-Q stays a mean, reference arms."""
import copy

import pytest
import torch
import torch.nn.functional as F

from core import grouped, local
from methods import mobihet_budget, mobihet_lite
from test_mobihet_compute import _ctx
from test_mobihet_lite import T3, big
from test_mobihet_macro import tensor_state_equal
from test_mobihet_refine import environment

TIER = dict(synth_tier_min=[800], synth_tier_rows=[[64, 64], [32, 16]], synth_late_from=1)


@pytest.mark.parametrize("knob", [{}, {"real_tier_rows": [64, 64]}])
def test_off_and_full_rows_are_lite_bit_exact(knob):
    torch.set_num_threads(1)
    old, new = mobihet_lite.Method(big(**T3, **TIER)), mobihet_budget.Method(big(**T3, **TIER, **knob))
    rng = torch.get_rng_state()
    want = [(old._round_private(r, .001), copy.deepcopy(old.state())) for r in (1, 2)]
    torch.set_rng_state(rng)
    for r, (info, state) in zip((1, 2), want):
        got = new._round_private(r, .001)
        assert got.pop("real_frac", 1.0) == 1.0 and got == info
        tensor_state_equal(state, new.state())


def _method(soft, **knob):
    env = environment(synth_tier_min=[10 ** 9], synth_tier_rows=[[64, 64], [16, 16]], synth_late_from=1, **knob)
    env.cfg = dict(env.cfg, dropout=0.0)
    m = mobihet_budget.Method(env)
    m.Q = torch.eye(16) * .9 + .1 / 16 if soft else None
    ctx = _ctx(m)
    ctx.step = 0
    m._rows_k, m._seen = torch.tensor([64, 16, 16]), torch.zeros(3, 16)
    return m, ctx


def _reference(m, ctx, real, rows, soft, q_real=False):
    """Explicit loss of vehicle g: its trained real rows + its first rows[g] synthetic rows, nothing else in the batch."""
    g = torch.Generator().manual_seed(1)
    xg = torch.randn(3, 64, 77, generator=g)
    lab = torch.arange(64) % 16
    out = []
    for v in range(3):
        nr, ns = int(real[v].sum()), rows[v]
        if nr + ns == 0:
            out.append(torch.zeros(()))
            continue
        x = torch.cat([ctx.x[v, :nr], xg[v, :ns]])[None]
        z, _ = grouped.forward({k: p[v:v + 1] for k, p in ctx.P.items()}, {k: p[v:v + 1] for k, p in ctx.B.items()}, x,
                               m.env.cfg, mask=torch.ones(1, nr + ns, dtype=torch.bool), train=True)
        z = z[0]
        loss = F.cross_entropy(z[:nr], ctx.y[v, :nr]) if nr else z.sum() * 0
        logp = F.log_softmax(z, -1)
        if ns and soft:
            lt = m.Q.clamp_min(1e-6).log()[lab[:ns]]
            loss = loss + (lt.exp() * (lt - logp[nr:])).sum(-1).mean()
        elif ns:
            loss = loss + F.cross_entropy(z[nr:], lab[:ns])
        if q_real and soft and nr:
            lt = m.Q.clamp_min(1e-6).log()[ctx.y[v, :nr]]
            loss = loss + (lt.exp() * (lt - logp[:nr])).sum(-1).mean()
        out.append(loss)
    return torch.stack(out)


@pytest.mark.parametrize("soft", [False, True])
def test_capped_real_rows_leave_the_step(soft):
    m, ctx = _method(soft, real_tier_rows=[5, 8])
    m._real_k = torch.tensor([5, 2, 8])                                 # vehicle 0: 8 real rows -> 5; vehicle 1: 3 -> 2
    loss, _ = m._loss_tier(ctx)
    real = ctx.mask & (torch.arange(8)[None] < m._real_k[:, None])
    assert real.sum(1).tolist() == [5, 2, 0]
    torch.testing.assert_close(loss, _reference(m, ctx, real, [64, 16, 16], soft), rtol=1e-4, atol=1e-5)
    assert torch.equal(m._seen.sum(1), real.sum(1).float())
    torch.testing.assert_close(m._csum.sum((1, 2)), real.sum(1).float())   # one probability vector per trained row
    grad = torch.autograd.grad(loss.sum(), list(ctx.P.values()))
    m2, ctx2 = _method(soft, real_tier_rows=[5, 8])
    m2._real_k = m._real_k
    ctx2.P = ctx.P
    ctx2.x = torch.where(real[..., None], ctx.x, ctx.x + 7.0)           # rows outside the budget cannot matter
    loss2, _ = m2._loss_tier(ctx2)
    for a, b in zip(grad, torch.autograd.grad(loss2.sum(), list(ctx.P.values()))):
        torch.testing.assert_close(a, b)


def test_reference_arms_without_synthetic_rows():
    m, ctx = _method(True, synth_off=True, q_real=True)
    loss, _ = m._loss_tier(ctx)
    torch.testing.assert_close(loss, _reference(m, ctx, ctx.mask, [0, 0, 0], True, q_real=True), rtol=1e-4, atol=1e-5)
    m, ctx = _method(True, synth_off=True)
    loss, _ = m._loss_tier(ctx)
    torch.testing.assert_close(loss, _reference(m, ctx, ctx.mask, [0, 0, 0], True), rtol=1e-4, atol=1e-5)
    env = environment(synth_tier_min=[10 ** 9], synth_tier_rows=[[64, 64], [16, 16]], synth_late_from=1, synth_off=True)
    env.hp["synth_step_floor"] = 16
    with pytest.raises(AssertionError):
        mobihet_budget.Method(env)


def test_round_trains_the_budgeted_rows_and_q_stays_a_distribution():
    torch.set_num_threads(1)
    env = big(**T3, **TIER, real_tier_rows=[8, 5])
    m = mobihet_budget.Method(env)
    n = torch.as_tensor(env.train.sizes)
    cap = torch.where(n >= 800, 5, 8)
    want = int(((n // 8) * cap + torch.minimum(n % 8, cap)).sum()) / int(n.sum())
    for r in (1, 2):
        info = m._round_private(r, .001)
        assert abs(info["real_frac"] - want) < 1e-9 and info["clients_without_applied"] == 0
        torch.testing.assert_close(m.Q.sum(-1), torch.ones(16), rtol=0, atol=1e-6)
        idx = list(env.groups_idx())[-1]                                 # last group's packets: mean over trained rows
        G = len(idx)
        got = m._csum[:G].sum(-1) / m.cnt[idx].clamp_min(1)
        ok = m._seen[:G] > 0
        assert ok.any() and (got[ok] - 1).abs().max() < .04 and (got[~ok] == 0).all()   # Q8 rounding: 16 * .5 / 255
