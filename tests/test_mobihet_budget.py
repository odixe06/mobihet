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


def _select(select, q=0, **knob):
    m, ctx = _method(True, real_tier_rows=[4, 8], real_select=select, real_select_q=q, **knob)
    ctx.rows = torch.arange(24).view(3, 8)
    m._real_k = torch.tensor([4, 2, 8])                                 # vehicle 0: 8 real rows -> 4; vehicle 1: 3 -> 2
    return m, ctx


def test_top_select_trains_the_highest_stored_loss_and_stores_the_new_one():
    m, ctx = _select("top")
    score = torch.tensor([.1, .9, .2, .8, .3, .7, .4, .6, 5., 1., 3.] + [9.] * 13 + [0.])   # last: sink
    m._score, m._trained = score.clone(), torch.zeros(())
    loss, _ = m._loss_tier(ctx)
    real = torch.zeros(3, 8, dtype=torch.bool)
    real[0, [1, 3, 5, 7]] = True                                        # top 4 of .1 .9 .2 .8 .3 .7 .4 .6
    real[1, [0, 2]] = True                                              # top 2 of 5 1 3
    order = torch.stack([torch.tensor([1, 3, 5, 7, 0, 2, 4, 6]), torch.tensor([0, 2, 1, 3, 4, 5, 6, 7]), torch.arange(8)])
    take = lambda t: torch.stack([t[v, order[v]] for v in range(3)])
    fb, ctx2 = _method(True, real_tier_rows=[4, 8])                     # FB on the slot reordered: same rows, first in slot
    fb._real_k, ctx2.P, ctx2.x, ctx2.y, ctx2.mask = m._real_k, ctx.P, take(ctx.x), take(ctx.y), take(ctx.mask)
    torch.testing.assert_close(loss, fb._loss_tier(ctx2)[0], rtol=1e-5, atol=1e-6)
    assert torch.equal(m._seen.sum(1), real.sum(1).float()) and int(m._trained) == 6
    left = ~real.view(-1)
    assert torch.equal(m._score[:24][left], score[:24][left]) and (m._score[:24][real.view(-1)] != score[:24][real.view(-1)]).all()


def test_untrained_rows_first_and_ties_keep_the_random_slot_order():
    m, ctx = _select("top")                                             # all +inf: same rows as FB
    m._real_rows(ctx)
    real, q = m._real_rows(ctx)
    want = ctx.mask & (torch.arange(8)[None] < m._real_k[:, None])
    assert torch.equal(real, want) and torch.equal(q, want)
    m._score[ctx.rows[0, :2]] = .5                                      # two rows already trained: two others take their place
    real, _ = m._real_rows(ctx)
    assert real[0].tolist() == [False, False, True, True, True, True, False, False]


def test_prop_select_uses_its_own_generator_and_never_picks_zero_loss_rows_first():
    m, ctx = _select("prop")
    m._score = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1.] + [1.] * 17)
    gen = m.env.gen.get_state()
    for _ in range(20):
        real, _ = m._real_rows(ctx)
        assert real[0].tolist() == [False] * 4 + [True] * 4 and real.sum(1).tolist() == [4, 2, 0]
    assert torch.equal(gen, m.env.gen.get_state())


def test_select_q_keeps_the_random_q_rows_and_picks_the_rest_by_loss():
    m, ctx = _select("top", q=2)
    m._real_k = torch.tensor([4, 3, 8])
    m._score = torch.tensor([.1, .2, .3, .4, .5, .6, .7, .8] + [1.] * 17)
    m._trained = torch.zeros(())
    m._loss_tier(ctx)
    assert m._seen.sum(1).tolist() == [2, 2, 0]                         # Q: slot positions 0-1 only
    assert int(m._trained) == 4 + 3
    assert (m._score[[0, 1, 6, 7]] != torch.tensor([.1, .2, .7, .8])).all()    # q rows 0-1 + top 2 of the rest
    assert torch.equal(m._score[2:6], torch.tensor([.3, .4, .5, .6]))


def test_row_score_round_trips_and_select_knobs_are_checked():
    m, ctx = _select("prop")
    m._score[:5] = torch.arange(5.)
    m._real_rows(ctx)
    s = m.state()
    m2, _ = _select("prop")
    m2.load_state(s)
    assert torch.equal(m2._score, m._score) and torch.equal(m2._real_rows(ctx)[0], m._real_rows(ctx)[0]) and "row_score" not in _method(True, real_tier_rows=[4, 8])[0].state()
    for bad in (dict(real_select="random", real_select_q=1), dict(real_select="top", real_select_q=4), dict(real_select="x")):
        with pytest.raises(AssertionError):
            _method(True, real_tier_rows=[4, 8], **bad)


@pytest.mark.parametrize("select,q", [("top", 0), ("prop", 0), ("top", 2)])
def test_select_round_same_compute_and_permutation_stream_as_fb(select, q):
    torch.set_num_threads(1)
    fb, new = (mobihet_budget.Method(big(**T3, **TIER, real_tier_rows=[8, 5], **k))
               for k in ({}, dict(real_select=select, real_select_q=q)))
    rng = torch.get_rng_state()
    want = [fb._round_private(r, .001)["real_frac"] for r in (1, 2)]
    torch.set_rng_state(rng)
    for r, w in zip((1, 2), want):
        info = new._round_private(r, .001)
        assert abs(info["real_frac"] - w) < 1e-12 and info["clients_without_applied"] == 0
        torch.testing.assert_close(new.Q.sum(-1), torch.ones(16), rtol=0, atol=1e-6)
    assert torch.equal(fb.env.gen.get_state(), new.env.gen.get_state())
    assert torch.isfinite(new._score[:-1]).float().mean() > .9           # nearly every row trained within two rounds


@pytest.mark.parametrize("select", ["random"])
def test_compact_without_floor_is_the_masked_round(select):
    """Whole-round state, random only: top moves the chosen rows to the front, which changes the BN summation order
    (~1e-8) and Adam amplifies that on near-zero gradients (state gap 1e-2 after round 2, 19/11/2026); top is checked
    step by step in fp64 by the floor test below (round 2 has the reordered rows)."""
    torch.set_num_threads(1)
    knob = dict(real_tier_rows=[8, 5], **({} if select == "random" else {"real_select": select}))
    made = []
    for compact in (False, True):
        env = big(**T3, **TIER, **knob, compact=compact)
        env.cfg = dict(env.cfg, dropout=0.0)
        made.append(mobihet_budget.Method(env))
    rng = torch.get_rng_state()
    want = [(made[0]._round_private(r, .001), copy.deepcopy(made[0].state())) for r in (1, 2)]
    torch.set_rng_state(rng)
    for r, (info, state) in zip((1, 2), want):
        assert made[1]._round_private(r, .001)["real_frac"] == info["real_frac"]
        for k in ("priv_p", "priv_b", "Q"):
            torch.testing.assert_close(made[1].state()[k], state[k], rtol=0, atol=1e-6)


@pytest.mark.parametrize("select, q", [("random", 0), ("top", 0), ("prop", 0), ("top", 2)])
def test_compact_step_equals_the_masked_step_along_a_round_with_floor(select, q, monkeypatch):
    """Floor steps: Adam amplifies 1e-8 rounding on near-zero gradients over a round, so compare every step of the
    compact run against the masked step on the same inputs (loss, gradient, BN), and check the executed widths.
    The comparison runs in fp64: in fp32 a ReLU input within rounding of 0 can flip sign between the two shapes and move
    a gradient by ~1e-3 (step 120: 1 element of 38 kink ops; either path is the one off vs fp64, 19/11/2026); the round
    itself continues on the fp32 compact step."""
    torch.set_num_threads(1)
    knob = {} if select == "random" else {"real_select": select, "real_select_q": q}
    env = big(**T3, **TIER, real_tier_rows=[8, 5], compact=True, **knob)
    env.cfg, env.hp["synth_step_floor"] = dict(env.cfg, dropout=0.0), 200   # toy vehicles: < 130 real steps -> floor steps
    m = mobihet_budget.Method(env)
    widths, loss_tier, seen, moved, want = [], m._loss_tier, [], [], []

    def spy(P, B, x, cfg, mask=None, train=True):
        widths.append(x.shape[1])
        return grouped.forward(P, B, x, cfg, mask=mask, train=train)

    def run(ctx, compact, f64, rng, gen, side, score, sgen):
        for k, v in side.items():
            setattr(m, k, v.clone().double() if f64 and k == "_csum" else v.clone())
        if score is not None:
            m._score = score.clone()
            m._sgen.set_state(sgen)                                      # prop: Gumbel draws
        torch.set_rng_state(rng)
        m.gen_g.set_state(gen)
        m.compact = compact
        P, B, x = ctx.P, ctx.B, ctx.x
        if f64:
            ctx.P = {k: v.detach().double().requires_grad_(True) for k, v in P.items()}
            ctx.B, ctx.x = {k: v.double() for k, v in B.items()}, x.double()
        try:
            out, new_b = loss_tier(ctx)
            g = torch.autograd.grad(out.sum(), list(ctx.P.values()), retain_graph=not f64, allow_unused=True)
        finally:
            ctx.P, ctx.B, ctx.x = P, B, x
        return out, g, new_b

    def both(ctx):
        state = (torch.get_rng_state(), m.gen_g.get_state(),             # gen_g: synthetic rows
                 {k: getattr(m, k).clone() for k in ("_csum", "_seen", "_trained")},
                 *((m._score.clone(), m._sgen.get_state()) if select != "random" else (None, None)))
        if select != "random":                                           # chosen rows not a slot prefix: compact reorders
            real = m._real_rows(ctx)[0]
            moved.append(bool((real != (torch.arange(real.shape[1]) < real.sum(1, keepdim=True))).any()))
        ref, g_ref, ref_b = run(ctx, False, True, *state)
        out, g, new_b = run(ctx, True, True, *state)
        torch.testing.assert_close(out, ref, rtol=1e-10, atol=1e-12)
        for a, b in zip(g, g_ref):
            if a is not None or b is not None:
                torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-12)
        for k in ref_b:
            torch.testing.assert_close(new_b[k], ref_b[k], rtol=1e-10, atol=1e-12)
        seen.append(ctx.step >= m._real_steps)
        c = ctx.clients[:ctx.G]                                          # executed width from the device tensors
        want.append((0 if seen[-1] else max(min(ctx.x.shape[1], v) for v in m._real_k[c].tolist()))
                    + max(16 if v == 8 else v for v in m._rows_k[c].tolist()))
        out, _, new_b = run(ctx, True, False, *state)
        return out, new_b
    monkeypatch.setattr(grouped, "TRAIN", spy)
    m._loss_tier = both
    for r in (1, 2):
        widths.clear(), want.clear()
        assert m._round_private(r, .001)["clients_without_applied"] == 0
        full = 8 + m.aux                                                 # masked engine: slot + round's widest dose
        compact = widths[2::3]                                           # masked fp64, compact fp64, compact fp32
        assert compact == want and max(compact) <= full and min(compact) < full   # floor steps: synthetic only
    assert any(seen) and not all(seen) and (select == "random" or any(moved))
