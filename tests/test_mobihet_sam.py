"""SAM reference gradients, single BN/Q pass, masks, zero mode and continuation."""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from core import grouped, local
from methods import mobihet_refine, mobihet_sam
from test_mobihet_refine import environment


def test_perturbation_is_client_local_and_rejects_nonfinite():
    g = torch.tensor([[3., 4.], [30., 40.], [float("nan"), 2.], [2., 3.], [0., 0.]])
    e, finite = mobihet_sam.perturbation(g, torch.tensor([True, True, True, False, True]), .05)
    torch.testing.assert_close(e[:2], torch.tensor([[.03, .04], [.03, .04]]))
    assert e[2:].count_nonzero() == 0 and finite.tolist() == [True, True, False, True, True]


def test_zero_rho_preserves_two_rounds_exactly():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        old, new = mobihet_refine.Method(environment()), mobihet_sam.Method(environment(sam_rho=0))
        initial = torch.get_rng_state()
        expected = []
        for r in (1, 2):
            info = old._round_private(r, .001)
            expected.append((copy.deepcopy(old.state()), info, torch.get_rng_state(), old.gen_g.get_state()))
        torch.set_rng_state(initial)
        for r, (state, info, rng, synth_rng) in zip((1, 2), expected):
            assert new._round_private(r, .001) == info
            for k, value in state.items():
                if torch.is_tensor(value):
                    assert torch.equal(new.state()[k], value), k
            assert torch.equal(torch.get_rng_state(), rng)
            assert torch.equal(new.gen_g.get_state(), synth_rng)
    finally:
        torch.set_num_threads(before)


@pytest.mark.parametrize("soft", [False, True])
@pytest.mark.parametrize("bad_first", [False, True])
def test_nonzero_gradient_matches_explicit_two_pass_reference(soft, bad_first):
    torch.set_num_threads(1)
    m = mobihet_sam.Method(environment(sam_rho=.05))
    G, b = 3, 8
    flat, buffers = m.env.lay.init(42, G)
    params = {k: v.detach().requires_grad_(True) for k, v in m.env.lay.params(flat).items()}
    mask = torch.zeros(G, b, dtype=torch.bool)
    mask[0, :5] = True  # row1 floor-only, row2 padded; row0 real tail.
    ctx = SimpleNamespace(Gfix=G, G=2, x=torch.randn(G, b, 77), y=torch.arange(G * b).view(G, b) % 16,
                          clients=torch.tensor([0, 1, 0]), active=torch.tensor([True, True, False]),
                          mask=mask, P=params, B=m.env.lay.bufs(buffers), flat=flat)
    m.Q = torch.eye(16) * .9 + .1 / 16 if soft else None
    m._shift, m._csum = torch.zeros(G, 16), torch.zeros(G, 16, 16)
    m._sam_stats, m._sam_error = torch.zeros(7, dtype=torch.float64), torch.zeros(())
    loss, bn = m._loss_stream(ctx)
    synthetic, labels = m._sam_inputs
    rng_start, rng_after = m._sam_cpu_rng.clone(), torch.get_rng_state()
    original_sum = m._csum.clone()
    original_buffers = buffers.clone()
    keys = [k for k, _ in m.env.lay.p]
    grads = torch.autograd.grad(torch.where(ctx.active, loss, 0.).sum(), tuple(params[k] for k in keys))
    grad = torch.cat([g.reshape(G, -1) for g in grads], 1)

    def reference(p):
        z, new_b = grouped.forward(p, ctx.B, torch.cat([ctx.x, synthetic], 1), m.env.cfg,
                                   mask=torch.cat([mask, ctx.active[:, None].expand(G, 64)], 1), train=True)
        ce = local.ce_loss(z[:, :b] + m._shift[:, None], ctx.y, mask)
        lp = z.log_softmax(-1)
        if soft:
            lq = m.Q.clamp_min(1e-6).log()[labels]
            kd = (lq.exp() * (lq - lp[:, b:])).sum(-1).mean(1)
        else:
            kd = F.cross_entropy(z[:, b:].reshape(-1, 16), labels.reshape(-1), reduction="none").view(G, 64).mean(1)
        return torch.where(ctx.active, ce + kd, 0.).sum(), new_b

    p0 = {k: v.detach().clone().requires_grad_(True) for k, v in params.items()}
    torch.set_rng_state(rng_start)
    l0, ref_bn = reference(p0)
    g0 = torch.autograd.grad(l0, tuple(p0[k] for k in keys))
    flat_g0 = torch.cat([g.reshape(G, -1) for g in g0], 1)
    torch.testing.assert_close(grad, flat_g0)
    norm = flat_g0.norm(dim=1, keepdim=True)
    delta = .05 * flat_g0 / norm.clamp_min(1e-12)
    pd = m.env.lay.params(delta)
    p1 = {k: (p0[k].detach() + pd[k]).requires_grad_(True) for k in keys}
    torch.set_rng_state(rng_start)
    l1, _ = reference(p1)
    g1 = torch.autograd.grad(l1, tuple(p1[k] for k in keys))
    expected = torch.cat([g.reshape(G, -1) for g in g1], 1)
    torch.set_rng_state(rng_after)
    if bad_first:
        grad[1, 0] = float("nan")
    m._sam_gradient(ctx, grad)
    torch.testing.assert_close(grad[[0, 2]], expected[[0, 2]])
    if bad_first:
        assert torch.isnan(grad[1]).all()  # must remain rejected by AdamW
    else:
        torch.testing.assert_close(grad, expected)
        assert torch.isfinite(grad).all() and grad[1].norm() > 0
    assert grad[2].count_nonzero() == 0
    assert ctx.P is params and m._sam_inputs is None
    assert torch.equal(m._csum, original_sum) and torch.equal(buffers, original_buffers)
    assert torch.equal(torch.get_rng_state(), rng_after)
    torch.testing.assert_close(m.env.lay.flat_bufs(bn), m.env.lay.flat_bufs(ref_bn))
    assert m._sam_stats[6] == 5  # no double count of real rows


def test_training_continuation_replays_parameters_bn_q_and_rng():
    torch.set_num_threads(1)
    m = mobihet_sam.Method(environment(sam_rho=.05))
    m.env.hp["synth_step_floor"] = 8
    a = m._round_private(1, .001)
    saved = copy.deepcopy(m.state())
    rngs = [torch.get_rng_state(), m.env.gen.get_state(), m.gen_g.get_state(), m.gen_q.get_state()]
    b = m._round_private(2, .001)
    state, rng_after = copy.deepcopy(m.state()), torch.get_rng_state()
    new = mobihet_sam.Method(environment(sam_rho=.05))
    new.env.hp["synth_step_floor"] = 8
    new.load_state(saved)
    torch.set_rng_state(rngs[0])
    for gen, value in zip((new.env.gen, new.gen_g, new.gen_q), rngs[1:]):
        gen.set_state(value)
    assert new._round_private(2, .001) == b
    for k, value in state.items():
        if torch.is_tensor(value):
            assert torch.equal(new.state()[k], value), k
    assert torch.equal(torch.get_rng_state(), rng_after)
    for info in (a, b):
        assert info["skipped_frac"] == 0 and info["clients_without_applied"] == 0
        assert info["sam_nonfinite_first"] == info["sam_nonfinite_second"] == 0
        assert info["sam_real_rows"] == len(m.env.train.Y)
        assert info["sam_perturb_norm_error_bound"] < 1e-6
