"""W20 loss algebra, synthetic step floor and compact bootstrap codec (CPU, no data)."""
import math

import torch
import torch.nn.functional as F

from core import local
from methods.mobihet_wstudy import hard_plus_nckd, nckd


def _rand(seed=0, n=64):
    g = torch.Generator().manual_seed(seed)
    t = torch.rand(n, 16, generator=g).pow(3)
    t = t / t.sum(-1, keepdim=True)
    z = torch.randn(n, 16, generator=g) * 3
    y = torch.randint(16, (n,), generator=g)
    return t.double(), z.double(), y


def test_kd_decomposes_into_binary_plus_weighted_nckd():
    t, z, y = _rand()
    logp = F.log_softmax(z, -1)
    full = (t * (t.log() - logp)).sum(-1)
    t_y, p_y = t.gather(1, y[:, None]).squeeze(1), logp.exp().gather(1, y[:, None]).squeeze(1)
    binary = t_y * (t_y / p_y).log() + (1 - t_y) * ((1 - t_y) / (1 - p_y)).log()
    torch.testing.assert_close(full, binary + nckd(t, z, y), rtol=0, atol=1e-6)


def test_nckd_leaves_true_class_logit_without_gradient():
    t, z, y = _rand(1)
    z = z.clone().requires_grad_(True)
    nckd(t, z, y).sum().backward()
    assert torch.all(z.grad.gather(1, y[:, None]).abs() < 1e-12)
    # other logits still learn the teacher's non-target relation
    assert z.grad.abs().sum() > 0


def test_confused_teacher_row_no_longer_pulls_true_class_down():
    # Q[13]-like row: .74 true, .25 benign. Full KD gradient on z_y is positive (push down) when p_y > t_y; NCKD's is 0.
    t = torch.zeros(1, 16, dtype=torch.double); t[0, 13], t[0, 0] = .74, .25; t[0, 1:13] = .01 / 12
    y = torch.tensor([13])
    z = torch.zeros(1, 16, dtype=torch.double); z[0, 13] = 6.
    z1 = z.clone().requires_grad_(True)
    (t * (t.log() - F.log_softmax(z1, -1))).sum().backward()
    z2 = z.clone().requires_grad_(True)
    nckd(t, z2, y).sum().backward()
    assert z1.grad[0, 13] > .2 and abs(float(z2.grad[0, 13])) < 1e-12


def test_hard_plus_nckd_is_ce_for_onehot_target():
    _, z, y = _rand(2)
    t = F.one_hot(y, 16).double()
    torch.testing.assert_close(hard_plus_nckd(t, z, y), F.cross_entropy(z, y, reduction="none"), rtol=0, atol=1e-5)  # fp32 logits


def test_nckd_finite_in_fp16_logits():
    t, z, y = _rand(3)
    assert torch.isfinite(nckd(t.float(), (z * 10).half(), y)).all()


def _nckd_w20(target, logits, y):
    """nckd as launched in W20 (commit 97cf51d)."""
    onehot = F.one_hot(y, target.shape[-1]).bool()
    t_y = target.gather(-1, y[..., None]).squeeze(-1)
    t_hat = target.masked_fill(onehot, 0.) / (1 - t_y).clamp_min(1e-6)[..., None]
    log_p_hat = F.log_softmax(logits.float() - 1000. * onehot, -1)
    kl = torch.where(onehot, 0., t_hat * (t_hat.clamp_min(1e-6).log() - log_p_hat)).sum(-1)
    return (1 - t_y) * kl


def test_nckd_default_is_w20_bitexact():
    t, z, y = _rand(5)
    for tt, zz in ((t, z), (t.float(), z.half())):
        assert torch.equal(nckd(tt, zz, y), _nckd_w20(tt, zz, y))


def test_nckd_local_all_kept_matches_nckd_and_ignores_absent_classes():
    t, z, y = _rand(6)
    keep = torch.ones(1, 16, dtype=torch.bool)
    torch.testing.assert_close(nckd(t, z, y, keep), nckd(t, z, y), rtol=0, atol=1e-6)  # fp32 logits
    keep = torch.rand(1, 16, generator=torch.Generator().manual_seed(7)) > .5
    keep[0, y] = True
    zz = z.clone().requires_grad_(True)
    nckd(t, zz, y, keep).sum().backward()
    assert torch.all(zz.grad[:, ~keep[0]].abs() < 1e-12) and zz.grad[:, keep[0]].abs().sum() > 0
    # a vehicle holding only the true class has no non-target term
    only = F.one_hot(y, 16).bool()
    assert torch.all(nckd(t, z, y, only) == 0)


class _Lay:
    """One scalar parameter per client; enough to drive local_epoch's schedule."""
    P, B = 1, 0
    p = [("w", (1,))]

    def params(self, flat):
        return {"w": flat[:, :1]}

    def bufs(self, flat):
        return {}


def _epoch(sizes, floor):
    off = torch.tensor([0, *torch.tensor(sizes).cumsum(0).tolist()])
    data = type("D", (), {"X": torch.zeros(int(off[-1]), 3), "Y": torch.zeros(int(off[-1]), dtype=torch.long), "off": off})
    seen, active_steps = [], []

    def loss_fn(ctx):
        seen.append(ctx.rows[ctx.mask].tolist())
        active_steps.append(ctx.active.clone())
        return ctx.P["w"].sum(1) * 0 + 1.0 * ctx.P["w"].sum(1), {}

    flat = torch.zeros(len(sizes), 1)
    hp = {"batch": 8, "group_rows": 64, "weight_decay": 0., "clip": 1., "amp": False, "synth_step_floor": floor}
    stats = local.local_epoch(_Lay(), flat, torch.zeros(len(sizes), 0), data, torch.arange(len(sizes)), {}, hp, 1e-3,
                              torch.Generator().manual_seed(0), loss_fn=loss_fn)
    return seen, torch.stack(active_steps), stats


def test_step_floor_keeps_one_pass_over_local_rows():
    sizes = [20, 5]
    for floor in (0, 6):
        seen, active, stats = _epoch(sizes, floor)
        rows = sorted(r for step in seen for r in step)
        assert rows == list(range(sum(sizes)))                      # every local row exactly once
        assert stats["steps"] == max(math.ceil(20 / 8), floor)
        real_steps = torch.tensor([math.ceil(20 / 8), math.ceil(5 / 8)])
        expected = torch.maximum(real_steps, torch.tensor(floor))
        assert torch.equal(stats["applied"], expected.float())
        assert torch.equal(active[:, :2].sum(0), expected)          # padded clients never active


def test_compact_cdf_grid_is_a_valid_marginal():
    g = torch.Generator().manual_seed(4)
    hist = torch.rand(16, 77, 256, generator=g).pow(8) * (torch.rand(16, 77, 256, generator=g) > .7)
    mass = hist / hist.sum(-1, keepdim=True).clamp_min(1)
    cdf = mass.cumsum(-1)
    q = torch.round(cdf.double() * 65535) / 65535
    q[..., -1] = torch.where(cdf[..., -1] > 0, 1., 0.).double()
    m = torch.diff(q, dim=-1, prepend=torch.zeros_like(q[..., :1]))
    assert (m >= 0).all() and torch.allclose(m.sum(-1), (cdf[..., -1] > 0).double())
    assert (q - cdf.double()).abs().max() <= .5 / 65535 + 1e-6
