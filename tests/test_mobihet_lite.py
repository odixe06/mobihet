"""Lite bootstrap v2 (km2): off = T3 path bit-exact, neutral knobs = lite6 mixture, byte accounting, 8-row dose rotation."""
import copy

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from core import grouped, local
from core.data import Split
from methods import mobihet_compute, mobihet_lite
from test_mobihet_compute import _ctx
from test_mobihet_macro import tensor_state_equal
from test_mobihet_refine import environment

KM = dict(cop_kmeans=4, cop_kmeans_w0=.5, cop_kmeans_q="int8")
T3 = dict(KM, cop_kmeans_lite=6, km_bytes=True)
KM2 = dict(KM, km2=True)


def big(**extra):
    """ToyEnv with enough rows per (class, cluster) for kept clusters and second-moment thresholds."""
    env = environment(**extra)
    rng = np.random.default_rng(3)
    n = rng.integers(700, 1000, env.K)
    off = np.concatenate([[0], np.cumsum(n)])
    y = torch.from_numpy(rng.integers(0, 16, off[-1]).astype(np.int8))
    x = torch.from_numpy(rng.standard_normal((int(off[-1]), 77))).float() * (1 + .2 * (y[:, None] % 3)) + .3 * y[:, None].float()
    env.train = Split(x, y, torch.from_numpy(off), torch.arange(env.K))
    env.n = torch.as_tensor(env.train.sizes, dtype=torch.float32)
    env.groups = local.group_schedule(env.train.sizes, list(range(env.K)), env.hp["batch"], env.hp["group_rows"])
    return env


def test_km2_off_is_t3_bit_exact():
    torch.set_num_threads(1)
    tier = dict(synth_tier_min=[800], synth_tier_rows=[[64, 64], [32, 16]], synth_late_from=1)
    old, new = mobihet_compute.Method(big(**T3, **tier)), mobihet_lite.Method(big(**T3, **tier))
    for k in ("km_p", "km_mu", "km_L", "cop_cdf"):
        assert torch.equal(getattr(old, k), getattr(new, k)), k
    assert old.bootstrap_bytes == new.bootstrap_bytes
    rng = torch.get_rng_state()
    want = [(old._round_private(r, .001), copy.deepcopy(old.state())) for r in (1, 2)]
    torch.set_rng_state(rng)
    for r, (info, state) in zip((1, 2), want):
        assert new._round_private(r, .001) == info
        tensor_state_equal(state, new.state())


def test_km2_neutral_knobs_give_the_lite6_mixture():
    old, new = mobihet_compute.Method(big(**T3)), mobihet_lite.Method(big(**KM2))
    for k in ("km_p", "km_mu", "km_L", "cop_cdf", "cop_mass"):
        assert torch.equal(getattr(old, k), getattr(new, k)), k
    assert old.km_kept == new.km_kept > 0


def test_km2_bytes_add_up_and_match_independent_coders():
    env = big(**KM2, km2_frac=.5, km2_tau=16, km2_bits=5, km2_mean8=True, cop_bins=64)
    m = mobihet_lite.Method(env)
    r = m.codec_bytes
    assert m.cop_cdf.shape[-1] == 64 and r["bins"] == 64 and r["kept"] > 0
    assert r["up_mean"] == pytest.approx(sum(r["up"].values()))
    assert r["down_mean"] == pytest.approx(sum(v for k, v in r["down"].items() if k != "chol_codes"))
    assert m.bootstrap_bytes == {"up": r["up_mean"], "down": r["down_mean"]}
    assert 0 < r["final_cells_s2_mean"] < r["final_cells_mean"] and 0 < r["s2_row_cover"] < 1
    # histogram uplink: sparse (delta index, count) LEB128 varints, re-encoded byte by byte
    X, Y, off = env.train.X.numpy(), env.train.Y.numpy().astype(np.int64), env.train.off.numpy()
    total = 0
    for v in range(env.K):
        x, y = X[off[v]:off[v + 1]], Y[off[v]:off[v + 1]]
        b = np.clip(((x.astype(np.float32) + 4) * (64 / 8)).astype(np.int64), 0, 63)
        h = np.bincount(((y[:, None] * 77 + np.arange(77)) * 64 + b).ravel(), minlength=16 * 77 * 64)
        idx = np.nonzero(h)[0]
        out = bytearray()
        for val in list(np.diff(idx, prepend=0)) + list(h[idx]):
            val = int(val)
            while True:
                out.append((val & 127) | (128 if val > 127 else 0))
                val >>= 7
                if not val:
                    break
        total += len(out)
    assert r["up"]["hist"] == pytest.approx(total / env.K)
    # CDF downlink: adjacent differences of the uint16 grid decode back to the tables the clients sample from
    codes = np.round(m.cop_cdf.double().numpy() * 65535).astype(np.int64)
    diff = np.diff(codes, axis=-1, prepend=0)
    assert diff.min() >= 0 and diff.max() < 65536
    assert torch.equal(torch.from_numpy(np.cumsum(diff, -1) / 65535).float(), m.cop_cdf)
    assert r["down"]["cdf"] == 28 + mobihet_lite.pack(diff.astype("<u2").tobytes())
    # 5-bit factors: strict lower part on a 31-level grid per row; means on a 255-level grid per component
    L = m.km_L.double()
    low = L - torch.diag_embed(torch.diagonal(L, dim1=-2, dim2=-1))
    for row in low.reshape(-1, 77)[::97]:
        if row.abs().max() > 0:
            assert row.unique().numel() <= 31
    assert all(v.unique().numel() <= 255 for v in m.km_mu.reshape(-1, 77))
    for r_ in (1, 2):
        info = m._round_private(r_, .001)
        assert info["clients_without_applied"] == 0 and np.isfinite(info["train_loss"])


def test_km2_tau_above_every_cell_falls_back_to_class_covariance():
    a, b = mobihet_lite.Method(big(**KM2, km2_tau=10 ** 9)), mobihet_lite.Method(big(**KM2))
    assert torch.equal(a.km_p, b.km_p) and torch.equal(a.km_mu, b.km_mu)       # weights and means use every cell
    assert a.codec_bytes["final_cells_s2_mean"] == 0 and a.codec_bytes["up"]["final"] < b.codec_bytes["up"]["final"]
    assert torch.equal(a.km_L[:, 0], a.km_L[:, 1])                             # every component = the class covariance


def test_km2_sampled_passes_cost_fewer_bytes():
    full, part = mobihet_lite.Method(big(**KM2)).codec_bytes, mobihet_lite.Method(big(**KM2, km2_frac=.5)).codec_bytes
    assert 0 < part["up"]["cls"] <= full["up"]["cls"] and 0 < part["up"]["lloyd"] < full["up"]["lloyd"]   # 6 vehicles: pass 0 may draw all
    assert part["down"]["centres"] < full["down"]["centres"] and part["up"]["hist"] == full["up"]["hist"]


def test_bins_need_km2():
    with pytest.raises(AssertionError):
        mobihet_lite.Method(big(**T3, cop_bins=64))


@pytest.mark.parametrize("soft", [False, True])
def test_eight_row_vehicle_rotates_classes(soft):
    env = environment(synth_tier_min=[10 ** 9], synth_tier_rows=[[64, 64], [16, 8]], synth_late_from=1)
    env.cfg = dict(env.cfg, dropout=0.0)
    m = mobihet_compute.Method(env)
    m.Q = torch.eye(16) * .9 + .1 / 16 if soft else None
    for step in (0, 1, 2):
        ctx = _ctx(m)
        ctx.step = step
        m._rows_k = torch.tensor([64, 8, 16])
        loss, _ = m._loss_tier(ctx)
        g = torch.Generator().manual_seed(1)
        sl = slice(8, 16) if step % 2 else slice(0, 8)
        xg = torch.randn(3, 64, 77, generator=g)[:, sl]
        lab = (torch.arange(64) % 16)[sl][None].expand(3, -1)
        assert sorted(lab[0].tolist()) == list(range(8, 16) if step % 2 else range(8))
        mask = torch.cat([ctx.mask, torch.ones(3, 8, dtype=torch.bool)], 1)
        z, _ = grouped.forward(ctx.P, ctx.B, torch.cat([ctx.x, xg], 1), env.cfg, mask=mask, train=True)
        ce = local.masked_mean(F.cross_entropy(z[:, :8].reshape(-1, 16), ctx.y.reshape(-1), reduction="none").view(3, 8),
                               ctx.mask)
        if soft:
            lt = m.Q.clamp_min(1e-6).log()[lab]
            syn = (lt.exp() * (lt - F.log_softmax(z[:, 8:], -1))).sum(-1).mean(1)
        else:
            syn = F.cross_entropy(z[:, 8:].reshape(-1, 16), lab.reshape(-1), reduction="none").view(3, 8).mean(1)
        torch.testing.assert_close(loss[1], (ce + syn)[1])
