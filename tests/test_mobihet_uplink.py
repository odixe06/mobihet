"""Same-class interpolation, real-logit statistics and decoded on-wire aggregation."""
from types import SimpleNamespace
import zlib

import numpy as np
import pytest
import torch

from core import grouped, local
from methods import mobihet_uplink as U
from methods.mobihet_refine import Method as CBB
from test_mobihet_refine import environment


def test_default_two_rounds_are_bit_exact_cbb():
    a, b = CBB(environment()), U.Method(environment())
    for r in (1, 2):
        rng = torch.get_rng_state().clone()
        ia = a._round_private(r, .001)
        after = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        ib = b._round_private(r, .001)
        assert torch.equal(after, torch.get_rng_state())
        assert all(torch.equal(getattr(a, k), getattr(b, k)) for k in ("priv_p", "priv_b", "Q"))
        assert ia["bytes_up"] == ib["bytes_up"] and ia["bytes_down"] == ib["bytes_down"]


def test_mix_uses_only_matching_real_rows_with_exact_fallback(monkeypatch):
    x = torch.randn(3, 64, 77)
    lab = torch.arange(64).remainder(16)[None].expand(3, -1)
    monkeypatch.setattr(CBB, "_synth", lambda self, ctx: (x, lab))
    m = U.Method(environment(same_class_mix=.5))
    ctx = SimpleNamespace(x=torch.randn(3, 8, 77), y=torch.arange(8)[None].expand(3, -1),
                          mask=torch.tensor([[True] * 8, [False] * 8, [True] + [False] * 7]))
    mixed, labels = m._synth(ctx)
    assert torch.equal(lab, labels) and torch.equal(mixed[1], x[1])
    for g in range(3):
        for j, y in enumerate(lab[g]):
            valid = (ctx.y[g] == y) & ctx.mask[g]
            if valid.any():
                torch.testing.assert_close(mixed[g, j], .5 * (x[g, j] + ctx.x[g, valid][0]))
            else:
                assert torch.equal(mixed[g, j], x[g, j])
    m.mix = 0
    state = m.gen_g.get_state().clone()
    got, _ = m._synth(ctx)
    assert torch.equal(got, x) and torch.equal(state, m.gen_g.get_state())


def test_lg_collects_centered_log_probabilities_only_on_real_rows(monkeypatch):
    m = U.Method(environment(uplink_mode="lg16"))
    G, b, a = 3, 8, 64
    ctx = SimpleNamespace(Gfix=G, x=torch.zeros(G, b, 77), y=torch.arange(G * b).view(G, b) % 16,
                          mask=torch.tensor([[True] * b, [True] * 3 + [False] * 5, [False] * b]),
                          active=torch.tensor([True, True, False]), clients=torch.arange(G), P={}, B={})
    z = torch.randn(G, b + a, 16, requires_grad=True)
    lab = torch.arange(a).remainder(16)[None].expand(G, -1)
    monkeypatch.setattr(m, "_synth", lambda ctx: (torch.zeros(G, a, 77), lab))
    monkeypatch.setattr(grouped, "TRAIN", lambda *args, **kw: (z, {}))
    m._shift, m._csum = torch.zeros(G, 16), torch.zeros(G, 16, 16)
    m.Q = torch.eye(16) * .9 + .1 / 16
    loss = m._loss_stream(ctx)[0]
    expected = torch.zeros(G, 16, 16)
    centered = z[:, :b].detach() - z[:, :b].detach().mean(-1, keepdim=True)
    for g in range(G):
        expected[g].index_add_(0, ctx.y[g, ctx.mask[g]], centered[g, ctx.mask[g]])
    torch.testing.assert_close(m._logsum, expected)
    assert torch.isfinite(torch.autograd.grad(loss[:2].sum(), z)[0]).all()


@pytest.mark.parametrize("mode", ["gp16", "lg16"])
def test_actual_frames_reproduce_teacher_bytes_and_state(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("MOBIHET_RECOVERY_DIR", str(tmp_path / "recovery"))
    m = U.Method(environment(uplink_mode=mode, uplink_audit=True))
    m.cnt[0, 0], m.cnt[1, 1] = 8, 9  # both sides of the minimum-count boundary
    q = torch.rand(m.env.K, 16, 16)
    q /= q.sum(-1, keepdim=True)
    center = q.log() - q.log().mean(-1, keepdim=True)

    def epoch(lay, flat, bufs, data, idx, *args, **kw):
        G = len(idx)
        m._csum[:G] = q[idx] * m.cnt[idx, :, None]
        m._logsum = torch.zeros_like(m._csum)
        m._logsum[:G] = center[idx] * m.cnt[idx, :, None]
        return {"loss": torch.zeros(G), "applied": torch.ones(G), "skipped": torch.zeros(G)}

    monkeypatch.setattr(local, "local_epoch", epoch)
    info = m._round_private(1, .001)
    z = np.load(tmp_path / "wire/rank0_r001.npz")
    total = torch.zeros(16, 16, dtype=torch.float64)
    wire = z["frames"].tobytes()
    for i, count in enumerate(z["counts"]):
        f = wire[z["offsets"][i]:z["offsets"][i + 1]]
        present = np.array([bool(int.from_bytes(f[32:34], "little") & (1 << c)) for c in range(16)])
        eligible = np.array([bool(int.from_bytes(f[34:36], "little") & (1 << c)) for c in range(16)])
        assert np.array_equal(present, count > 0) and np.array_equal(eligible, count >= 8)
        raw = zlib.decompress(f[37:]) if f[36] else f[37:]
        split = 32 * int(eligible.sum())
        values = torch.zeros(16, 16)
        big = torch.from_numpy(np.frombuffer(raw[:split], "<f2").astype(np.float32))
        values[eligible] = big.view(-1, 16)
        if mode == "gp16":
            values = values.clamp_min(1e-6).log()
            values -= values.mean(-1, keepdim=True)
        small = present & ~eligible
        p = torch.from_numpy(np.frombuffer(raw[split:], np.uint8).copy()).float().view(-1, 16) / 255
        lp = p.clamp_min(1e-6).log()
        values[small] = lp - lp.mean(-1, keepdim=True)
        total += values.double() * torch.from_numpy(count)[:, None]
    torch.testing.assert_close(total, torch.from_numpy(z["global_sum"]), rtol=0, atol=1e-12)
    p = (total / m.cnt.sum(0).double()[:, None]).float().softmax(-1)
    codes = (p * 255).round()
    assert torch.equal(m.Q, codes / codes.sum(-1, keepdim=True))
    assert info["bytes_up"] == len(wire) / m.env.K and info["bytes_down"] == len(z["down"])
    restored = U.Method(environment(uplink_mode=mode))
    restored.load_state(m.state())
    assert torch.equal(restored.Q, m.Q)
    assert local.local_epoch is epoch
