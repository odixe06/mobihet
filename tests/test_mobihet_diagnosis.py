"""Causal loss cells, decoded wire precision and non-mutating shadow diagnostics."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from core import local
from methods import mobihet, mobihet_diagnostics
from test_methods import ToyEnv


@pytest.mark.parametrize("target", ["soft", "onehot"])
@pytest.mark.parametrize("weight", ["adaptive", "fixed"])
@pytest.mark.parametrize("gate", [False, True])
def test_factorial_loss_matches_independent_formula(monkeypatch, target, weight, gate):
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", synth_target=target, kd_weight=weight,
                  real_kd=gate, real_kd_gate=gate)
    m = mobihet.Method(env)
    G, b, a = 2, 8, m.aux
    ctx = SimpleNamespace(Gfix=G, x=torch.zeros(G, b, 77),
                          y=torch.arange(G * b).view(G, b) % 16,
                          mask=torch.ones(G, b, dtype=torch.bool), P={}, B={})
    logits = torch.randn(G, b + a, 16, requires_grad=True)
    lab = m.labels[None].expand(G, a)
    monkeypatch.setattr(m, "_synth", lambda ctx: (torch.zeros(G, a, 77), lab))
    monkeypatch.setattr(mobihet.grouped, "TRAIN", lambda *args, **kwargs: (logits, {}))
    m._shift = torch.randn(G, 16) * .2
    q = torch.eye(16) * .65 + .35 / 16
    q[13] = q[0].clone()  # Confidently wrong teacher: gate must reject this class.
    m.Q, m._csum = q, torch.zeros(G, 16, 16)
    got = m._loss_private(ctx)[0]
    p = logits.log_softmax(-1)
    ce = F.cross_entropy((logits[:, :b] + m._shift[:, None]).reshape(-1, 16),
                         ctx.y.flatten(), reduction="none").view(G, b).mean(1)
    tr = q[ctx.y]
    lr = (tr * (tr.log() - p[:, :b])).sum(-1)
    lr = torch.where(tr.argmax(-1) == ctx.y, lr, 0).mean(1) if gate else torch.zeros(G)
    ts = q[lab]
    ls = -(p[:, b:].gather(-1, lab[..., None])).squeeze(-1).mean(1) if target == "onehot" else (
        ts * (ts.log() - p[:, b:])).sum(-1).mean(1)
    ct = -tr.gather(-1, ctx.y[..., None]).log().squeeze(-1).mean(1)
    coef = torch.ones(G) if weight == "fixed" else 1 / (ce + ct).detach()
    expected = ce + coef * (lr + ls)
    torch.testing.assert_close(got, expected)
    actual_grad = torch.autograd.grad(got.sum(), logits, retain_graph=True)[0]
    expect_grad = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(actual_grad, expect_grad)


def test_class_teacher_aggregates_decoded_upload_and_download(monkeypatch):
    env = ToyEnv()
    env.hp.update(arm="none", q_codec="mean_fp16")
    m = mobihet.Method(env)
    q = torch.rand(env.K, 16, 16)
    q /= q.sum(-1, keepdim=True)
    sums = q * m.cnt[..., None]

    def epoch(lay, flat, bufs, data, idx, *args, **kwargs):
        m._csum[:len(idx)] = sums[idx]
        return {"loss": torch.zeros(len(idx)), "applied": torch.ones(len(idx)), "skipped": torch.zeros(len(idx))}

    monkeypatch.setattr(local, "local_epoch", epoch)
    info = m.round(1, .001)
    # Explicit serialization on CPU is independent of the tensor cast used in the method.
    packet = (sums / m.cnt[..., None].clamp_min(1)).half().numpy().tobytes()
    decoded = torch.from_numpy(np.frombuffer(packet, dtype=np.float16).copy().reshape(env.K, 16, 16)).float()
    expected = (decoded.double() * m.cnt[..., None].double()).sum(0) / m.cnt.sum(0)[:, None].double()
    down = expected.float().half().numpy().tobytes()
    expected = torch.from_numpy(np.frombuffer(down, dtype=np.float16).copy().reshape(16, 16)).float()
    expected /= expected.sum(-1, keepdim=True)
    assert torch.equal(m.Q, expected)
    assert len(packet) == env.K * 512 and len(down) == 512
    assert info["bytes_up"] == info["bytes_down"] == 544


def test_shadow_diagnostics_preserve_training_state_and_rng(tmp_path):
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", synth_rows=16, q_codec="mean_fp16",
                  diagnostic_ids=[0, 1])
    m = mobihet.Method(env)
    m.round(1, .001)
    before = copy.deepcopy(m.state())
    rng = torch.get_rng_state().clone()
    eg, gg = env.gen.get_state().clone(), m.gen_g.get_state().clone()
    result = mobihet_diagnostics.run(m, 2, m.Q, tmp_path)
    assert result["train_private/f1_macro"] >= 0
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(eg, env.gen.get_state()) and torch.equal(gg, m.gen_g.get_state())
    for key, value in before.items():
        if torch.is_tensor(value):
            assert torch.equal(value, m.state()[key]), key
    with np.load(tmp_path / "train_cms_r002.npz") as z:
        assert z["private"].sum() == len(env.train.Y)
    assert (tmp_path / "diagnostics_rank0.jsonl").exists()
