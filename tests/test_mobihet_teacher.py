"""Teacher collection and class-support personalization, independent of performance claims."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from core import grouped, local
from methods import mobihet, mobihet_diagnostics
from methods.mobihet_teacher import class_probability_sums
from test_methods import ToyEnv


@pytest.mark.parametrize("support", ["off", "uniform", "local"])
@pytest.mark.parametrize("real_kd", [False, True])
def test_fallback_personalization_loss_and_gradient(monkeypatch, support, real_kd):
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", synth_rows=16, synth_target="fallback",
                  kd_weight="fixed", kd_support=support, real_kd=real_kd, real_kd_gate=real_kd)
    m = mobihet.Method(env)
    G, b, a = 2, 8, m.aux
    clients = torch.tensor([0, 1])
    y = torch.arange(G * b).view(G, b) % 16
    lab = torch.arange(a)[None].expand(G, -1)
    ctx = SimpleNamespace(Gfix=G, x=torch.zeros(G, b, 77), y=y,
                          clients=clients, mask=torch.ones(G, b, dtype=torch.bool), P={}, B={})
    ctx.mask[0, -2:] = False
    logits = torch.randn(G, b + a, 16, requires_grad=True)
    monkeypatch.setattr(m, "_synth", lambda ctx: (torch.zeros(G, a, 77), lab))
    monkeypatch.setattr(grouped, "TRAIN", lambda *args, **kw: (logits, {}))
    m._shift = torch.randn(G, 16) * .2
    q = torch.eye(16) * .65 + .35 / 16
    q[13] = q[0].clone()
    m.Q, m._csum = q, torch.zeros(G, 16, 16)
    # Ensure absent, sparse and abundant classes exist, with distinct weights on the two clients.
    m.cnt[:G, :3] = torch.tensor([[0., 1., 80.], [80., 1., 0.]])
    got = m._loss_private(ctx)[0]
    p = logits.log_softmax(-1)
    count = m.cnt[clients]
    weight = torch.ones_like(count) if support == "off" else (
        torch.full_like(count, .5) if support == "uniform" else env.hp["batch"] / (env.hp["batch"] + count))
    ce_rows = F.cross_entropy((logits[:, :b] + m._shift[:, None]).reshape(-1, 16),
                             y.flatten(), reduction="none").view(G, b)
    ce = (ce_rows * ctx.mask).sum(1) / ctx.mask.sum(1)
    tr = q[y]
    rrows = (tr * (tr.log() - p[:, :b])).sum(-1)
    rrows = rrows * weight.gather(1, y) * (tr.argmax(-1) == y)
    real = (rrows * ctx.mask).sum(1) / ctx.mask.sum(1) if real_kd else torch.zeros(G)
    rho = weight.gather(1, lab) * (q[lab].argmax(-1) == lab)
    target = rho[..., None] * q[lab] + (1 - rho[..., None]) * F.one_hot(lab, 16)
    # Cross entropy minus target entropy, calculated separately from the implementation expression.
    synth = -(target * p[:, b:]).sum(-1) + (target * target.clamp_min(1e-6).log()).sum(-1)
    expected = ce + real + synth.mean(1)
    torch.testing.assert_close(got, expected)
    gg = torch.autograd.grad(got.sum(), logits, retain_graph=True)[0]
    ge = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(gg, ge)
    assert torch.equal(target[:, 13], F.one_hot(lab[:, 13], 16).float())
    if support == "local":
        assert rho[0, 0] == 1 and rho[0, 0] > rho[0, 1] > rho[0, 2]


def test_postepoch_matches_per_client_reference_without_state_or_rng_changes():
    env = ToyEnv()
    env.hp.update(arm="none", q_codec="mean_fp16", q_collect="postepoch")
    m = mobihet.Method(env)
    m.round(1, .001)
    state = copy.deepcopy(m.state())
    rng, eg, gg = torch.get_rng_state().clone(), env.gen.get_state().clone(), m.gen_g.get_state().clone()
    sums, counts = class_probability_sums(env, m.priv_p, m.priv_b, max_rows=32)
    expected = torch.zeros_like(sums)
    for k in range(env.K):
        x = env.train.X[env.train.off[k]:env.train.off[k + 1]][None]
        y = env.train.Y[env.train.off[k]:env.train.off[k + 1]].long()
        z, _ = grouped.forward(env.lay.params(m.priv_p[k:k + 1]), env.lay.bufs(m.priv_b[k:k + 1]),
                               x, env.cfg, train=False)
        expected[k].index_add_(0, y, z[0].softmax(-1))
    torch.testing.assert_close(sums, expected)
    assert torch.equal(counts, m.cnt)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(eg, env.gen.get_state()) and torch.equal(gg, m.gen_g.get_state())
    for k, v in state.items():
        if torch.is_tensor(v):
            assert torch.equal(v, m.state()[k]), k


def test_postepoch_upload_replaces_online_statistics_and_decodes_both_ways(monkeypatch):
    env = ToyEnv()
    env.hp.update(arm="none", q_codec="mean_fp16", q_collect="postepoch")
    m = mobihet.Method(env)
    q = torch.rand(env.K, 16, 16)
    q /= q.sum(-1, keepdim=True)
    sums = q * m.cnt[..., None]

    def epoch(lay, flat, bufs, data, idx, *args, **kwargs):
        m._csum.fill_(999)  # Must never contaminate postepoch statistics.
        return {"loss": torch.zeros(len(idx)), "applied": torch.ones(len(idx)), "skipped": torch.zeros(len(idx))}

    monkeypatch.setattr(local, "local_epoch", epoch)
    monkeypatch.setattr("methods.mobihet_teacher.class_probability_sums", lambda *args: (sums, m.cnt))
    info = m.round(1, .001)
    wire = (sums / m.cnt[..., None].clamp_min(1)).half().numpy().tobytes()
    decoded = torch.from_numpy(np.frombuffer(wire, dtype=np.float16).copy().reshape(env.K, 16, 16)).float()
    expected = (decoded.double() * m.cnt[..., None].double()).sum(0) / m.cnt.sum(0)[:, None]
    down = expected.float().half().numpy().tobytes()
    expected = torch.from_numpy(np.frombuffer(down, dtype=np.float16).copy().reshape(16, 16)).float()
    expected /= expected.sum(1, keepdim=True)
    assert torch.equal(m.Q, expected)
    assert info["bytes_up"] == info["bytes_down"] == 544 and info["q_collect_seconds"] >= 0


def test_personalized_shadow_matches_loss_without_mutation(tmp_path):
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", synth_rows=16, q_codec="mean_fp16", q_collect="postepoch",
                  kd_weight="fixed", synth_target="fallback", kd_support="local", real_kd_gate=True,
                  diagnostic_ids=[0, 1], diagnostic_rounds=[15, 20])
    m = mobihet.Method(env)
    m.round(1, .001)
    state = copy.deepcopy(m.state())
    rng = torch.get_rng_state().clone()
    result = mobihet_diagnostics.run(m, 15, m.Q, tmp_path)
    assert result["train_private/f1_macro"] >= 0
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in state.items():
        if torch.is_tensor(v):
            assert torch.equal(v, m.state()[k]), k
    assert (tmp_path / "train_cms_r015.npz").exists()


def test_sqrt_count_server_aggregation_uses_class_support_and_same_packet(monkeypatch):
    env = ToyEnv()
    env.hp.update(arm="none", q_codec="mean_fp16", q_agg="sqrt_count")
    m = mobihet.Method(env)
    q = torch.rand(env.K, 16, 16)
    q /= q.sum(-1, keepdim=True)
    sums = q * m.cnt[..., None]

    def epoch(lay, flat, bufs, data, idx, *args, **kwargs):
        m._csum[:len(idx)] = sums[idx]
        return {"loss": torch.zeros(len(idx)), "applied": torch.ones(len(idx)), "skipped": torch.zeros(len(idx))}

    monkeypatch.setattr(local, "local_epoch", epoch)
    info = m.round(1, .001)
    wire = (sums / m.cnt[..., None].clamp_min(1)).half().numpy().tobytes()
    decoded = np.frombuffer(wire, dtype=np.float16).reshape(env.K, 16, 16).astype(np.float64)
    weight = np.sqrt(m.cnt.numpy().astype(np.float64))
    expected = (decoded * weight[..., None]).sum(0) / weight.sum(0)[:, None]
    expected = expected.astype(np.float32).astype(np.float16).astype(np.float32)
    expected /= expected.sum(1, keepdims=True)
    np.testing.assert_allclose(m.Q.numpy(), expected, rtol=2e-7, atol=1e-8)
    assert info["bytes_up"] == info["bytes_down"] == 544


def test_external_diagnostic_excludes_own_source_and_preserves_state(tmp_path, monkeypatch):
    import hashlib
    from methods import mobihet_external_diagnostic as external

    env = ToyEnv(K=4)
    env.hp.update(arm="none", diagnostic_ids=[0, 1], external_diagnostics=True)
    m = mobihet.Method(env)
    root = tmp_path / "npy"
    root.mkdir()
    y = np.tile(np.repeat(np.arange(16), 3), 4).astype(np.int8)
    source = np.repeat(np.arange(4), 48)
    x = np.zeros((len(y), 77), dtype=np.float16)
    x[:, 0], x[:, 1] = source, y
    for name, value in {"X": x, "Y": y, "off": np.arange(5) * 48, "ids": np.arange(4)}.items():
        np.save(root / f"test_{name}.npy", value)
    monkeypatch.setenv("V4_NPY", str(root))
    monkeypatch.setattr(grouped, "EVAL", lambda P, B, batch, cfg, **kw:
                        (F.one_hot(batch[:, :, 1].long(), 16).float() * 10, {}))
    state, rng = copy.deepcopy(m.state()), torch.get_rng_state().clone()
    external.run(m, 1, tmp_path)
    with np.load(tmp_path / "external_cms_rank0_r001.npz") as z:
        assert np.array_equal(z["ids"], [0, 1])
        assert z["private"].sum() == 2 * 16 * 9
        assert np.array_equal(z["private"], np.tile(np.eye(16, dtype=np.int64) * 9, (2, 1, 1)))
        for i, cid in enumerate(z["ids"]):
            digest = hashlib.sha256()
            for _, sources, rows in external.test_pool(root):
                selected = rows[sources != cid][:256]
                assert (source[selected] != cid).all()
                digest.update(selected.astype("<i8").tobytes())
            assert digest.hexdigest() == z["selection_sha256"][i]
    assert torch.equal(rng, torch.get_rng_state())
    for k, v in state.items():
        if torch.is_tensor(v):
            assert torch.equal(v, m.state()[k]), k
