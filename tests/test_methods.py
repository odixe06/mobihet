"""Every method runs 2 rounds on a toy CPU environment, produces finite models and per-client/global confusion matrices,
and its state round-trips through state()/load_state()."""
import copy
import importlib

import numpy as np
import pytest
import torch

from core import evaluate, local
from core.dagsnet import width
from core.data import Split
from core.flat import Layout

METHODS = ["fedavg", "mobihet"]
ARMS = ["dml", "srv", "cheap", "none"]


class ToyEnv:
    def __init__(self, K=6, seed=0):
        rng = np.random.default_rng(seed)
        self.cfg = width(0.25)
        self.hp = {"batch": 8, "group_rows": 32, "weight_decay": 1e-4, "clip": 1.0, "amp": False, "seed": 42,
                   "lr_max": 1e-3, "lr_min": 1e-5, "rounds": 2}
        self.lay = Layout(self.cfg)
        self.dev = torch.device("cpu")
        self.world, self.rank = 1, 0

        def split(lo, hi):
            n = rng.integers(lo, hi, K)
            off = np.concatenate([[0], np.cumsum(n)])
            y = torch.from_numpy(rng.integers(0, 16, off[-1]).astype(np.int8))
            x = torch.from_numpy(rng.standard_normal((int(off[-1]), 77))).float() + 0.3 * y[:, None].float()
            return Split(x, y, torch.from_numpy(off), torch.arange(K))
        self.train, self.test = split(10, 50), split(4, 12)
        self.K = K
        self.ids = np.arange(K)
        self.all_ids = np.arange(K)
        self.n = torch.as_tensor(self.train.sizes, dtype=torch.float32)
        self.groups = local.group_schedule(self.train.sizes, list(range(K)), self.hp["batch"], self.hp["group_rows"])
        self.gen = torch.Generator().manual_seed(0)

    def allreduce(self, t):
        return t

    def gather_cms(self, cm):
        return np.asarray(cm)

    def groups_idx(self):
        for g in self.groups:
            yield torch.as_tensor(g)

    def eval_global(self, p, b):
        return evaluate.global_cm(self.lay, p, b, self.test, self.cfg)

    def eval_clients(self, bp, bb):
        return evaluate.own_cm(self.lay, bp, bb, self.test, self.cfg, torch.arange(self.K))


@pytest.mark.parametrize("name", METHODS)
def test_method_runs(name):
    env = ToyEnv()
    M = importlib.import_module(f"methods.{name}").Method(env)
    for r in (1, 2):
        info = M.round(r, 1e-3)
        assert np.isfinite(info["train_loss"])
    views = M.evaluate()
    for v, cms in views.items():
        assert cms.shape == (env.K, 16, 16)
        assert cms.sum() == len(env.test.Y), v
    s = M.state()
    M2 = importlib.import_module(f"methods.{name}").Method(ToyEnv())
    M2.load_state(s)
    for v in views:
        assert (M2.evaluate()[v] == views[v]).all()


@pytest.mark.parametrize("arm", ARMS)
def test_mobihet_arms(arm):
    env = ToyEnv()
    env.hp.update(arm=arm, m_transfer=64, server_steps=5)
    M = importlib.import_module("methods.mobihet").Method(env)
    for r in (1, 2):
        info = M.round(r, 1e-3)
        assert np.isfinite(info["train_loss"]) and info["bytes_up"] > 0, info
    views = M.evaluate()
    assert all(c.sum() == len(env.test.Y) for c in views.values())
    if arm != "dml":
        assert ("proxy" in views) == (arm != "none")
    M2 = importlib.import_module("methods.mobihet").Method(ToyEnv())
    M2.env.hp.update(arm=arm, m_transfer=64, server_steps=5)
    M2.load_state(M.state())
    assert (M2.evaluate()["private"] == views["private"]).all()


@pytest.mark.parametrize("synth", ["off", "oracle", "short", "mix", "copula", "copula_mix", "copula_cluster"])
def test_mobihet_synth(synth, monkeypatch):
    env = ToyEnv()
    env.hp.update(arm="none", synth=synth)
    cls = importlib.import_module("methods.mobihet").Method
    if synth == "copula_cluster":
        monkeypatch.setattr(cls, "_client_clusters", lambda self: (torch.arange(env.K) % 8, 8))
    M = cls(env)
    ctx = type("C", (), {})()
    ctx.Gfix, ctx.clients = 2, torch.tensor([0, 1])
    ctx.x, ctx.y = env.train.X[:2 * 64].view(2, 64, -1), env.train.Y[:2 * 64].long().view(2, 64)
    ctx.mask = torch.ones(2, 64, dtype=torch.bool)
    if synth != "off":
        xg, lab = M._synth(ctx)
        assert xg.shape == (2, M.aux, ctx.x.shape[-1]) and lab.shape == (2, M.aux) and torch.isfinite(xg.float()).all()
        if synth in ("copula", "copula_mix", "copula_cluster", "mix", "short"):
            assert xg.float().abs().max() <= 4 + 1e-3
    for r in (1, 2):
        assert np.isfinite(M.round(r, 1e-3)["train_loss"])


def test_cluster_copula_never_samples_empty_component(monkeypatch):
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula_cluster", synth_rows=64)
    cls = importlib.import_module("methods.mobihet").Method
    # All clients occupy group 3. Other seven components must carry zero probability.
    monkeypatch.setattr(cls, "_client_clusters", lambda self: (torch.full((env.K,), 3), 8))
    M = cls(env)
    assert torch.equal(M.cop_cluster_prob[:, 3], torch.ones(16))
    assert M.cop_cluster_prob.sum() == 16
    assert M.cop_cdf.shape == (16 * 8, 77, 256)
    assert np.isfinite(M.round(1, 1e-3)["train_loss"])


@pytest.mark.parametrize("extra", [{"synth_rows": 64}, {"synth_target": "onehot"}])
def test_synthetic_ablation_resume(extra):
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", **extra)
    cls = importlib.import_module("methods.mobihet").Method
    M = cls(env)
    assert np.isfinite(M.round(1, 1e-3)["train_loss"])
    state = copy.deepcopy(M.state())
    M2 = cls(env)
    M2.load_state(state)
    # Each copy continues from the same method/environment and global torch RNG states.
    rng, eg, gg = torch.get_rng_state(), env.gen.get_state(), M.gen_g.get_state()
    M.round(2, 1e-3)
    torch.set_rng_state(rng); env.gen.set_state(eg); M2.gen_g.set_state(gg)
    M2.round(2, 1e-3)
    assert torch.equal(M.priv_p, M2.priv_p)
    assert torch.equal(M.priv_b, M2.priv_b)
    assert torch.equal(M.Q, M2.Q)


def test_onehot_changes_synthetic_loss_only(monkeypatch):
    from types import SimpleNamespace
    import torch.nn.functional as F
    from methods import mobihet

    env = ToyEnv()
    env.hp.update(arm="none", synth="copula")
    M = mobihet.Method(env)
    G, b, a = 2, 8, M.aux
    ctx = SimpleNamespace(Gfix=G, x=env.train.X[:G * b].view(G, b, -1),
                          y=env.train.Y[:G * b].long().view(G, b), mask=torch.ones(G, b, dtype=torch.bool), P={}, B={})
    lab = M.labels[None].expand(G, a)
    logits = torch.randn(G, b + a, 16)
    monkeypatch.setattr(M, "_synth", lambda ctx: (torch.zeros(G, a, 77), lab))
    monkeypatch.setattr(mobihet.grouped, "TRAIN", lambda *args, **kwargs: (logits, {}))
    M._shift = torch.zeros(G, 16)
    M.Q = torch.full((16, 16), 1 / 16)
    M._csum = torch.zeros(G, 16, 16)
    soft = M._loss_private(ctx)[0]
    M.synth_target = "onehot"
    hard = M._loss_private(ctx)[0]
    lsp = F.log_softmax(logits, -1)
    ce = F.cross_entropy(logits[:, :b].reshape(-1, 16), ctx.y.reshape(-1), reduction="none").view(G, b).mean(1)
    raw = 1 / (ce + torch.tensor(16.0).log())
    soft_g = ((1 / 16) * (-torch.tensor(16.0).log() - lsp[:, b:])).sum(-1).mean(1)
    hard_g = -lsp[:, b:].gather(-1, lab[..., None]).squeeze(-1).mean(1)
    torch.testing.assert_close(hard - soft, raw * (hard_g - soft_g))


@pytest.mark.parametrize("anchor,synth", [("class", "copula"), ("ensemble", "copula"), ("ensemble", "gauss")])
def test_anchor_consensus(anchor, synth):
    from methods.mobihet import Method
    env = ToyEnv()
    env.hp.update(arm="none", synth=synth, anchor=anchor, m_transfer=64)
    m = Method(env)
    # Shared support cannot depend on rank-specific streaming-sampler seed.
    env2 = ToyEnv()
    env2.hp.update(env.hp)
    env2.rank = 1
    replica = Method(env2)
    assert torch.equal(m.anchor_X, replica.anchor_X)
    for r in (1, 2):
        info = m.round(r, 1e-3)
        assert np.isfinite(info['train_loss'])
        extra = 32 + 64 * 16 if anchor == "ensemble" else 0
        assert info['bytes_up'] == info['bytes_down'] == 544 + extra
        if anchor == "ensemble":
            assert m.anchor_Q.shape == (64, 16)
            torch.testing.assert_close(m.anchor_Q.sum(-1), torch.ones(64))
            # Rebuild stored consensus from post-epoch privates and decoded wire values.
            p = (m._probs(m.priv_p, m.priv_b, m.anchor_X, env.K) * 255).round()
            p = p / p.sum(-1, keepdim=True).clamp_min(1)
            w = m.cnt[:, m.anchor_y].double()
            target = (w[..., None] * p.double()).sum(0) / w.sum(0)[:, None].clamp_min(1)
            q8 = (target * 255).round()
            torch.testing.assert_close(m.anchor_Q, (q8 / q8.sum(-1, keepdim=True).clamp_min(1)).float())
    state = copy.deepcopy(m.state())
    replica.load_state(state)
    assert np.array_equal(m.evaluate()['private'], replica.evaluate()['private'])
    if anchor == "ensemble":
        assert torch.equal(m.anchor_Q, replica.anchor_Q)


def test_anchor_indices_keep_sample_target_identity():
    from types import SimpleNamespace
    from methods.mobihet import Method
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", anchor="ensemble", m_transfer=64)
    m = Method(env)
    ctx = SimpleNamespace(Gfix=2, x=env.train.X[:1])
    x, y = m._synth(ctx)
    assert torch.equal(x, m.anchor_X[m._anchor_idx])
    assert torch.equal(y, m.anchor_y[m._anchor_idx])
    assert torch.equal(y, m.labels[None].expand(2, -1))


def test_real_kd_ablation_runs():
    from methods.mobihet import Method
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", real_kd=False)
    m = Method(env)
    for r in (1, 2):
        assert np.isfinite(m.round(r, 1e-3)['train_loss'])


def test_augmentation_only_has_no_learned_teacher_or_packets():
    from methods.mobihet import Method
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", synth_rows=64, real_kd=False, no_kd=True)
    m = Method(env)
    for r in (1, 2):
        info = m.round(r, 1e-3)
        assert np.isfinite(info['train_loss'])
        assert info['bytes_up'] == info['bytes_down'] == 0
        assert m.Q is None and m.bucket_Q is None
    assert sum(m.bootstrap_bytes.values()) > 0


def test_bucket_teacher_decodes_wire_and_falls_back_for_empty_buckets():
    from methods.mobihet import Method
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", kd_bins=8, real_kd=False, synth_rows=64)
    m = Method(env)
    assert torch.equal(m.bucket_cnt.sum(-1), m.cnt)
    for r in (1, 2):
        info = m.round(r, 1e-3)
        assert np.isfinite(info['train_loss'])
        assert info['bytes_up'] == info['bytes_down'] == 4128
        torch.testing.assert_close(m.bucket_Q.sum(-1), torch.ones(16, 8))
        torch.testing.assert_close(m.Q.sum(-1), torch.ones(16))
    # Exercise empty bucket fallback against a stored sum/count calculation.
    count = m.bucket_tot.clone()
    count[:, 0] = 0
    sums = torch.rand(16, 8, 16).double() * count[..., None]
    sums[:, 0] = 0
    m.bucket_tot = count
    marginal = sums.sum(1) / count.sum(1)[:, None].clamp_min(1)
    expected = marginal.half().float()
    expected /= expected.sum(-1, keepdim=True)
    m._aggregate_buckets(sums)
    torch.testing.assert_close(m.bucket_Q[:, 0], expected)
    env2 = ToyEnv()
    env2.hp.update(env.hp)
    clone = Method(env2)
    clone.load_state(m.state())
    assert torch.equal(clone.bucket_Q, m.bucket_Q)
    assert torch.equal(clone.router, m.router)


def test_bucket_router_fixed_across_training_seed():
    from methods.mobihet import Method
    env = ToyEnv()
    env.hp.update(arm="none", kd_bins=8)
    a = Method(env)
    env.hp['seed'] = 43
    b = Method(env)
    assert torch.equal(a.router, b.router)


def test_real_kd_gate_removes_only_wrong_teacher_rows(monkeypatch):
    from types import SimpleNamespace
    from methods import mobihet
    import torch.nn.functional as F
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", synth_rows=16)
    m = mobihet.Method(env)
    G, b, a = 2, 8, 16
    y = torch.arange(G * b).view(G, b) % 16
    ctx = SimpleNamespace(Gfix=G, x=torch.zeros(G, b, 77), y=y, mask=torch.ones(G, b, dtype=torch.bool), P={}, B={})
    lab = torch.arange(a)[None].expand(G, a)
    logits = torch.randn(G, b + a, 16)
    q = torch.eye(16) * .8 + .2 / 16
    q[13] = q[0]
    m.Q, m._shift = q, torch.zeros(G, 16)
    monkeypatch.setattr(m, '_synth', lambda ctx: (torch.zeros(G, a, 77), lab))
    monkeypatch.setattr(mobihet.grouped, 'TRAIN', lambda *args, **kwargs: (logits, {}))
    m._csum = torch.zeros(G, 16, 16)
    base = m._loss_private(ctx)[0]
    m.real_kd_gate = True
    gated = m._loss_private(ctx)[0]
    lt, lp = q[y].clamp_min(1e-6).log(), F.log_softmax(logits[:, :b], -1)
    removed = ((lt.exp() * (lt - lp)).sum(-1) * (y == 13)).mean(1)
    ce = F.cross_entropy(logits[:, :b].reshape(-1, 16), y.reshape(-1), reduction='none').view(G, b).mean(1)
    ce_t = -lt.gather(-1, y[..., None]).squeeze(-1).mean(1)
    torch.testing.assert_close(base - gated, removed / (ce + ce_t))
