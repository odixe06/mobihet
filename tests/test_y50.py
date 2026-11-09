"""Y50: prior-coordinate loss algebra, locked hp changes and decision rules (CPU, no data)."""
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from core import grouped
from methods import mobihet_prior
from methods.mobihet_wstudy import WMixin
from scripts import f50_common, y50_common as Y

G, B, A = 3, 5, 16


def _method(shift, q):
    m = mobihet_prior.Method.__new__(mobihet_prior.Method)
    g = torch.Generator().manual_seed(0)
    lab = (torch.arange(A) % 16)[None].expand(G, -1)
    m.env, m.aux, m.Q, m.w_synth = SimpleNamespace(cfg={}), A, q, "full"
    m.prior_shift, m._shift = shift, torch.randn(G, 16, generator=g)
    m._csum = torch.zeros(G, 16, 16)
    m._synth = lambda ctx: (torch.zeros(G, A, 1), lab)
    return m


def _ctx():
    g = torch.Generator().manual_seed(1)
    return SimpleNamespace(Gfix=G, x=torch.zeros(G, B, 1), y=torch.randint(16, (G, B), generator=g), P=None, B=None,
                           mask=torch.ones(G, B, dtype=torch.bool), active=torch.ones(G, dtype=torch.bool))


def _logits(monkeypatch):
    z = torch.randn(G, B + A, 16, generator=torch.Generator().manual_seed(2))
    monkeypatch.setattr(grouped, "TRAIN", lambda *a, **k: (z, None), raising=False)
    return z


def _q():
    q = torch.rand(16, 16, generator=torch.Generator().manual_seed(3)) + torch.eye(16) * 4
    return q / q.sum(-1, keepdim=True)


def test_zero_shift_is_core_loss_and_q_sum(monkeypatch):
    _logits(monkeypatch)
    for q in (None, _q()):
        a, b = _method(torch.zeros(16), q), _method(torch.zeros(16), q)
        la, _ = mobihet_prior.Method._loss_stream(a, _ctx())
        lb, _ = WMixin._loss_stream(b, _ctx())
        torch.testing.assert_close(la, lb, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(a._csum, b._csum, rtol=1e-6, atol=1e-6)


def test_shift_acts_on_synthetic_and_q_not_on_real_ce(monkeypatch):
    z = _logits(monkeypatch)
    s = torch.linspace(-1.4, 2.3, 16)
    q = _q()
    m, ctx = _method(s, q), _ctx()
    got, _ = mobihet_prior.Method._loss_stream(m, ctx)
    ce = F.cross_entropy((z[:, :B] + m._shift[:, None]).reshape(-1, 16), ctx.y.reshape(-1), reduction="none").view(G, B).mean(1)
    lab = (torch.arange(A) % 16)[None].expand(G, -1)
    t = q[lab]
    kl = (t * (t.log() - F.log_softmax(z[:, B:] + s, -1))).sum(-1).mean(1)
    torch.testing.assert_close(got, ce + kl, rtol=1e-5, atol=1e-5)
    want = torch.zeros(G, 16, 16)
    want.index_put_((torch.arange(G)[:, None].expand_as(ctx.y), ctx.y), F.softmax(z[:, :B] + s, -1), accumulate=True)
    torch.testing.assert_close(m._csum, want)



def _diff(a, b):
    return sorted(set(a) ^ set(b))


def test_each_cell_changes_one_thing():
    core = f50_common.hp_for("core", 42)
    assert _diff(Y.hp_for("prior", 42), core) == []
    assert _diff(Y.hp_for("nofloor", 42), core) == ["synth_step_floor=16"]
    assert _diff(Y.hp_for("oracle", 42), core) == ["synth=copula", "synth=oracle"]
    assert _diff(Y.hp_for("noq", 43), f50_common.hp_for("core", 43)) == ["no_kd=false", "no_kd=true"]


def _d(**kw):
    return {"accuracy": 0., "f1_macro": 0., "other15": .001, "benign": .001, "p10": 0., "low256": 0., "timeDelay": 0., "absent": 0.} | kw


def test_prior_rule():
    assert Y.prior_rule([_d(), _d()])["pass"]
    assert not Y.prior_rule([_d(), _d(benign=0.)])["pass"]
    assert not Y.prior_rule([_d(), _d(timeDelay=-.02)])["pass"]
    assert not Y.prior_rule([_d(absent=-.01), _d()])["pass"]
    assert not Y.prior_rule([_d(p10=-.006), _d()])["pass"]


def test_ablation_and_oracle_rules():
    assert Y.ablation_rule([_d(), _d()])["verdict"] == "removable"
    assert Y.ablation_rule([_d(p10=-.007), _d()])["verdict"] == "unresolved"
    assert Y.ablation_rule([_d(low256=-.004), _d(low256=-.004)])["verdict"] == "needed"
    assert Y.oracle_rule(_d(f1_macro=.01)) == "generator_worth_it"
    assert Y.oracle_rule(_d(f1_macro=.0001)) == "stop_generator"
    assert Y.oracle_rule(_d(f1_macro=.002)) == "small_ceiling"


def test_short_quota_reserves_setup_round_and_margin():
    from scripts.run_y50 import short_cap
    assert short_cap(1.0, 7.0) == 0
    assert short_cap(1.45, 7.0) == 1.45 / 1.15
    assert short_cap(10.0, 2.0) == 2.0
