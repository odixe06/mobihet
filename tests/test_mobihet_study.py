"""Scientific controls, information alignment and decoded teacher carriers for P20."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core import grouped
from methods.mobihet import Method as Base
from methods.mobihet_study import Method, reception, teacher_signature
from methods.mobihet_study_wire import encode, decode, roundtrip
from scripts.p20_common import CELLS, settings
from test_methods import ToyEnv


def environment(mode):
    e = ToyEnv()
    e.hp.update(settings(mode), teacher_rows=32, proxy_steps=2)
    return e


def test_wire_rejects_corruption_wrong_round_or_kind():
    x = np.arange(64).reshape(4, 16) / 64
    p = encode(x, 1, 3)
    assert len(p) == 32 + x.size * 2
    np.testing.assert_array_equal(decode(p, 1, 3), x.astype(np.float16))
    for bad, kind, r in ((p[:-1] + bytes([p[-1] ^ 1]), 1, 3), (p, 2, 3), (p, 1, 4), (p[:-1], 1, 3)):
        with pytest.raises(AssertionError): decode(bad, kind, r)


def test_margin_strength_matches_mass_per_client_and_stays_detached():
    q = torch.eye(16) * .7 + .3 / 16
    q[1] = torch.ones(16) / 16
    q[2] = q[0]
    count = torch.arange(80).view(5, 16).float()
    a, b = reception(q, count, "margin"), reception(q, count, "strength")
    torch.testing.assert_close((a * count).sum(1), (b * count).sum(1))
    assert a[:, 1:3].eq(0).all()
    assert not a.requires_grad and not b.requires_grad


@pytest.mark.parametrize("mode", ["augment", "class", "gate"])
def test_streaming_controls_equal_original_weights_bn_q_and_rng(mode):
    e = environment(mode)
    m = Method(e)
    rng = e.gen.get_state()
    ref_env = environment(mode)
    ref = Base(ref_env)
    e.gen.set_state(rng); ref_env.gen.set_state(rng)
    for r in (1, 2):
        torch.manual_seed(100 + r); a = m.round(r, .001)
        torch.manual_seed(100 + r); b = ref.round(r, .001)
        for name in ("priv_p", "priv_b"):
            assert torch.equal(getattr(m, name), getattr(ref, name))
        if m.Q is not None: assert torch.equal(m.Q, ref.Q)
        assert a["bytes_up"] == b["bytes_up"] and a["bytes_down"] == b["bytes_down"]


@pytest.mark.parametrize("mode", CELLS)
def test_all_cells_run_decode_and_restore(mode):
    env = environment(mode)
    m = Method(env)
    previous = None
    for r in (1, 2):
        result = m.round(r, .001)
        assert result["teacher_consumed_sha256"] == previous
        previous = result["teacher_produced_sha256"]
        assert previous == (None if m.no_kd else teacher_signature(mode, m.Q, m.bank_X, m.bank_target, m.teacher_p, m.teacher_b))
        assert np.isfinite(result["train_loss"]) and result["skipped_frac"] == 0
        assert torch.isfinite(m.priv_p).all() and torch.isfinite(m.priv_b).all()
        assert m.evaluate()["private"].sum() == len(env.test.Y)
    s = copy.deepcopy(m.state())
    other = Method(environment(mode)); other.load_state(s)
    for name, value in s.items():
        if torch.is_tensor(value): assert torch.equal(value, other.state()[name]), name
    if mode.startswith("proxy"):
        assert torch.equal(m.teacher_p, m.server_p.half().float())
        assert torch.equal(m.teacher_b, m.server_b.half().float())
        x = m.bank_X[None]
        torch.testing.assert_close(m.proxy_probability(x), other.proxy_probability(x), rtol=0, atol=0)
    if m.bank_X is not None:
        gen = m.gen_g.get_state().clone()
        x1, y1 = m.make_bank(3)
        x2, y2 = other.make_bank(3)
        assert torch.equal(x1, x2) and torch.equal(y1, y2)
        assert torch.equal(gen, m.gen_g.get_state())


def test_projection_enforces_v4_missingness_without_changing_real_rows():
    e = environment("project_class")
    m = Method(e)
    x = e.train.X[:8][None].clone()
    ctx = SimpleNamespace(Gfix=1, x=x, clients=torch.tensor([0]))
    syn, _ = m._synth(ctx)
    for feature, flag in m.projection_pairs:
        assert torch.isin(syn[..., flag], torch.tensor([0., 1.])).all()
        assert syn[..., feature][syn[..., flag] == 1].eq(0).all()
    assert torch.equal(x, e.train.X[:8][None])
