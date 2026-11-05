"""Replay integration against audited N inputs and the unchanged live training path."""
import copy

import numpy as np
import pytest
import torch

from methods import mobihet, mobihet_replay as replay
from test_methods import ToyEnv


def environment(real=False, mode="full"):
    env = ToyEnv()
    env.hp.update(arm="none", synth="copula", synth_rows=16, synth_target="soft", real_kd=real,
                  real_kd_gate=real, q_codec="mean_fp16", q_collect="online", q_agg="count", kd_support="off",
                  kd_weight="fixed", replay_mode=mode, replay_source_sha256=replay.frozen.SHA256)
    return env


@pytest.mark.parametrize("seed", [42, 43])
def test_frozen_packets_and_transforms_preserve_required_information(seed):
    q = replay.source_schedule(seed, torch.device("cpu"), replay.frozen.SHA256)
    torch.testing.assert_close(q.sum(-1), torch.ones(20, 16), atol=2e-7, rtol=0)
    uniform, permuted = (replay.transform(q, mode) for mode in ("uniform", "permuted"))
    for target in (uniform, permuted):
        assert torch.equal(target.diagonal(dim1=-2, dim2=-1), q.diagonal(dim1=-2, dim2=-1))
        torch.testing.assert_close(target.double().sum(-1), q.double().sum(-1), atol=1e-7, rtol=0)
    for c in range(16):
        other = torch.arange(16) != c
        assert torch.equal(permuted[:, c, other].sort(-1).values, q[:, c, other].sort(-1).values)
        assert torch.equal(uniform[:, c, other], uniform[:, c, other][:, :1].expand(-1, 15))
    assert not torch.equal(q, permuted)
    assert np.array_equal(replay.permutation(), replay.permutation())


@pytest.mark.parametrize("real,mode", [(False, "full"), (True, "full"), (False, "uniform"), (False, "permuted")])
def test_replay_matches_manual_teacher_injection_and_state_roundtrip(real, mode, tmp_path):
    torch.manual_seed(123)
    m = replay.Method(environment(real, mode))
    control = mobihet.Method(environment(real, mode))
    for r in (1, 2, 3):
        control.Q = None if r == 1 else m.replay_target[r - 2].clone()
        rng = torch.get_rng_state()
        info = m.round(r, .001)
        torch.set_rng_state(rng)
        control.round(r, .001)
        assert torch.equal(m.priv_p, control.priv_p) and torch.equal(m.priv_b, control.priv_b)
        assert torch.equal(m.Q_live, control.Q)
        assert torch.equal(m.Q, m.replay_target[r - 1])
        assert info["replay_consumed_round"] == (None if r == 1 else r - 1)
        assert info["bytes_up"] == info["bytes_down"] == 544
    path = tmp_path / "state.pt"
    torch.save(m.state(), path)
    restored = replay.Method(environment(real, mode))
    restored.load_state(torch.load(path, weights_only=True))
    assert np.array_equal(restored.evaluate()["private"], m.evaluate()["private"])
    bad = copy.deepcopy(m.state()); bad["Q"][13, 0] += .01
    with pytest.raises(AssertionError): restored.load_state(bad)


def test_refuse_wrong_input_hash_and_transformed_real_gate():
    env = environment(); env.hp["replay_source_sha256"] = "wrong"
    with pytest.raises(AssertionError, match="SHA"): replay.Method(env)
    with pytest.raises(AssertionError, match="gate"): replay.Method(environment(True, "uniform"))
