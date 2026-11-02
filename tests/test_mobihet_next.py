"""Q20 controls, finite decoded aggregates, state and actual mixed-support behavior."""
import copy
import os

import numpy as np
import pytest
import torch

from core import grouped
from methods.mobihet_study import Method as Parent, decode_probability
from methods.mobihet_next import Method, signature
from scripts.q20_common import CELLS, settings
from test_methods import ToyEnv


def environment(mode):
    env = ToyEnv()
    env.hp.update(settings(mode))
    return env


@pytest.fixture(autouse=True)
def guards(tmp_path, monkeypatch):
    # CPU arithmetic tests do not certify compilation; real CUDA preflight does.
    monkeypatch.setenv("MOBIHET_RECOVERY_DIR", str(tmp_path))
    monkeypatch.setattr(grouped, "TRAIN", lambda *a, **k: grouped.forward(*a, **k))


@pytest.mark.parametrize("mode", CELLS)
def test_rounds_aggregate_targets_and_state(mode):
    env = environment(mode); model = Method(env); previous = None
    for r in (1, 2):
        result = model.round(r, .001)
        assert result["teacher_consumed_sha256"] == previous
        previous = signature(model.Q, model.bank_X, model.bank_target, mode)
        assert result["teacher_produced_sha256"] == previous
        assert result["bytes_up"] == result["bytes_down"] == (544 if mode == "stream" else 33344)
        assert np.isfinite(result["train_loss"]) and result["skipped_frac"] == 0
        if mode not in {"stream", "query"}:
            expected = decode_probability((model.class_sums / model.class_totals[:, None]).float().half())
            assert torch.equal(model.Q, expected)
        if mode != "stream":
            assert torch.equal(model.bank_body, (model.aggregate_sums / model.aggregate_weights[:, None]).float().half())
            assert torch.equal(model.bank_target, decode_probability(model.bank_body))
    state = copy.deepcopy(model.state()); restored = Method(environment(mode)); restored.load_state(state)
    for key, value in state.items():
        if torch.is_tensor(value): assert torch.equal(value, restored.state()[key]), key
    assert np.array_equal(model.evaluate()["private"], restored.evaluate()["private"])


def test_stream_and_query_preserve_parent_training_and_rng():
    reference = Parent(environment("stream")); stream = Method(environment("stream")); query = Method(environment("query"))
    for r in (1, 2, 3):
        for model in (reference, stream, query):
            torch.manual_seed(77 + r); model.round(r, .001)
        for key in ("priv_p", "priv_b", "Q"):
            assert torch.equal(getattr(reference, key), getattr(stream, key)), key
            assert torch.equal(getattr(reference, key), getattr(query, key)), key
        assert torch.equal(reference.gen_g.get_state(), query.gen_g.get_state())
        assert torch.equal(reference.env.gen.get_state(), query.env.gen.get_state())


def test_mixed_first_round_matches_before_teacher_and_inputs_match():
    first = None
    for mode in CELLS[2:]:
        model = Method(environment(mode)); torch.manual_seed(123)
        model.round(1, .001)
        state = tuple(getattr(model, k).clone() for k in ("priv_p", "priv_b", "Q"))
        if first is None: first = state
        else:
            for x, y in zip(first, state): assert torch.equal(x, y), mode
    a, b = Method(environment("mixed_class")), Method(environment("mixed_input"))
    from types import SimpleNamespace
    ctx = SimpleNamespace(Gfix=2, x=a.env.train.X[:1][None].expand(2, -1, -1), clients=torch.tensor([0, 1]))
    ax, ay = a._synth(ctx); bx, by = b._synth(ctx)
    assert torch.equal(ax, bx) and torch.equal(ay, by) and ax.shape == (2, 64, 77)
    assert torch.equal(a.bank_index, b.bank_index)
    assert torch.equal(ay[:, :32], ay[:, 32:])
    assert all(torch.equal(torch.bincount(labels, minlength=16), torch.full((16,), 2)) for labels in ay[:, :32])
