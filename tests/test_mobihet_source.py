"""B18 source isolation, default identity, codec and canonical partition checks."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import data as D
from methods import mobihet_refine, mobihet_source
from scripts.audit_b18 import decode
from scripts.b18_partition import roles_for, subset
from test_mobihet_refine import environment


def equal_state(a, b):
    for k in a:
        if torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k


def test_default_two_rounds_exact(tmp_path, monkeypatch):
    monkeypatch.setenv("MOBIHET_RECOVERY_DIR", str(tmp_path / "recovery"))
    torch.set_num_threads(1)
    old = mobihet_refine.Method(environment())
    new = mobihet_source.Method(environment())
    before = torch.get_rng_state()
    expected = []
    for r in (1, 2):
        info = old._round_private(r, .001)
        expected.append((copy.deepcopy(old.state()), info, torch.get_rng_state(), old.gen_g.get_state()))
    torch.set_rng_state(before)
    for r, (state, info, rng, synth_rng) in zip((1, 2), expected):
        got = new._round_private(r, .001)
        assert got == info
        equal_state(state, new.state())
        assert torch.equal(rng, torch.get_rng_state()) and torch.equal(synth_rng, new.gen_g.get_state())


@pytest.mark.parametrize("eligible", [False, True])
def test_actual_source_replacement_and_fallback(tmp_path, monkeypatch, eligible):
    monkeypatch.setenv("MOBIHET_RECOVERY_DIR", str(tmp_path / "E/recovery"))
    env = environment(source_mode="E", source_root=str(tmp_path))
    shadow = tmp_path / "shadow"
    (shadow / "source").mkdir(parents=True)
    (shadow / "config.json").write_text(json.dumps({"payload_bytes": {"bootstrap": {"up": 0, "down": 0}}}))
    (shadow / "rounds.jsonl").write_text("".join(json.dumps({"round": r, "bytes_up": 7., "bytes_down": 8., "source_probe_E_up": 9.}) + "\n" for r in (1, 2)))
    codes = np.broadcast_to(np.roll(np.eye(16, dtype=np.uint8) * 255, 1, axis=1), (env.K, 16, 16)).copy()
    nc = np.full((env.K, 16), 8 if eligible else 0, dtype=np.int64)
    for r in (1, 2):
        np.savez(shadow / f"source/rank0_r{r:03d}.npz", ids=env.ids, counts=nc, E=codes)
    method = mobihet_source.Method(env)
    base = mobihet_refine.Method(environment())
    rng = torch.get_rng_state()
    base._round_private(1, .001)
    torch.set_rng_state(rng)
    info = method._round_private(1, .001)
    assert torch.equal(base.priv_p, method.priv_p) and torch.equal(base.priv_b, method.priv_b)
    if eligible:
        assert torch.equal(method.Q, torch.roll(torch.eye(16), 1, dims=1))
    else:
        assert torch.equal(base.Q, method.Q)
    assert info["bytes_up"] == info["source_main_up"] + 16
    assert info["bytes_down"] == info["source_main_down"] + 8
    # Full state boundary must preserve the replaced teacher for round2.
    state, env_rng, rng = copy.deepcopy(method.state()), env.gen.get_state(), torch.get_rng_state()
    resumed = mobihet_source.Method(environment(source_mode="E", source_root=str(tmp_path)))
    resumed.load_state(state)
    resumed.env.gen.set_state(env_rng)
    torch.set_rng_state(rng)
    method._round_private(2, .001)
    expected = copy.deepcopy(method.state())
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    resumed._round_private(2, .001)
    equal_state(expected, resumed.state())
    assert torch.equal(after, torch.get_rng_state())


@pytest.mark.parametrize("counted", [False, True])
@pytest.mark.parametrize("n", [0, 1, 16])
def test_packet_roundtrip(counted, n):
    rng = np.random.default_rng(91)
    q = rng.integers(0, 256, (16, 16), dtype=np.uint8)
    present = np.arange(16) < n
    counts = rng.integers(8, 50, 16)
    blob = mobihet_source.frame(q, present, counts if counted else None)
    got, mask, c = decode(blob, counted)
    assert np.array_equal(mask, present) and np.array_equal(got[present], q[present])
    assert not got[~present].any()
    if counted:
        assert np.array_equal(c[present], counts[present]) and not c[~present].any()
    with pytest.raises(AssertionError):
        decode(b"x" + blob[1:], counted)


def test_temporal_ties_matched_counts_and_cache_order():
    y = np.tile(np.arange(16, dtype=np.int8), 100)
    t = np.repeat(np.arange(800), 2)
    r, a, h, cut = roles_for(y, t, 72)
    assert not np.any((r & 2 > 0) & (r & 4 > 0))
    assert t[r & 1 > 0].max() < t[r & 1 == 0].min()
    assert np.array_equal(np.bincount(y[r & 2 > 0], minlength=16), np.bincount(y[r & 4 > 0], minlength=16))
    assert np.array_equal(r, roles_for(y, t, 72)[0])
    split = D.Split(np.arange(len(y) * 2).reshape(len(y), 2), y, np.array([0, len(y)]), np.array([72]))
    part = dict(ids=np.array([72]), off=np.array([0, len(y)]), roles=r,
                label_sha256=np.array([hashlib.sha256(y.tobytes()).hexdigest().encode()]))
    out = subset(split, part, 1)
    assert np.array_equal(out.Y, y[t <= cut])
    split.Y = y[::-1].copy()
    with pytest.raises(AssertionError, match="row order"):
        subset(split, part, 1)
