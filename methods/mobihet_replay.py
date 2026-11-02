"""Counterfactual diagnostic: keep training/aggregation, replace downlink by a fixed audited Q history."""
import base64
import hashlib
import zlib

import numpy as np
import torch

from methods.mobihet import Method as LiveMethod
from methods import mobihet_replay_data as frozen


def permutation():
    gen = np.random.default_rng(20261101)
    order = np.stack([gen.permutation(np.delete(np.arange(16), c)) for c in range(16)])
    assert hashlib.sha256(order.astype("<i8").tobytes()).hexdigest() == "4173050877f74af67ea5230ab4844c18694aab431e1b438a3346e3de7c3f3236", "Permutation changed across runtimes"
    return order


def transform(q, mode):
    assert mode in ("full", "uniform", "permuted")
    if mode == "full": return q.clone()
    out = q.clone()
    for c, order in enumerate(permutation()):
        others = torch.arange(16, device=q.device) != c
        if mode == "uniform":
            out[..., c, others] = q[..., c, others].double().mean(-1, keepdim=True).float()
        else:
            out[..., c, others] = q[..., c, torch.as_tensor(order, device=q.device)]
    return out


def source_schedule(seed, device, expected_sha):
    raw = zlib.decompress(base64.b64decode(frozen.PAYLOAD))
    assert hashlib.sha256(raw).hexdigest() == expected_sha == frozen.SHA256, "Teacher input SHA mismatch"
    assert seed in (42, 43) and len(raw) == 61440
    q = np.frombuffer(raw[:40960], dtype="<f4").reshape(2, 20, 16, 16)[seed - 42].copy()
    packet = np.frombuffer(raw[40960:], dtype="<f2").reshape(2, 20, 16, 16)[seed - 42].copy()
    expected = torch.from_numpy(q).to(device)
    decoded = torch.from_numpy(packet).to(device).float()
    decoded = decoded / decoded.sum(-1, keepdim=True)
    assert torch.isfinite(expected).all() and (expected >= 0).all()
    torch.testing.assert_close(expected.sum(-1), torch.ones(20, 16, device=device), rtol=0, atol=2e-7)
    # CPU and CUDA reduction order can differ by one fp32 ULP. The authoritative target is unchanged.
    torch.testing.assert_close(decoded, expected, rtol=0, atol=2e-7)
    if expected.is_cuda:
        assert torch.equal(decoded, expected), "GPU fp16 downlink did not decode to frozen teacher"
    return expected


class Method(LiveMethod):
    name = "mobihet_replay"

    def __init__(self, env):
        hp = env.hp
        assert hp["arm"] == "none" and hp["synth"] == "copula" and hp["synth_target"] == "soft"
        assert hp["q_codec"] == "mean_fp16" and hp["q_collect"] == "online" and hp["q_agg"] == "count"
        assert hp["kd_weight"] == "fixed" and hp["kd_support"] == "off"
        assert not hp.get("no_kd", False) and hp.get("anchor", "off") == "off" and hp.get("kd_bins", 1) == 1
        self.replay_mode = hp["replay_mode"]
        assert self.replay_mode == "full" or not hp["real_kd"], "No gate after target transformation"
        self.replay_raw = source_schedule(hp["seed"], env.dev, hp["replay_source_sha256"])
        self.replay_target = transform(self.replay_raw, self.replay_mode)
        super().__init__(env)
        self.Q_live = self.Q_raw = None

    def round(self, r, lr):
        assert r == self.version + 1 and 1 <= r <= 20
        previous = self.Q
        if r > 1:
            assert torch.equal(previous, self.replay_target[r - 2]), "Wrong consumed teacher lag"
        info = super().round(r, lr)  # Collect the same train-only uplinks; live output is diagnostic only.
        self.Q_live = self.Q
        self.Q_raw = self.replay_raw[r - 1].clone()
        self.Q = self.replay_target[r - 1].clone()  # Input for next round, including driver shadow probes.
        info.update(q_true_probability_mean=float(self.Q.diag().mean()),
                    q_rejected_classes=(self.Q.argmax(-1) != torch.arange(16, device=self.env.dev)).nonzero().flatten().tolist(),
                    q_entropy_mean=float(-(self.Q * self.Q.clamp_min(1e-6).log()).sum(-1).mean()),
                    q_change_l1=None if previous is None else float((self.Q - previous).abs().sum(-1).mean()),
                    q_live_true_probability_mean=float(self.Q_live.diag().mean()),
                    replay_consumed_round=None if r == 1 else r - 1, replay_next_round=r,
                    replay_source_sha256=frozen.SHA256)
        return info

    def state(self):
        return super().state() | {"Q_live": None if self.Q_live is None else self.Q_live.cpu(),
                                 "Q_raw": None if self.Q_raw is None else self.Q_raw.cpu(), "replay_source_sha256": frozen.SHA256}

    def load_state(self, s):
        assert s["replay_source_sha256"] == frozen.SHA256
        super().load_state(s)
        for key in ("Q_live", "Q_raw"):
            setattr(self, key, None if s[key] is None else s[key].to(self.env.dev))
        if self.version:
            assert torch.equal(self.Q, self.replay_target[self.version - 1])
