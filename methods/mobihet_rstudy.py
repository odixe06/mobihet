"""R20: identical Q controls or a common, train-only class/input teacher history."""
import base64
import hashlib
import zlib

import numpy as np
import torch

from methods.mobihet_next import Method as Live, signature
from methods.mobihet_study import decode_probability, require_shared


def bank_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def load_schedule(seed, device, expected_sha):
    from methods import mobihet_rstudy_data as frozen
    raw = zlib.decompress(base64.b64decode(frozen.PAYLOAD))
    assert hashlib.sha256(raw).hexdigest() == expected_sha == frozen.SHA256
    assert seed in (42, 43)
    shapes = {"Q": (2, 20, 16, 16), "class_body": (2, 20, 16, 16),
              "target": (2, 20, 1024, 16), "body": (2, 20, 1024, 16)}
    result, offset = {}, 0
    for name, shape in shapes.items():
        dtype = "<f2" if name.endswith("body") else "<f4"
        n = int(np.prod(shape)) * np.dtype(dtype).itemsize
        values = np.frombuffer(raw[offset:offset + n], dtype=dtype).reshape(shape)[seed - 42].copy()
        result[name] = torch.from_numpy(values).to(device)
        offset += n
    assert offset == len(raw)
    for name, body in (("Q", "class_body"), ("target", "body")):
        assert torch.isfinite(result[name]).all() and (result[name] >= 0).all()
        decoded = decode_probability(result[body])
        if result[name].is_cuda:
            assert torch.equal(decoded, result[name]), "Frozen downlink GPU decode mismatch; no target repair"
        else:
            # Loader inspection only. Runtime CUDA must decode exactly; auditor uses per-entry intervals.
            torch.testing.assert_close(decoded, result[name], atol=3e-7, rtol=0)
    result["banks"] = frozen.SOURCES[seed - 42]["banks"]
    return result


class Replay(Live):
    def __init__(self, env):
        assert env.hp["next_mode"] in {"mixed_input", "real", "margin", "strength"}
        self.source_sha = env.hp["r_source_sha256"]
        self.history = load_schedule(env.hp["seed"], env.dev, self.source_sha)
        super().__init__(env)
        self.Q_live = self.bank_target_live = self.bank_body_live = None

    def round(self, r, lr):
        assert r == self.version + 1 and 1 <= r <= len(self.history["Q"])
        if r > 1:
            assert torch.equal(self.Q, self.history["Q"][r - 2])
            assert torch.equal(self.bank_target, self.history["target"][r - 2])
        info = super().round(r, lr)
        assert bank_sha(self.bank_X) == self.history["banks"][r - 1]
        self.Q_live, self.bank_target_live, self.bank_body_live = self.Q, self.bank_target, self.bank_body
        info["live_teacher_produced_sha256"] = info["teacher_produced_sha256"]
        info["live_teacher_received_agreement"] = info["teacher_received_agreement"]
        self.Q = self.history["Q"][r - 1].clone()
        self.bank_target = self.history["target"][r - 1].clone()
        self.bank_body = self.history["body"][r - 1].clone()
        current = signature(self.Q, self.bank_X, self.bank_target, self.variant)
        require_shared(self.env, current)
        info.update(teacher_produced_sha256=current, replay_source_sha256=self.source_sha,
                    teacher_received_agreement=float((self.bank_target.argmax(1) == self.bank_y).float().mean()))
        return info

    def state(self):
        return super().state() | {key: None if getattr(self, key) is None else getattr(self, key).detach().cpu()
            for key in ("Q_live", "bank_target_live", "bank_body_live")} | {"r_source_sha256": self.source_sha}

    def load_state(self, state):
        assert state["r_source_sha256"] == self.source_sha
        super().load_state(state)
        for key in ("Q_live", "bank_target_live", "bank_body_live"):
            setattr(self, key, None if state[key] is None else state[key].to(self.env.dev))
        if self.version:
            assert torch.equal(self.Q, self.history["Q"][self.version - 1])
            assert torch.equal(self.bank_target, self.history["target"][self.version - 1])


def Method(env):
    assert env.hp["r_kind"] in {"repeat", "replay"}
    return Replay(env) if env.hp["r_kind"] == "replay" else Live(env)
