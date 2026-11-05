"""MobiHET Core: lossless zlib frame for the existing uint16 CDF downlink."""
import hashlib
import json
import os
from pathlib import Path
import struct
import zlib

import numpy as np
import torch
from methods.mobihet_wstudy import WLive

HEADER = struct.Struct("<4s7I")


def encode(codes):
    codes = np.asarray(codes, dtype="<u2")
    c, d, nb = codes.shape
    raw = codes.tobytes()
    body = zlib.compress(raw, 6)
    return HEADER.pack(b"MCDF", 1, c, d, nb, len(raw), len(body), zlib.crc32(body)) + body


def decode(packet):
    magic, version, c, d, nb, raw_len, body_len, crc = HEADER.unpack(packet[:HEADER.size])
    assert (magic, version, c, d, nb) == (b"MCDF", 1, 16, 77, 256)
    body = packet[HEADER.size:]
    assert len(body) == body_len and zlib.crc32(body) == crc
    raw = zlib.decompress(body)
    assert len(raw) == raw_len == c * d * nb * 2
    return np.frombuffer(raw, dtype="<u2").reshape(c, d, nb).copy()


def tensor_sha(t):
    return hashlib.sha256(t.cpu().contiguous().numpy().tobytes()).hexdigest()


class Method(WLive):
    def _copula_bootstrap(self, client, NB=256):
        assert NB == 256 and self.w_boot == "compact"
        super()._copula_bootstrap(client, NB)
        original_cdf, original_mass = self.cop_cdf, self.cop_mass
        codes = torch.round(original_cdf.double() * 65535).cpu().numpy().astype("<u2")
        packet = encode(codes)
        received = decode(packet)
        assert np.array_equal(codes, received)
        cdf = torch.from_numpy(received.astype(np.int32)).to(self.env.dev).double() / 65535
        self.cop_cdf = cdf.float().contiguous()
        self.cop_mass = torch.diff(cdf, dim=-1, prepend=torch.zeros_like(cdf[..., :1])).float()
        assert torch.equal(self.cop_cdf, original_cdf), "Lossless CDF changed decoded values"
        assert torch.equal(self.cop_mass, original_mass), "Lossless CDF changed bin masses"
        self.bootstrap_bytes["down"] += len(packet) - codes.nbytes
        path = Path(os.environ["MOBIHET_RECOVERY_DIR"]).parent
        path.mkdir(parents=True, exist_ok=True)
        rank = self.env.rank
        (path / f"cdf_rank{rank}.bin").write_bytes(packet)
        (path / f"cdf_codec_rank{rank}.json").write_text(json.dumps(dict(
            codec="zlib6_uint16_v1", packet_bytes=len(packet), raw_bytes=codes.nbytes,
            packet_sha256=hashlib.sha256(packet).hexdigest(), cdf_sha256=tensor_sha(self.cop_cdf),
            mass_sha256=tensor_sha(self.cop_mass), bitexact_cdf=True, bitexact_mass=True,
            bootstrap_bytes=self.bootstrap_bytes), indent=2) + "\n")
