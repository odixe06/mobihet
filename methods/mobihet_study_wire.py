"""P20 fp16 packets:32-byte versioned header, shape, body length and CRC; real decode path."""
import struct
import zlib

import numpy as np
import torch

HEADER = struct.Struct("<4sBBHIIIIII")
assert HEADER.size == 32


def encode(array, kind, round_id):
    a = np.asarray(array, dtype="<f2")
    assert a.ndim == 2 and np.isfinite(a).all()
    body = a.tobytes()
    return HEADER.pack(b"MHP2", 1, kind, 0, round_id, *a.shape, len(body), zlib.crc32(body), 0) + body


def decode(packet, kind, round_id):
    magic, version, message, reserved, r, rows, cols, length, crc, tail = HEADER.unpack(packet[:32])
    assert (magic, version, message, reserved, r, tail) == (b"MHP2", 1, kind, 0, round_id, 0)
    body = packet[32:]
    assert len(body) == length == rows * cols * 2 and zlib.crc32(body) == crc
    a = np.frombuffer(body, dtype="<f2").reshape(rows, cols).copy()
    assert np.isfinite(a).all()
    return a


def roundtrip(tensor, kind, round_id):
    """(client,row,col) -> actual packets -> decoded tensor on original device; no raw rows."""
    a = tensor.detach().half().cpu().numpy()
    packets = [encode(row, kind, round_id) for row in a]
    decoded = np.stack([decode(p, kind, round_id) for p in packets])
    assert len({len(p) for p in packets}) == 1
    return torch.from_numpy(decoded).to(tensor.device), len(packets[0])
