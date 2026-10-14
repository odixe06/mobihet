"""Nonlearned, fixed input-cell counts; never serialize client records."""
import struct
import zlib
import numpy as np

PAIRS = [("f_snd_spd", "f_sess_implied_spd"),
         ("f_dist", "f_sess_pos_pred_err"),
         ("f_sender_rate_5s", "f_sender_share_5s"),
         ("f_snd_acl", "f_sess_acl_residual")]
EDGES = np.array([-1., 0., 1.], dtype=np.float32)
C, B = 16, 4
HEADER = struct.Struct("<4sBBHII16s")


def bins(x, names, p):
    assert p in (2, 4) and np.isfinite(x).all()
    axes = np.array([[names.index(a), names.index(b)] for a, b in PAIRS[:p]])
    # An edge belongs to the bin to its right; outside bins are unbounded.
    ids = np.searchsorted(EDGES, x[:, axes], side="right")
    return ids[:, :, 0] * B + ids[:, :, 1]


def count(x, y, names, p):
    ids = bins(x, names, p)
    out = np.zeros((p, B * B, C), dtype=np.uint32)
    for r in range(p):
        out[r] = np.bincount(ids[:, r] * C + y, minlength=B*B*C).reshape(B*B, C)
    n = np.bincount(y, minlength=C).astype(np.uint64)
    assert (out.sum(axis=1) == n).all()
    return out, n


def targets(counts, n, eligible):
    likelihood = (counts.astype(np.float64) + 1.) / (n + B*B)
    t = likelihood / likelihood.sum(axis=-1, keepdims=True)
    assert np.isfinite(t).all() and np.allclose(t.sum(axis=-1), 1.)
    return t.astype(np.float32), eligible


def lookup(x, names, table, eligible):
    ids = bins(x, names, len(table))
    mask = eligible[np.arange(len(table))[None, :], ids]
    t = table[np.arange(len(table))[None, :], ids]
    mass = (t * mask[:, :, None]).sum(axis=1)
    denominator = mask.sum(axis=1)
    ok = denominator > 0
    mass[ok] /= denominator[ok, None]
    mass[~ok] = 1. / C   # Diagnostic only: training must skip auxiliary loss here.
    return mass, ok


def packet(kind, p, payload, identity):
    assert len(identity) == 16
    header = HEADER.pack(b"MHBC", 1, kind, p, len(payload), zlib.crc32(payload), identity)
    result = header + payload
    magic, schema, actual, size, length, crc, got_id = HEADER.unpack_from(result)
    assert (magic, schema, actual, size, got_id) == (b"MHBC", 1, kind, p, identity)
    assert length == len(result)-HEADER.size and crc == zlib.crc32(result[HEADER.size:])
    return result


def packets(counts, n, table, eligible, prior, identity):
    p = len(counts)
    up = counts.astype("<u4").tobytes() + (counts > 0).astype("u1").tobytes() + n.astype("<u8").tobytes()
    bits = np.repeat(eligible[:, :, None], C, axis=-1).ravel()
    down = table.astype("<f4").tobytes() + np.packbits(bits, bitorder="little").tobytes() + prior.astype("<f4").tobytes()
    return packet(1, p, up, identity), packet(2, p, down, identity)
