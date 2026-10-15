"""Quantized classwise CVaR controls; no record carrier or learned auxiliary model."""
import struct
import zlib
import numpy as np

C, B = 16, 33
ALPHA, LAMBDA = .25, .25
EDGES = np.asarray([0.] + [2.**(-12+j*15/31) for j in range(32)], dtype='<f4')
HEADER = struct.Struct('<4sBBHII16s')


def packet(kind, version, body, identity):
    return HEADER.pack(b'CT01', 1, kind, 0, version, zlib.crc32(body), identity) + body


def decode(data, kind, version, identity):
    magic, schema, actual, flags, observed, crc, fingerprint = HEADER.unpack_from(data)
    body = data[HEADER.size:]
    assert (magic, schema, actual, flags, observed, fingerprint) == (b'CT01', 1, kind, 0, version, identity)
    assert zlib.crc32(body) == crc
    return body


def histogram(loss, y):
    assert np.isfinite(loss).all() and (loss >= 0).all()
    bins = np.searchsorted(EDGES[1:], loss, side='right')
    return np.bincount(y.astype('int64')*B+bins, minlength=C*B).reshape(C, B).astype('uint64')


def threshold(h, contributors, minimum=128):
    n = h.sum(1)
    target = np.ceil((1-ALPHA)*n).astype('uint64')
    bins = (h.cumsum(1) < target[:, None]).sum(1)
    valid = (n >= minimum) & (contributors >= 5) & (bins < B-1)
    zeta = np.zeros(C, dtype='<f4')
    zeta[valid] = EDGES[bins[valid]+1]
    return zeta, valid


def weights(loss, y, zeta, valid):
    tail = (loss > zeta[y]) & valid[y]
    return np.where(valid[y], 1-LAMBDA+LAMBDA/ALPHA*tail, 1.).astype('float32'), tail


def snapshot(local, version, identity):
    decoded, uplinks = [], []
    for h in local:
        assert h.max() <= np.iinfo('uint32').max
        body = h.astype('<u4').tobytes() + h.sum(1).astype('<u8').tobytes()
        wire = packet(1, version, body, identity)
        body = decode(wire, 1, version, identity)
        restored = np.frombuffer(body, '<u4', count=C*B).reshape(C, B).astype('uint64')
        assert np.array_equal(restored.sum(1), np.frombuffer(body, '<u8', offset=C*B*4))
        decoded.append(restored)
        uplinks.append(wire)
    total = np.stack(decoded).sum(0)
    contributors = (np.stack(decoded).sum(2) > 0).sum(0)
    zeta, valid = threshold(total, contributors)
    n = total.sum(1)
    prior = ((n+1)/(n.sum()+C)).astype('<f4')
    down = packet(2, version, zeta.tobytes()+prior.tobytes()+np.packbits(valid, bitorder='little').tobytes(), identity)
    body = decode(down, 2, version, identity)
    zeta = np.frombuffer(body, '<f4', count=C).copy()
    prior = np.frombuffer(body, '<f4', count=C, offset=4*C).copy()
    valid = np.unpackbits(np.frombuffer(body, 'u1', offset=8*C), bitorder='little').astype(bool)
    assert len(down) == 162 and all(len(p) == 2272 for p in uplinks)
    return zeta, valid, prior, uplinks, down


def probe(losses, labels, local, zeta, valid):
    counts = np.zeros(C, dtype='int64')
    tails = np.zeros(C, dtype='int64')
    compared = np.zeros(C, dtype='int64')
    different = np.zeros(C, dtype='int64')
    for loss, y, hist in zip(losses, labels, local):
        own, own_valid = threshold(hist, np.full(C, 5), minimum=32)
        pool_tail = weights(loss, y, zeta, valid)[1]
        own_tail = weights(loss, y, own, own_valid)[1]
        support = valid[y] & own_valid[y]
        counts += np.bincount(y, minlength=C)
        tails += np.bincount(y[pool_tail], minlength=C)
        compared += np.bincount(y[support], minlength=C)
        different += np.bincount(y[support & (pool_tail != own_tail)], minlength=C)
    fraction = tails/counts.clip(1)
    membership = different/compared.clip(1)
    coverage = valid & (fraction >= .10) & (fraction <= .40)
    macro_difference = float(membership[compared > 0].mean())
    passed = bool(coverage.sum() >= 12 and macro_difference >= .05)
    return dict(pass_probe=passed, valid=valid.tolist(), tail_fraction=fraction.tolist(),
                membership_difference=membership.tolist(), macro_membership_difference=macro_difference,
                classes_in_tail_range=int(coverage.sum()), counts=counts.tolist())
