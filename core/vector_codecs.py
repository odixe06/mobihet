"""V2 public arithmetic and real wire codecs; no records or model-specific losses."""
import math
import struct
import zlib
import numpy as np
import torch

HEADER = struct.Struct('<4sBBBBIIIIII')  # 32 application bytes
KINDS = {'dense': 1, 'sketch': 2, 'coeff': 3, 'patch': 4, 'counts': 5, 'moments': 6}


def pack_vector(kind, values, base=0, version=0, count=None, half=False):
    assert kind in KINDS and kind != 'patch'
    source = values.detach().float().cpu().numpy()
    assert np.isfinite(source).all() and (not half or (np.abs(source) <= 65504).all())
    a = source.astype('<f2' if half else '<f4')
    assert np.isfinite(a).all(), 'nonfinite after wire quantization'
    body = (struct.pack('<Q', int(count)) if count is not None else b'') + a.tobytes()
    return HEADER.pack(b'MHV2', 1, KINDS[kind], 2 if half else 4, int(count is not None),
                       base, version, a.size, 0, zlib.crc32(body), len(body)) + body


def pack_patch(indices, values, base, version):
    i = indices.detach().cpu().numpy().astype('<u4')
    v = values.detach().float().cpu().numpy().astype('<f2')
    assert i.shape == v.shape and np.isfinite(v).all() and len(np.unique(i)) == len(i)
    a = np.empty(len(i), dtype=[('i', '<u4'), ('v', '<f2')]); a['i'], a['v'] = i, v
    body = a.tobytes()
    return HEADER.pack(b'MHV2', 1, KINDS['patch'], 2, 0, base, version, len(i), 0,
                       zlib.crc32(body), len(body)) + body


def unpack(packet, device='cpu', expected_base=None):
    assert isinstance(packet, bytes) and len(packet) >= HEADER.size
    magic, schema, kind, dtype, flags, base, version, n, reserved, crc, size = HEADER.unpack_from(packet)
    assert magic == b'MHV2' and schema == 1 and kind in KINDS.values() and reserved == 0
    assert dtype in (2, 4) and flags in (0, 1)
    body = packet[HEADER.size:]
    assert len(body) == size and zlib.crc32(body) == crc
    assert expected_base is None or base == expected_base, 'wrong patch/update base'
    if kind == KINDS['patch']:
        assert flags == 0 and dtype == 2 and size == n * 6
        a = np.frombuffer(body, dtype=[('i', '<u4'), ('v', '<f2')])
        idx = torch.from_numpy(a['i'].astype('int64')).to(device)
        val = torch.from_numpy(a['v'].astype('float32')).to(device)
        assert len(torch.unique(idx)) == n
        count = None
    else:
        assert size == n * dtype + 8 * flags
        count = struct.unpack_from('<Q', body)[0] if flags else None
        a = np.frombuffer(body, dtype='<f2' if dtype == 2 else '<f4', offset=8 * flags)
        val = torch.from_numpy(a.astype('float32')).to(device); idx = None
    assert torch.isfinite(val).all()
    return dict(kind=kind, base=base, version=version, values=val, indices=idx, count=count)


class SecureSum:
    """Trusted sum emulator. Caller only receives cohort total; NOT cryptography/DP."""
    def __init__(self, kind, base, size, device, minimum=5):
        self.kind, self.base, self.size, self.minimum = kind, base, size, minimum
        self._total = torch.zeros(size, device=device, dtype=torch.float32)
        self._count, self._clients = 0, set()

    def add(self, client, packet):
        assert client not in self._clients, 'duplicate contribution'
        p = unpack(packet, self._total.device, self.base)
        assert p['kind'] == KINDS[self.kind] and p['values'].numel() == self.size and p['count'] is not None
        self._total.add_(p['values']); self._count += p['count']; self._clients.add(client)

    def finish(self):
        assert len(self._clients) >= self.minimum and self._count > 0, 'insufficient cohort'
        assert torch.isfinite(self._total).all()
        return self._total.clone(), self._count


class Layout:
    def __init__(self, model):
        self.items = [(name, p.shape, p.numel()) for name, p in model.named_parameters()]
        self.size = sum(n for _, _, n in self.items)

    def flatten(self, model):
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def mapping(self, vector):
        assert vector.numel() == self.size
        out, at = {}, 0
        for name, shape, n in self.items:
            out[name] = vector[at:at+n].view(shape); at += n
        return out

    @torch.no_grad()
    def load(self, model, vector):
        params = dict(model.named_parameters())
        for name, value in self.mapping(vector).items():
            params[name].copy_(value)


class CountSketch:
    def __init__(self, size, depth, width, seed, device):
        assert depth % 2 == 1 and width > 0
        self.size, self.depth, self.width = size, depth, width
        rng = np.random.default_rng(seed)
        self.buckets = torch.from_numpy(rng.integers(width, size=(depth, size))).to(device)
        self.signs = torch.from_numpy(rng.integers(2, size=(depth, size)).astype('float32') * 2 - 1).to(device)

    def encode(self, x):
        assert x.numel() == self.size
        out = torch.zeros(self.depth, self.width, device=x.device, dtype=torch.float32)
        for j in range(self.depth):
            out[j].scatter_add_(0, self.buckets[j], x.float() * self.signs[j])
        return out.reshape(-1)

    def estimate(self, sketch):
        s = sketch.view(self.depth, self.width)
        values = torch.stack([s[j, self.buckets[j]] * self.signs[j] for j in range(self.depth)])
        # CUDA median returns nondeterministic tie indices even when only values are used.
        return values.sort(dim=0, stable=True).values[self.depth // 2]

    def patch(self, accumulated, k, base):
        estimate = self.estimate(accumulated)
        idx = torch.argsort(estimate.abs(), descending=True, stable=True)[:min(k, self.size)]
        idx = idx.sort().values
        packet = pack_patch(idx, estimate[idx], base, base + 1)
        decoded = unpack(packet, estimate.device, base)
        delta = torch.zeros_like(estimate); delta[decoded['indices']] = decoded['values']
        return packet, delta


class SeedBasis:
    """Fixed public Rademacher basis; FO coefficients, never a second learned module."""
    def __init__(self, model, k, seed, device):
        self.layout = Layout(model); self.base = self.layout.flatten(model).float().to(device)
        self.k, self.seed = k, seed
        self.z = torch.empty(self.layout.size, k, device=device, dtype=torch.float32)
        at = 0
        for block, (_, _, n) in enumerate(self.layout.items):
            scale = max(float(self.base[at:at+n].square().mean().sqrt()), 1e-3) / math.sqrt(k)
            rng = np.random.default_rng(np.random.SeedSequence([seed, block]))
            # Construct one parameter block at a time; do not allocate a second full CPU basis.
            signs = rng.integers(2, size=(n, k), dtype=np.int8)
            self.z[at:at+n] = torch.from_numpy(signs).to(device).float().mul_(2).sub_(1).mul_(scale)
            at += n

    def vector(self, a):
        # Public basis and reconstruction are fp32 even while model forward uses AMP.
        with torch.autocast(self.base.device.type, enabled=False):
            return self.base + self.z @ a.float()

    def forward(self, model, a, x):
        return torch.func.functional_call(model, self.layout.mapping(self.vector(a)), (x,), strict=False)


def ratio_weights(logits, labels, cap=10., shuffle=None):
    raw = logits.float().gather(1, labels[:, None]).squeeze(1).clamp(-10, 10).exp().clamp_max(cap)
    weight = torch.empty_like(raw)
    for y in torch.unique(labels):
        idx = torch.where(labels == y)[0]; v = raw[idx]
        if shuffle is not None:
            v = v[torch.randperm(len(v), device=v.device, generator=shuffle)]
        weight[idx] = v / v.mean()
    return weight, raw
