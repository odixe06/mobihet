"""Additive input covariance sketches, real codecs and matched Gaussian controls."""
import hashlib
import json
import struct
import zlib
import numpy as np
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.v2_math import Layout
from mh.v2_methods import Method as Prior
from mh.v7_basis import OMEGA, OMEGA_BYTES, OMEGA_SHA

D, C, EPS = 66, 16, 1e-4
HEADER = struct.Struct('<4sBBBBIIIIII')
FAMILIES = {'n': ['n_diag', 'n_full', 'n_lr4', 'n_lr8'] +
                 [f'n_nys{rank}_{mode}' for rank in (4, 8) for mode in ('fl', 'own', 'shuffle')]}
ARMS = FAMILIES['n']


def encode(kind, values, context, counts=None, contributors=None):
    dtype = {1: '<f4', 2: '<f2', 3: '<f8'}[kind]
    a = np.asarray(values, dtype=dtype)
    assert np.isfinite(a).all(), 'V7 nonfinite after wire quantization'
    assert len(context) == 16 and (counts is None) == (kind != 1)
    prefix = context
    if counts is not None:
        n = np.asarray(counts, dtype='<u8')
        bits = np.asarray(contributors, dtype='<f4')
        assert n.shape == bits.shape == (C,) and np.isin(bits, [0, 1]).all()
        prefix += n.tobytes() + bits.tobytes()
    body = prefix + a.tobytes()
    return HEADER.pack(b'MHN7', 1, kind, a.dtype.itemsize, int(counts is not None),
                       0, 0, a.size, 0, zlib.crc32(body), len(body)) + body


def decode(packet, kind, size, context):
    assert len(packet) >= HEADER.size
    magic, schema, actual, dtype, flags, base, version, n, reserved, crc, length = HEADER.unpack_from(packet)
    assert magic == b'MHN7' and schema == 1 and actual == kind and n == size
    assert dtype == {1: 4, 2: 2, 3: 8}[kind] and flags == int(kind == 1)
    assert base == version == reserved == 0
    body = packet[HEADER.size:]
    assert len(body) == length and zlib.crc32(body) == crc and body[:16] == context
    offset, counts, contributors = 16, None, None
    if flags:
        counts = np.frombuffer(body, '<u8', C, offset).copy()
        offset += 8 * C
        contributors = np.frombuffer(body, '<f4', C, offset).astype(np.float64)
        offset += 4 * C
        assert np.isin(contributors, [0, 1]).all()
    assert len(body) == offset + size * dtype
    values = np.frombuffer(body, {1: '<f4', 2: '<f2', 3: '<f8'}[kind], size, offset).astype(np.float64)
    assert np.isfinite(values).all()
    return counts, contributors, values


def symmetric(values, dimension):
    a = np.zeros((dimension, dimension), dtype=np.float64)
    i, j = np.triu_indices(dimension)
    a[i, j] = values
    a[j, i] = values
    return a


class Method(Prior):
    def __init__(self, run, initialize=True):
        self.R, self.arm = run, run.a.arm
        self.role, self.proxy, self.version, self.diag = 'private', [], 0, {}
        self.layout = Layout(run.init)
        self.global_vector = self.layout.flatten(run.init).clone()
        self.form = self.arm.split('_')[1]
        self.rank = int(self.form[-1]) if self.form.startswith(('nys', 'lr')) else 0
        self.mode = self.arm.split('_')[-1] if self.form.startswith('nys') else 'fl'
        fixture = hashlib.sha256((P.ROOT / 'papers/pilot/fixture/manifest.json').read_bytes()).hexdigest()
        self.context = hashlib.sha256(json.dumps(dict(arm=self.arm, seed=run.a.seed,
            fixture=fixture, omega=OMEGA_SHA, epsilon=EPS, schema=1), sort_keys=True).encode()).digest()[:16]
        self.packets, self.tables = {}, []
        if initialize:
            self.make_tables()

    def upload(self, x, y):
        n = torch.bincount(y, minlength=C).cpu().numpy().astype('uint64')
        rows = []
        omega = torch.from_numpy(OMEGA[:, :self.rank].copy()).to(x.device)
        for c in range(C):
            a = x[y == c].double()
            blocks = [a.sum(0)]
            if self.form == 'diag':
                blocks.append(a.square().sum(0))
            elif self.form.startswith('nys'):
                z = a @ omega
                blocks += [a.square().sum(0), (a.T @ z).flatten(),
                           (z.T @ z)[torch.triu_indices(self.rank, self.rank, device=x.device).unbind()]]
            else:
                q = a.T @ a
                blocks.append(q[torch.triu_indices(D, D, device=x.device).unbind()])
            rows.append(torch.cat(blocks).cpu().numpy())
        a = np.stack(rows)
        return encode(1, a, self.context, n, (n > 0).astype('float32')), a.shape[1]

    def reconstruct(self, n, data):
        factors, diagnostics = [], []
        for c in range(C):
            count = max(int(n[c]), 1)
            mu = data[c, :D] / count
            if self.form == 'diag':
                var = data[c, D:] / count - mu * mu
                if n[c] == 0:
                    var = np.ones(D)
                factors.append(np.concatenate([mu, np.sqrt(np.maximum(var, EPS))]))
                continue
            if self.form.startswith('nys'):
                var = data[c, D:2*D] / count - mu * mu
                omega = OMEGA[:, :self.rank]
                a = mu @ omega
                y = data[c, 2*D:2*D+D*self.rank].reshape(D, self.rank) / count - np.outer(mu, a)
                b = symmetric(data[c, 2*D+D*self.rank:], self.rank) / count - np.outer(a, a)
                eigen, vectors = np.linalg.eigh((b + b.T) / 2)
                u = (y @ vectors) / np.sqrt(np.maximum(eigen, EPS))[None]
                diagnostics.append(dict(minimum_w_eigenvalue=float(eigen.min()),
                    floored_eigenvalues=int((eigen < EPS).sum()),
                    w_vs_omega_y=float(np.linalg.norm(b - omega.T @ y))))
            else:
                cov = symmetric(data[c, D:], D) / count - np.outer(mu, mu)
                cov = (cov + cov.T) / 2
                if n[c] == 0:
                    cov = np.eye(D)
                if self.form == 'full':
                    l = np.linalg.cholesky(cov + EPS * np.eye(D))
                    factors.append(np.concatenate([mu, l[np.tril_indices(D)]]))
                    continue
                eigen, vectors = np.linalg.eigh(cov)
                u = vectors[:, -self.rank:] * np.sqrt(np.maximum(eigen[-self.rank:], 0))[None]
                var = np.diag(cov)
            if n[c] == 0:
                u, var = np.zeros((D, self.rank)), np.ones(D)
            sd = np.sqrt(np.maximum(var - (u*u).sum(1), EPS))
            factors.append(np.concatenate([mu, u.flatten(), sd]))
        return np.stack(factors), diagnostics

    def table(self, values):
        mu = values[:, :D]
        if self.form == 'full':
            l = np.zeros((C, D, D))
            i, j = np.tril_indices(D)
            l[:, i, j] = values[:, D:]
            assert (np.diagonal(l, axis1=1, axis2=2) > 0).all()
            fields = dict(mu=mu, factor=l)
        elif self.form == 'diag':
            fields = dict(mu=mu, sd=values[:, D:])
        else:
            fields = dict(mu=mu, factor=values[:, D:D+D*self.rank].reshape(C, D, self.rank),
                          sd=values[:, D+D*self.rank:])
        return {name: torch.from_numpy(a.astype('float32')).to(self.R.dev) for name, a in fields.items()}

    def make_tables(self):
        R = self.R
        if self.form.startswith('nys'):
            basis = encode(3, OMEGA, self.context)
            _, _, received = decode(basis, 3, OMEGA.size, self.context)
            assert received.tobytes() == OMEGA_BYTES
            self.packets['basis'] = basis
            for k in range(R.K):
                self.record(basis, 0, k, 'down', 'public_omega')
        packets, width = [], None
        for k, (x, y) in enumerate(zip(R.X, R.Y)):
            packet, size = self.upload(x, y)
            assert width is None or width == size
            width = size
            packets.append(packet)
            if self.mode != 'own':
                self.record(packet, 0, k, 'up', 'moments')
        assert R.K >= 5
        counts, contributors = np.zeros(C, dtype=np.uint64), np.zeros(C)
        total = np.zeros((C, width))
        decoded = []
        for packet in packets:
            n, bits, values = decode(packet, 1, C*width, self.context)
            a = values.reshape(C, width)
            counts += n
            contributors += bits
            total += a
            if self.mode == 'own':
                decoded.append((n, a))
        assert (counts > 0).all() and (contributors >= 5).all(), 'V7 population class cohort gate'
        # Trusted sum emulator gives RSU only cohort totals, not client records/statistics.
        groups = decoded if self.mode == 'own' else [(counts, total)]
        for n, a in groups:
            values, diag = self.reconstruct(n, a)
            packet = encode(2, values, self.context)
            _, _, wire = decode(packet, 2, values.size, self.context)
            table = self.table(wire.reshape(values.shape))
            if self.mode == 'shuffle':
                signs = np.random.default_rng(7603).integers(0, 2, D)*2-1
                table['factor'] *= torch.from_numpy(signs.astype('float32')).to(R.dev)[None, :, None]
            self.tables.append(table)
            if self.mode != 'own':
                self.latest_packet = packet
                self.packets['factors'] = packet
                for k in range(R.K):
                    self.record(packet, 0, k, 'down', 'gaussian')
            self.diag['reconstruction'] = diag
        self.diag.update(class_counts=counts.tolist(), contributors=contributors.tolist(),
                         rank=self.rank, form=self.form)

    def draw(self, labels, g, k):
        t = self.tables[k] if self.mode == 'own' else self.tables[0]
        z = torch.randn(len(labels), D, device=self.R.dev, generator=g)
        x = t['mu'][labels]
        if self.form == 'full':
            x = x + (t['factor'][labels] @ z[:, :, None]).squeeze(-1)
        elif self.form == 'diag':
            x = x + t['sd'][labels] * z
        else:
            u = torch.randn(len(labels), self.rank, device=self.R.dev, generator=g)
            x = x + (t['factor'][labels] @ u[:, :, None]).squeeze(-1) + t['sd'][labels]*z
        assert torch.isfinite(x).all() and torch.isfinite(x.half()).all(), 'V7 synthetic fp16 input overflow'
        return x

    def private_loss(self, k, x, y, g, r):
        real = F.cross_entropy(self.R.priv[k](x)[0].float(), y)
        labels = torch.arange(self.R.a.aux_batch, device=self.R.dev) % C
        with torch.autocast('cuda', enabled=False):
            synthetic = self.draw(labels, g, k)
        extra = F.cross_entropy(self.R.priv[k](synthetic)[0].float(), labels)
        return real + self.R.a.aux_weight*extra, torch.stack([real.detach(), extra.detach()])

    def start_round(self):
        pass

    def aggregate(self, r):
        pass

    def state(self):
        return dict(global_vector=self.global_vector, version=self.version,
                    tables=self.tables, packets=self.packets, diag=self.diag)
