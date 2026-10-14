"""One-shot cohort moments, marginal/joint32 masks, analytic G59/G61 decoder."""
import hashlib
import struct
import numpy as np
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.v2_math import Layout
from mh.v2_methods import Method as Base
from mh.v4_schema import FIRST, PROFILES, STRUCTURAL
from mh.v6_schema import NAMES, MEAN, SCALE, PROVENANCE

HEADER = struct.Struct('<4sBBHIII16s')
KINDS = {'counts': 1, 'eligibility': 2, 'moments': 3, 'factors': 4, 'mask': 5}
VEL = ['f_rcv_vx', 'f_rcv_vy', 'f_snd_vx', 'f_snd_vy', 'f_hed_diff_cos']
BEARING = ['f_bearing_sin', 'f_bearing_cos']
BINARY = [FIRST] + sum(PROFILES, [])


def encode(values, kind, context, half=False):
    array = values.detach().cpu().numpy().astype('<f2' if half else '<f8').reshape(-1)
    assert np.isfinite(array).all(), 'nonfinite V6 encoded values'
    body = HEADER.pack(b'MHG6', KINDS[kind], 2 if half else 8, 0,
                       len(array), 0, 0, context) + array.tobytes()
    return body + hashlib.sha256(body).digest()


def decode(packet, kind, context, device):
    body, checksum = packet[:-32], packet[-32:]
    assert hashlib.sha256(body).digest() == checksum, 'V6 payload checksum'
    magic, type_, width, flags, size, base, version, actual = HEADER.unpack(body[:HEADER.size])
    assert (magic, type_, flags, base, version, actual) == (b'MHG6', KINDS[kind], 0, 0, 0, context)
    assert width in (2, 8) and len(body) == HEADER.size + size * width
    values = np.frombuffer(body[HEADER.size:], dtype='<f2' if width == 2 else '<f8').copy()
    assert np.isfinite(values).all()
    return torch.from_numpy(values).to(device=device, dtype=torch.float64)


def restore_derived(x, dimension):
    raw = x.double() * x.new_tensor(SCALE, dtype=torch.float64) + x.new_tensor(MEAN, dtype=torch.float64)
    index = {name: j for j, name in enumerate(NAMES)}
    for side in ('rcv', 'snd'):
        speed = raw[:, index[f'f_{side}_spd']]
        for axis, trig in (('x', 'sin'), ('y', 'cos')):
            raw[:, index[f'f_{side}_v{axis}']] = speed * raw[:, index[f'f_{side}_hed_{trig}']]
    raw[:, index['f_hed_diff_cos']] = sum(
        raw[:, index[f'f_rcv_hed_{trig}']] * raw[:, index[f'f_snd_hed_{trig}']]
        for trig in ('sin', 'cos'))
    if dimension == 59:
        distance = raw[:, index['f_dist']]
        inverse = torch.where(distance > 0, distance.reciprocal(), torch.zeros_like(distance))
        raw[:, index['f_bearing_sin']] = raw[:, index['f_dx']] * inverse
        raw[:, index['f_bearing_cos']] = raw[:, index['f_dy']] * inverse
    return ((raw - x.new_tensor(MEAN, dtype=torch.float64)) /
            x.new_tensor(SCALE, dtype=torch.float64)).float()


class Method(Base):
    def __init__(self, run, initialize=True):
        self.R, self.arm = run, run.a.arm
        self.proxy, self.role, self.version = [], 'private', 0
        self.layout = Layout(run.init)
        self.global_vector = self.layout.flatten(run.init).clone()
        self.diag = {}
        fields = self.arm.split('_')
        self.dimension = int(fields[0][1:])
        self.mode = fields[-1]
        self.joint = 'joint' in fields
        self.plain = 'plain' in fields
        dropped = [] if self.dimension == 66 else VEL + (BEARING if self.dimension == 59 else [])
        self.columns = [j for j, name in enumerate(NAMES) if name not in dropped and
                        (self.plain or j not in BINARY)]
        fixture = hashlib.sha256((P.ROOT / 'papers/pilot/fixture/manifest.json').read_bytes()).hexdigest()
        identity = f'{self.arm}|{getattr(run.a, "seed", 0)}|{fixture}|{PROVENANCE["scaler.json"]}'
        self.context = hashlib.sha256(identity.encode()).digest()[:16]
        mean = torch.tensor(MEAN, device=run.dev, dtype=torch.float64)
        scale = torch.tensor(SCALE, device=run.dev, dtype=torch.float64)
        self.atom0 = (-mean / scale).half().double()
        self.atom1 = ((1 - mean) / scale).half().double()
        self.sampler_stats = torch.zeros(6, device=run.dev, dtype=torch.float64)
        if initialize:
            self.statistics()

    def mask_stats(self, x, y):
        zero, one = x.double() == self.atom0, x.double() == self.atom1
        assert (zero[:, BINARY] | one[:, BINARY]).all(), 'V6 binary schema'
        assert zero[one[:, FIRST]][:, STRUCTURAL].all(), 'V6 first13 schema'
        categories = []
        for group in PROFILES:
            assert (one[:, group].sum(1) <= 1).all(), 'V6 profile schema'
            categories.append((one[:, group].long() * torch.arange(1, 4, device=x.device)).sum(1))
        state = one[:, FIRST].long() * 16 + categories[0] * 4 + categories[1]
        if self.joint:
            stats = torch.zeros(16, 32, 15, device=x.device, dtype=torch.float64)
            for c in range(16):
                for s in range(32):
                    subset = (y == c) & (state == s)
                    n = subset.sum()
                    stats[c, s, 0] = n
                    stats[c, s, 1] = n > 0
                    stats[c, s, 2:] = zero[subset][:, STRUCTURAL].sum(0)
        else:
            stats = torch.zeros(16, 24, device=x.device, dtype=torch.float64)
            for c in range(16):
                subset = y == c
                stats[c, 0] = subset.sum()
                stats[c, 1] = subset.any()
                stats[c, 2] = one[subset, FIRST].sum()
                for offset, category in zip((3, 7), categories):
                    stats[c, offset:offset+4] = torch.bincount(category[subset], minlength=4)
                stats[c, 11:] = zero[subset][:, STRUCTURAL].sum(0)
        return stats, state

    def aggregate_packets(self, vectors, kind):
        total = None
        assert len(vectors) >= 5, 'V6 cohort smaller than5'
        for k, values in enumerate(vectors):
            packet = encode(values, kind, self.context)
            decoded = decode(packet, kind, self.context, self.R.dev)
            self.record(packet, 0, k, 'up', f'geometry_{kind}')
            total = decoded.clone() if total is None else total + decoded
        return total

    def broadcast(self, values, kind, half=False):
        packet = encode(values, kind, self.context, half)
        decoded = decode(packet, kind, self.context, self.R.dev).reshape(values.shape)
        for k in range(self.R.K):
            self.record(packet, 0, k, 'down', f'geometry_{kind}')
        self.packets[kind] = packet
        return decoded

    def eligibility(self, stats, own=False):
        if not self.joint:
            return torch.empty(16, 0, device=self.R.dev, dtype=torch.bool)
        minimum = stats.new_tensor([60] * 16 + [47] * 16)[None, :]
        eligible = stats[:, :, 0] >= minimum
        if not own:
            eligible &= stats[:, :, 1] >= 5
        return eligible

    def parameters(self, stats):
        if self.joint:
            n = stats[:, :, 0]
            weights = n / n.sum(1, keepdim=True).clamp_min(1)
            atoms = stats[:, :, 2:] / n[:, :, None].clamp_min(1)
            return torch.cat([weights[:, :, None], atoms], 2)
        n = stats[:, 0:1].clamp_min(1)
        first = stats[:, 2:3] / n
        receiver = stats[:, 3:7] / n
        sender = stats[:, 7:11] / n
        receiver[:, 0] += (stats[:, 0] == 0).double()
        sender[:, 0] += (stats[:, 0] == 0).double()
        atoms = ((stats[:, 11:] - stats[:, 2:3]) /
                 (stats[:, 0:1] - stats[:, 2:3]).clamp_min(1)).clamp(0, 1)
        return torch.cat([first, receiver, sender, atoms], 1)

    def moment_vector(self, x, y, state, keys):
        d = len(self.columns)
        triangle = torch.triu_indices(d, d, device=x.device)
        values = []
        for c, s in keys:
            selected = y == c
            if s >= 0:
                selected &= state == s
            u = x[selected][:, self.columns].double()
            q = u.T @ u
            values.append(torch.cat([u.new_tensor([len(u)]), u.sum(0), q[triangle[0], triangle[1]]]))
        return torch.stack(values)

    def factors(self, moments):
        d = len(self.columns)
        n = moments[:, 0:1]
        mu = moments[:, 1:1+d] / n.clamp_min(1)
        cov = torch.zeros(len(moments), d, d, device=moments.device, dtype=torch.float64)
        triangle = torch.triu_indices(d, d, device=moments.device)
        cov[:, triangle[0], triangle[1]] = moments[:, 1+d:] / n
        cov = cov + cov.transpose(1, 2) - torch.diag_embed(cov.diagonal(dim1=1, dim2=2))
        cov -= mu[:, :, None] * mu[:, None, :]
        cov = .5 * (cov + cov.transpose(1, 2))
        # Absent own class: declared standard Gaussian, no population moments.
        absent = n[:, 0] == 0
        cov[absent] = torch.eye(d, device=cov.device, dtype=cov.dtype)
        chol = torch.linalg.cholesky(cov + 1e-4 * torch.eye(d, device=cov.device, dtype=cov.dtype))
        lower = torch.tril_indices(d, d, device=cov.device)
        return torch.cat([mu, chol[:, lower[0], lower[1]]], 1)

    def table(self, factors, keys, parameters):
        d = len(self.columns)
        lower = torch.tril_indices(d, d, device=self.R.dev)
        chol = torch.zeros(len(keys), d, d, device=self.R.dev, dtype=torch.float32)
        chol[:, lower[0], lower[1]] = factors[:, d:].float()
        mapping = torch.arange(16, device=self.R.dev)[:, None].expand(16, 32).clone()
        for i, (c, s) in enumerate(keys[16:], 16):
            mapping[c, s] = i
        return dict(mu=factors[:, :d].float(), chol=chol, mapping=mapping,
                    parameters=parameters.float(), keys=keys)

    def statistics(self):
        self.packets = {}
        local, states = zip(*(self.mask_stats(x, y) for x, y in zip(self.R.X, self.R.Y)))
        own = self.mode == 'own'
        if own:
            tables = []
            for k, stats in enumerate(local):
                eligible = self.eligibility(stats, own=True)
                keys = [(c, -1) for c in range(16)] + [tuple(v) for v in eligible.nonzero().tolist()]
                moments = self.moment_vector(self.R.X[k], self.R.Y[k], states[k], keys)
                factors = self.factors(moments).half().double()
                assert torch.isfinite(factors).all(), 'V6 own factor fp16 overflow'
                tables.append(self.table(factors, keys, self.parameters(stats)))
            self.tables = tables
            return
        uploads = [torch.cat([stats.flatten(),
                   (stats[:, :, 0].sum(1) if self.joint else stats[:, 0]).gt(0).double()])
                   for stats in local]
        received = self.aggregate_packets(uploads, 'counts')
        total = received[:-16].reshape(local[0].shape)
        class_n = total[:, :, 0].sum(1) if self.joint else total[:, 0]
        # Class indicators travelled through the same cohort sum, never raw labels.
        class_contributors = received[-16:]
        assert (class_n >= 67).all() and (class_contributors >= 5).all(), 'V6 global class cohort gate'
        eligible = self.eligibility(total)
        if self.joint:
            eligible = self.broadcast(eligible.double(), 'eligibility').bool()
        keys = [(c, -1) for c in range(16)] + [tuple(v) for v in eligible.nonzero().tolist()]
        vectors = [self.moment_vector(x, y, state, keys)
                   for x, y, state in zip(self.R.X, self.R.Y, states)]
        moments = self.aggregate_packets(vectors, 'moments').reshape(vectors[0].shape)
        factors = self.broadcast(self.factors(moments), 'factors', half=True)
        parameters = self.broadcast(self.parameters(total), 'mask')
        self.tables = [self.table(factors, keys, parameters)]
        self.diag.update(eligible_cells=int(eligible.sum()), class_counts=class_n.tolist())

    def draw(self, labels, g, k):
        table = self.tables[k] if self.mode == 'own' else self.tables[0]
        parameter = table['parameters']
        if self.joint:
            weights = parameter[:, :, 0].clone()
            weights[:, 0] += (weights.sum(1) == 0).float()
            state = torch.multinomial(weights[labels], 1, generator=g).squeeze(1)
            first, receiver, sender = state // 16, state % 16 // 4, state % 4
            atoms = parameter[labels, state, 1:]
        else:
            first = (torch.rand(len(labels), device=self.R.dev, generator=g) < parameter[labels, 0]).long()
            receiver = torch.multinomial(parameter[labels, 1:5], 1, generator=g).squeeze(1)
            sender = torch.multinomial(parameter[labels, 5:9], 1, generator=g).squeeze(1)
            atoms = parameter[labels, 9:]
            state = first * 16 + receiver * 4 + sender
        row = table['mapping'][labels, state]
        # Both draws share class/state; shuffle changes alignment only.
        noises = torch.randn(2, len(labels), len(self.columns), device=self.R.dev, generator=g)
        values = table['mu'][row][None] + torch.matmul(table['chol'][row][None], noises[..., None]).squeeze(-1)
        x = self.atom0.float().expand(2, len(labels), 66).clone()
        x[:, :, self.columns] = values
        if not self.plain:
            x[:, :, FIRST] = torch.where(first.bool(), self.atom1[FIRST], self.atom0[FIRST]).float()
            for group, category in zip(PROFILES, (receiver, sender)):
                for offset, j in enumerate(group, 1):
                    x[:, :, j] = torch.where(category == offset, self.atom1[j], self.atom0[j]).float()
            zero = torch.rand(2, len(labels), len(STRUCTURAL), device=self.R.dev, generator=g) < atoms[None]
            zero |= first.bool()[None, :, None]
            x[:, :, STRUCTURAL] = torch.where(zero, self.atom0[STRUCTURAL].float(), x[:, :, STRUCTURAL])
        derived = [] if self.dimension == 66 else [NAMES.index(n) for n in VEL + (BEARING if self.dimension == 59 else [])]
        if derived:
            x = torch.stack([restore_derived(v, self.dimension) for v in x])
            if self.mode == 'shuffle':
                x[0, :, derived] = x[1, :, derived]
        assert torch.isfinite(x).all() and torch.isfinite(x.half()).all(), 'V6 synthetic fp16 input overflow'
        raw = x[0].double() * x.new_tensor(SCALE, dtype=torch.float64) + x.new_tensor(MEAN, dtype=torch.float64)
        distance = raw[:, NAMES.index('f_dist')]
        bearing = raw[:, [NAMES.index(n) for n in BEARING]]
        self.sampler_stats[0] += len(labels)
        self.sampler_stats[1] += (distance <= 0).sum()
        self.sampler_stats[2] += (distance.abs() < 1e-6).sum()
        self.sampler_stats[3] += ((bearing.square().sum(1).sqrt()-1).abs() > .01).sum()
        self.sampler_stats[4] = torch.maximum(self.sampler_stats[4], bearing.abs().max())
        self.sampler_stats[5] += (row < 16).sum() if self.joint else 0
        return x[0]

    def start_round(self):
        self.sampler_stats.zero_()

    def aggregate(self, r):
        values = self.sampler_stats.tolist()
        self.diag['sampler'] = dict(rows=int(values[0]), nonpositive_distance=int(values[1]),
                                   distance_abs_lt_1e6=int(values[2]), bearing_unit_error=int(values[3]),
                                   max_abs_bearing=values[4], fallback_rows=int(values[5]))

    def private_loss(self, k, x, y, g, r):
        real = F.cross_entropy(self.R.priv[k](x)[0].float(), y)
        labels = torch.arange(self.R.a.aux_batch, device=self.R.dev) % 16
        with torch.autocast('cuda', enabled=False):
            synthetic = self.draw(labels, g, k)
        extra = F.cross_entropy(self.R.priv[k](synthetic)[0].float(), labels)
        return real + self.R.a.aux_weight * extra, torch.stack([real.detach(), extra.detach()])

    def state(self):
        return dict(global_vector=self.global_vector, version=0, tables=self.tables,
                    packets=self.packets, diag=self.diag, atom0=self.atom0, atom1=self.atom1,
                    sampler_stats=self.sampler_stats)
