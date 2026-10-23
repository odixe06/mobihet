"""P34: structure-preserving support generators on the P32 frozen-head harness (Project48 teacher).

Priors ported for m4b (receiver profile and m4b columns stay 0): the V4-S structural sampler (.compat/mh/v4_support.py) and the V6 G61
joint pool sampler (archive geometry_moments.py). New: copula with (a) an exact atom slot per variable, (b) per-class first-in-session state,
(d) the sender profile as one categorical variable, (c) extractor identities recomputed from primitives. All statistics are sums over the
pooled sampled rows (= SecureSum totals). Anchors gaussian/copula/real (P32) and own_bal (P33b) are reused from the same harness; the Gaussian
arm of client 0 is re-run and must reproduce P32. Protocol: docs/research/methods/structured-support-generator-2026-10-22.md. No official test.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fd_probe23 import C, basis, decode_stats, encode_stats, fd, fit, forward, P, report, sha, virtual
from fd_probe24 import adapt
from fd_probe32 import summary
from fd_probe33 import geometry, per_class_rows
from mh.metrics import metrics_from_confusion
from mh.v2_math import pack_vector, unpack

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))
from core.feature_support_schema import ATOM0, ATOM1, FIRST, PROFILES, STRUCTURAL

SND = PROFILES[1]   # receiver profile (PROFILES[0]) is zeroed by m4b
VEL = ['f_rcv_vx', 'f_rcv_vy', 'f_snd_vx', 'f_snd_vy', 'f_hed_diff_cos']
DERIVED = ['f_snd_vx', 'f_snd_vy', 'f_rcv_vx', 'f_rcv_vy', 'f_hed_diff_cos', 'f_dist', 'f_bearing_sin', 'f_bearing_cos', 'f_rel_speed',
           'f_closing_speed', 'f_spd_diff', 'f_snd_noise_mag', 'f_sess_dt_send', 'f_sess_dpos', 'f_sess_implied_spd', 'f_sess_dspd']
ARMS = ['v4s', 'g61', 'cop_atom', 'cop_atom_state', 'pcop_noid', 'pcop']


# ---- V4-S structural sampler (port of .compat/mh/v4_support.py, structural arm) ----
def v4s_fit(x, y, atom0):
    D = x.shape[1]; zero = x == atom0; m = torch.zeros(C, 1 + 3 * D, device=x.device, dtype=torch.float64)
    one = x == atom0.new_tensor(ATOM1).half().float()
    assert (zero[:, [FIRST] + SND] | one[:, [FIRST] + SND]).all() and zero[one[:, FIRST]][:, STRUCTURAL].all()
    assert (one[:, SND].sum(1) <= 1).all()
    for c in range(C):
        v = x[y == c].double(); z = zero[y == c]; other = v.masked_fill(z, 0)
        m[c, 0] = len(v); m[c, 1:1 + D] = z.sum(0); m[c, 1 + D:1 + 2 * D] = other.sum(0); m[c, 1 + 2 * D:] = other.square().sum(0)
    n = m[:, 0:1]; atoms = m[:, 1:1 + D]; count = (n - atoms).clamp_min(1)
    sums, squares = m[:, 1 + D:1 + 2 * D], m[:, 1 + 2 * D:]
    nonmu = sums / count; nonsd = (squares / count - nonmu.square()).clamp_min(1e-4).sqrt()
    a0 = atom0.double(); mu = (sums + atoms * a0) / n.clamp_min(1)
    sd = ((squares + atoms * a0.square()) / n.clamp_min(1) - mu.square()).clamp_min(1e-4).sqrt()
    table = torch.stack([mu, sd, atoms / n.clamp_min(1), nonmu, nonsd])
    up = len(pack_vector('moments', m.flatten(), count=len(y))); packet = pack_vector('moments', table.flatten(), half=True)
    return unpack(packet, x.device)['values'].reshape(5, C, D).float(), dict(up=up, down=len(packet))


def v4s_draw(table, labels, seed, atom0, atom1):
    g = torch.Generator(device=labels.device).manual_seed(seed); dev = labels.device; n, D = len(labels), table.shape[2]
    mu, sd, rho, nonmu, nonsd = table
    x = nonmu[labels] + nonsd[labels] * torch.randn(n, D, device=dev, generator=g)
    x = torch.where(torch.rand(x.shape, device=dev, generator=g) < rho[labels], atom0, x)
    first = torch.rand(n, device=dev, generator=g) < (1 - rho[labels, FIRST])
    x[:, FIRST] = torch.where(first, atom1[FIRST], atom0[FIRST])
    probs = (1 - rho[:, SND]).clamp(0, 1); unknown = (1 - probs.sum(1, keepdim=True)).clamp_min(0)
    p = torch.cat([unknown, probs], 1); p = p / p.sum(1, keepdim=True).clamp_min(1e-8)
    category = torch.multinomial(p[labels], 1, generator=g).squeeze(1)
    for offset, j in enumerate(SND, 1): x[:, j] = torch.where(category == offset, atom1[j], atom0[j])
    pfirst = 1 - rho[:, FIRST:FIRST + 1]
    conditional = ((rho[:, STRUCTURAL] - pfirst) / (1 - pfirst).clamp_min(1e-8)).clamp(0, 1)
    mask = torch.rand(n, len(STRUCTURAL), device=dev, generator=g) < conditional[labels]
    fresh = nonmu[labels][:, STRUCTURAL] + nonsd[labels][:, STRUCTURAL] * torch.randn(mask.shape, device=dev, generator=g)
    x[:, STRUCTURAL] = torch.where(mask | first[:, None], atom0[STRUCTURAL], fresh)
    return x


# ---- V6 G61 joint pool sampler (port of geometry_moments.py; state = first x sender profile) ----
def g61_fit(x, y, owner, atom0, atom1, columns):
    zero, one = x == atom0, x == atom1
    state = one[:, FIRST].long() * 16 + (one[:, SND].long() * torch.arange(1, 4, device=x.device)).sum(1)
    per_client = torch.bincount((owner * C + y) * 32 + state, minlength=20 * C * 32).view(20, C, 32).double()
    n, contributors = per_client.sum(0), (per_client > 0).sum(0).double()   # contributors: clients holding the cell
    cell, zs = (y * 32 + state).cpu(), zero[:, STRUCTURAL].double().cpu()   # weighted bincount is deterministic only on CPU
    atoms = torch.stack([torch.bincount(cell, weights=zs[:, j], minlength=C * 32) for j in range(len(STRUCTURAL))], 1)
    atoms = atoms.view(C, 32, -1).to(x.device)
    eligible = (n >= n.new_tensor([60] * 16 + [47] * 16)[None]) & (contributors >= 5)
    keys = [(c, -1) for c in range(C)] + [tuple(v) for v in eligible.nonzero().tolist()]
    d = len(columns); lower = torch.tril_indices(d, d, device=x.device); factors = []
    for c, s in keys:
        u = x[(y == c) & ((state == s) if s >= 0 else True)][:, columns].double(); m = len(u)
        mu = u.mean(0); cov = u.T @ u / m - mu[:, None] * mu[None, :]; cov = .5 * (cov + cov.T)
        chol = torch.linalg.cholesky(cov + 1e-4 * torch.eye(d, device=x.device, dtype=torch.float64))
        factors.append(torch.cat([mu, chol[lower[0], lower[1]]]))
    packet = pack_vector('moments', torch.stack(factors).flatten(), half=True)
    factors = unpack(packet, x.device)['values'].reshape(len(keys), -1).float()
    chol = torch.zeros(len(keys), d, d, device=x.device); chol[:, lower[0], lower[1]] = factors[:, d:]
    mapping = torch.arange(C, device=x.device)[:, None].expand(C, 32).clone()
    for i, (c, s) in enumerate(keys[C:], C): mapping[c, s] = i
    weights = n / n.sum(1, keepdim=True).clamp_min(1); rate = atoms / n[:, :, None].clamp_min(1)
    up = len(pack_vector('counts', torch.cat([n.flatten(), contributors.flatten(), atoms.flatten()]), count=len(y))) + \
        len(pack_vector('moments', torch.zeros(len(keys) * (1 + d + d * (d + 1) // 2), device=x.device), count=len(y)))
    down = len(packet) + len(pack_vector('moments', torch.cat([weights[:, :, None], rate], 2).flatten()))
    return dict(mu=factors[:, :d], chol=chol, mapping=mapping, weights=weights.float(), rate=rate.float(), columns=columns,
                eligible=int(eligible.sum())), dict(up=up, down=down)


def g61_draw(t, labels, seed, atom0, atom1, scale, mean, index):
    g = torch.Generator(device=labels.device).manual_seed(seed); dev = labels.device; n = len(labels)
    weights = t['weights'].clone(); weights[:, 0] += (weights.sum(1) == 0).float()
    state = torch.multinomial(weights[labels], 1, generator=g).squeeze(1)
    first, sender = state // 16, state % 4
    row = t['mapping'][labels, state]
    noise = torch.randn(n, len(t['columns']), device=dev, generator=g)
    x = atom0.expand(n, 66).clone()
    x[:, t['columns']] = t['mu'][row] + (t['chol'][row] @ noise[:, :, None]).squeeze(-1)
    x[:, FIRST] = torch.where(first.bool(), atom1[FIRST], atom0[FIRST])
    for offset, j in enumerate(SND, 1): x[:, j] = torch.where(sender == offset, atom1[j], atom0[j])
    zero = (torch.rand(n, len(STRUCTURAL), device=dev, generator=g) < t['rate'][labels, state]) | first.bool()[:, None]
    x[:, STRUCTURAL] = torch.where(zero, atom0[STRUCTURAL], x[:, STRUCTURAL])
    raw = x.double() * scale + mean   # restore_derived(x, 61): velocities and heading difference from speed and heading
    for side in ('rcv', 'snd'):
        for axis, trig in (('x', 'sin'), ('y', 'cos')):
            raw[:, index[f'f_{side}_v{axis}']] = raw[:, index[f'f_{side}_spd']] * raw[:, index[f'f_{side}_hed_{trig}']]
    raw[:, index['f_hed_diff_cos']] = sum(raw[:, index[f'f_rcv_hed_{tr}']] * raw[:, index[f'f_snd_hed_{tr}']] for tr in ('sin', 'cos'))
    return ((raw - mean) / scale).float()


# ---- copula with exact atoms (P32 copula, round-18 fd.make_copula, plus one atom slot per variable) ----
def acop_fit(v, y, atom):
    """Pooled per-class histograms of asinh(z) in B public bins, the grid shifted per variable so that its public atom value is a bin edge;
    rows equal to the atom go to their own slot (atom = nan: no slot). Normal-score correlations as in the P32 copula. Sums only."""
    B, lim = fd.COP_BINS, fd.COP_LIM; dev = v.device; d = v.shape[1]; S = B + 1
    edges = torch.linspace(-lim, lim, B + 1, device=dev); width = float(edges[1] - edges[0])
    has = ~torch.isnan(atom); a = torch.asinh(torch.where(has, atom, torch.zeros_like(atom)))
    k_in = (a[:, None] - edges[None, 1:-1]).abs().argmin(1) + 1
    kstar = torch.where(has, k_in, torch.full_like(k_in, B)); delta = torch.where(has, a - edges[k_in], torch.zeros_like(a))
    is_atom = v == atom[None]
    t = (torch.asinh(v) - delta[None]).clamp(-lim, lim - 1e-6); b = torch.bucketize(t, edges[1:-1])
    slot = torch.where(is_atom, kstar[None].expand_as(b), b + (b >= kstar[None]).long())
    feature = torch.arange(d, device=dev)
    counts = torch.bincount(((y[:, None] * d + feature) * S + slot).flatten(), minlength=C * d * S).double()
    hist_packet = pack_vector('counts', counts, count=len(y)); counts = unpack(hist_packet, dev)['values'].double().view(C, d, S)
    n = counts[:, 0].sum(1)
    cum = torch.cat([torch.zeros(C, d, 1, device=dev, dtype=torch.float64), counts.cumsum(2)], 2) / n.clamp_min(1)[:, None, None]
    flat = cum.float().flatten(); at = (y[:, None] * d + feature) * (S + 1) + slot; lo, hi = flat[at], flat[at + 1]
    frac = torch.where(is_atom, torch.full_like(t, .5), (t - edges[b]) / width)
    s = torch.special.ndtri((lo + frac * (hi - lo)).clamp(1e-6, 1 - 1e-6)).double()
    tri = torch.triu_indices(d, d, device=dev)
    t1 = torch.stack([s[y == c].sum(0) for c in range(C)]); t2 = torch.stack([s[y == c].T @ s[y == c] for c in range(C)])
    score_packet = pack_vector('moments', torch.cat([t1, t2[:, tri[0], tri[1]]], 1).flatten(), count=len(y))
    total = unpack(score_packet, dev)['values'].double().view(C, -1)
    mean = total[:, :d] / n.clamp_min(1)[:, None]; second = torch.zeros(C, d, d, dtype=torch.float64, device=dev)
    second[:, tri[0], tri[1]] = total[:, d:]; second = second + second.triu(1).transpose(1, 2)
    cov = (second / n.clamp_min(1)[:, None, None] - mean[:, :, None] * mean[:, None, :]).cpu()
    var = cov.diagonal(dim1=1, dim2=2); live = var > 1e-8
    sd = var.clamp_min(1e-8).sqrt(); corr = cov / (sd[:, :, None] * sd[:, None, :])
    corr = torch.where(live[:, :, None] & live[:, None, :], corr, torch.zeros_like(corr)); corr.diagonal(dim1=1, dim2=2).fill_(1.)
    value, vector = torch.linalg.eigh(corr)
    corr = vector @ torch.diag_embed(value.clamp_min(1e-4)) @ vector.transpose(1, 2)
    unit = corr.diagonal(dim1=1, dim2=2).sqrt(); corr = corr / (unit[:, :, None] * unit[:, None, :])
    factor = torch.linalg.cholesky(corr).float().to(dev)
    corr_packet = pack_vector('moments', factor[:, tri[1], tri[0]].flatten())
    low = torch.zeros(C, d, d, device=dev); low[:, tri[1], tri[0]] = unpack(corr_packet, dev)['values'].view(C, -1)
    up = len(hist_packet) + len(score_packet); down = len(pack_vector('counts', counts.flatten())) + len(corr_packet)
    return dict(cum=cum.float(), factor=low, kstar=kstar, delta=delta, atom=atom, edges=edges, width=width, n=n), dict(up=up, down=down)


def acop_draw(t, labels, g):
    B = fd.COP_BINS; d = len(t['kstar']); out = []
    for i in range(0, len(labels), 512):
        lab = labels[i:i + 512]
        z = torch.randn(len(lab), d, device=lab.device, generator=g)
        u = torch.special.ndtr((t['factor'][lab] @ z[:, :, None]).squeeze(-1)).clamp(1e-6, 1 - 1e-6)
        seq = t['cum'][lab]
        slot = (torch.searchsorted(seq, u[:, :, None]).squeeze(-1) - 1).clamp(0, B)
        lo, hi = seq.gather(2, slot[:, :, None]).squeeze(-1), seq.gather(2, (slot + 1)[:, :, None]).squeeze(-1)
        frac = ((u - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)
        b = (slot - (slot > t['kstar'][None]).long()).clamp(max=B - 1)
        val = torch.sinh(t['edges'][b] + frac * t['width'] + t['delta'][None])
        out.append(torch.where(slot == t['kstar'][None], t['atom'][None].expand_as(val), val))
    return torch.cat(out)


def derive(x, nonfirst, scale, mean, index):
    """Extractor identities (vnx_features.py) in raw scale; session ones on non-first rows only. Heading pairs normalized first."""
    raw = x.double() * scale + mean; R = lambda name: raw[:, index[name]]
    def put(name, value, rows=None):
        if rows is None: raw[:, index[name]] = value
        else: raw[rows, index[name]] = value[rows]
    trig = {}
    for side in ('snd', 'rcv'):
        s, c = R(f'f_{side}_hed_sin'), R(f'f_{side}_hed_cos'); r = torch.hypot(s, c); ok = r > 1e-6
        s, c = torch.where(ok, s / r.clamp_min(1e-6), torch.zeros_like(s)), torch.where(ok, c / r.clamp_min(1e-6), torch.ones_like(c))
        put(f'f_{side}_hed_sin', s); put(f'f_{side}_hed_cos', c); trig[side] = (s, c)
        put(f'f_{side}_vx', R(f'f_{side}_spd') * s); put(f'f_{side}_vy', R(f'f_{side}_spd') * c)
    put('f_hed_diff_cos', trig['snd'][0] * trig['rcv'][0] + trig['snd'][1] * trig['rcv'][1])
    dx, dy = R('f_dx'), R('f_dy'); dist = torch.hypot(dx, dy); inv = torch.where(dist > 0, 1 / dist.clamp_min(1e-12), torch.zeros_like(dist))
    put('f_dist', dist); put('f_bearing_sin', dx * inv); put('f_bearing_cos', dy * inv)
    rvx, rvy = R('f_snd_vx') - R('f_rcv_vx'), R('f_snd_vy') - R('f_rcv_vy')
    put('f_rel_speed', torch.hypot(rvx, rvy)); put('f_closing_speed', -(rvx * dx + rvy * dy) * inv)
    put('f_spd_diff', R('f_snd_spd') - R('f_rcv_spd'))
    put('f_snd_noise_mag', torch.hypot(R('f_snd_pos_noise_x'), R('f_snd_pos_noise_y')))
    implied = (R('f_snd_spd') + R('f_sess_spd_residual')).clamp_min(0); dt = R('f_sess_dt')
    put('f_sess_implied_spd', implied, nonfirst); put('f_sess_spd_residual', implied - R('f_snd_spd'), nonfirst)
    put('f_sess_dpos', implied * dt, nonfirst); put('f_sess_dspd', (R('f_sess_acl_residual') + R('f_snd_acl')) * dt, nonfirst)
    put('f_sess_dt_send', dt - R('f_sess_dt_skew'), nonfirst)
    return ((raw - mean) / scale).float()


def state_copula(x, y, atom0, atom1, cols, profile, seed, labels):
    """(a)+(b) [+(d)]: one exact-atom copula per first-in-session state; returns rows (66 cols, z scale), state per row, bytes."""
    dev = x.device; first = x[:, FIRST] == atom1[FIRST]
    v = x[:, cols]; atom = atom0[cols]
    if profile:   # sender profile as one categorical variable 0/1/2 (always known in the data), no atom slot
        code = (x[:, SND] == atom1[SND]).float().argmax(1).float()
        v = torch.cat([v, code[:, None]], 1); atom = torch.cat([atom, atom.new_tensor([float('nan')])])
    p_first = torch.stack([first[y == c].float().mean() for c in range(C)])
    g = torch.Generator(device=dev).manual_seed(seed)
    state = torch.rand(len(labels), device=dev, generator=g) < p_first[labels]
    out = atom0.expand(len(labels), 66).clone(); size = dict(up=0, down=0)
    for s in (False, True):
        rows = first == s
        if not rows.any(): continue
        tables, b = acop_fit(v[rows], y[rows], atom); size = {k: size[k] + b[k] for k in size}
        idx = torch.where(state == s)[0]
        if not len(idx): continue
        assert (tables['n'][labels[idx]] > 0).all()
        val = acop_draw(tables, labels[idx], g)
        out[idx[:, None], torch.tensor(cols, device=dev)[None]] = val[:, :len(cols)]
        if profile:
            code = val[:, -1].round().clamp(0, 2).long()
            for offset, j in enumerate(SND): out[idx, j] = torch.where(code == offset, atom1[j], atom0[j])
    out[:, FIRST] = torch.where(state, atom1[FIRST], atom0[FIRST])
    sess = STRUCTURAL + [FIRST + 1]   # 13 change features + f_sess_idx: raw 0 on the first message of a session
    out[:, sess] = torch.where(state[:, None], atom0[sess], out[:, sess])
    return out, state, size


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True); p.add_argument('--out', required=True)
    p.add_argument('--p32', required=True); p.add_argument('--p33b', required=True)
    p.add_argument('--clients', default=','.join(map(str, range(20)))); p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--bank', type=int, default=4096)
    a = p.parse_args(); t0 = time.monotonic(); log = lambda *v: print(round(time.monotonic() - t0, 1), *v, flush=True)
    fd.deterministic(); torch.set_num_threads(2); dev = torch.device('cuda')
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected
    p32 = json.loads(Path(a.p32).read_text()); p33b = json.loads(Path(a.p33b).read_text())
    assert p32['weights_sha'] == sha(a.weights) == p33b['weights_sha']; saved = np.load(Path(a.p32).with_suffix('.npz'))
    names = manifest['features']; index = {n: j for j, n in enumerate(names)}; zero = [index[n] for n in fd.BLOCK + fd.POSITION]
    active = [i for i in range(66) if i not in zero]
    scaler = json.loads((ROOT / 'knowledge/scaler.json').read_text())['features']
    mean = torch.tensor([scaler[n]['mean'] for n in names], device=dev, dtype=torch.float64)
    scale = torch.tensor([scaler[n]['std_used'] for n in names], device=dev, dtype=torch.float64)
    atom0 = torch.tensor(ATOM0, device=dev, dtype=torch.float64).half().float()   # fixture stores fp16 z values
    atom1 = torch.tensor(ATOM1, device=dev, dtype=torch.float64).half().float()
    assert torch.allclose(atom0.double(), (-mean / scale).half().double())
    def read(name):
        x = torch.from_numpy(np.load(fx / f'{name}_X.npy').astype('float32')).to(dev); x[:, zero] = 0
        return x, torch.from_numpy(np.load(fx / f'{name}_y.npy').astype('int64')).to(dev)
    w = torch.load(a.weights, map_location=dev, weights_only=True)
    teacher = P.make('dagsnet').to(dev); teacher.load_state_dict(w['proxy'][0], strict=True); teacher.eval()
    teacher.requires_grad_(False); proj = basis(teacher.head[5].weight, 48)
    # P32 setup, unchanged: rows, Project48 calibration, Gaussian bank labels, real-support picks (excluded from the geometry reference).
    gen = torch.Generator(device='cpu').manual_seed(22)
    own, total, input_n, input_s, input_q = {}, 0, torch.zeros(C, device=dev, dtype=torch.float64), \
        torch.zeros(C, 66, device=dev, dtype=torch.float64), torch.zeros(C, 66, 66, device=dev, dtype=torch.float64)
    for k in range(20):
        x, y = read(f'train_{k}'); idx = torch.randperm(len(y), generator=gen)[:16384].to(dev); x, y = x[idx], y[idx]
        z, h, _ = forward(teacher, x, feats=True)
        values, _ = encode_stats(h, y, 'project48', proj); total = total + values
        own[k] = (x, y, h.float(), z)
        one = F.one_hot(y, C).double(); input_n += one.sum(0); input_s += one.T @ x.double()
        for c in range(C): xx = x[y == c].double(); input_q[c] += xx.T @ xx
    n, mu, cov = decode_stats(total, 'project48', proj)
    bank_h, bank_y = virtual(mu, cov, n / n.sum(), 200000, 23)
    fc = torch.nn.Linear(48, C).to(dev)
    with torch.no_grad(): fc.weight.copy_(teacher.head[5].weight @ proj.float()); fc.bias.copy_(teacher.head[5].bias)
    fc = fit(fc, bank_h, bank_y, 3000, 23); del bank_h, bank_y
    calibrated = copy.deepcopy(teacher)
    with torch.no_grad(): calibrated.head[5].weight.copy_(fc.weight @ proj.float().T); calibrated.head[5].bias.copy_(fc.bias)
    calibrated.requires_grad_(False)
    im = input_s / input_n[:, None]
    ic = (input_q - input_n[:, None, None] * im[:, :, None] * im[:, None, :]) / (input_n[:, None, None] - 1)
    gaussian, labels = virtual(im, ic, torch.ones(C, device=dev) / C, a.bank, 24); gaussian[:, zero] = 0
    pool_x = torch.cat([own[k][0] for k in range(20)]); pool_y = torch.cat([own[k][1] for k in range(20)])
    rng = np.random.default_rng(27); pool_lab = pool_y.cpu().numpy(); lab = labels.cpu().numpy(); pick = np.empty(len(lab), dtype='int64')
    for c in range(C):
        slots = np.where(lab == c)[0]; pick[slots] = rng.choice(np.where(pool_lab == c)[0], len(slots), replace=True)
    owner = torch.arange(20, device=dev).repeat_interleave(16384)
    # Supports.
    supports, sizes = {}, {}
    table, sizes['v4s'] = v4s_fit(pool_x, pool_y, atom0); supports['v4s'] = v4s_draw(table, labels, 41, atom0, atom1)
    g61_cols = [j for j in active if names[j] not in VEL and j not in [FIRST] + SND]
    table, sizes['g61'] = g61_fit(pool_x, pool_y, owner, atom0, atom1, g61_cols)
    supports['g61'] = g61_draw(table, labels, 42, atom0, atom1, scale, mean, index); log('g61 eligible states', table['eligible'])
    tables, sizes['cop_atom'] = acop_fit(pool_x[:, active], pool_y, atom0[active])
    supports['cop_atom'] = atom0.expand(a.bank, 66).clone(); supports['cop_atom'][:, active] = \
        acop_draw(tables, labels, torch.Generator(device=dev).manual_seed(43))
    no_first = [j for j in active if j != FIRST]
    supports['cop_atom_state'], _, sizes['cop_atom_state'] = state_copula(pool_x, pool_y, atom0, atom1, no_first, False, 44, labels)
    no_prof = [j for j in no_first if j not in SND]
    supports['pcop_noid'], _, sizes['pcop_noid'] = state_copula(pool_x, pool_y, atom0, atom1, no_prof, True, 45, labels)
    primitive = [j for j in no_prof if names[j] not in DERIVED]
    assert len(primitive) == 33, len(primitive)
    rows, state, sizes['pcop'] = state_copula(pool_x, pool_y, atom0, atom1, primitive, True, 46, labels)
    supports['pcop'] = derive(rows, ~state, scale, mean, index)
    for s, x in supports.items():
        x[:, zero] = 0; assert torch.isfinite(x).all() and torch.isfinite(x.half()).all(), s
    # Structure checks on pcop: identities idempotent, first => session zeros, one-hot profile.
    again = derive(supports['pcop'], ~state, scale, mean, index); again[:, zero] = 0
    identity_error = float(((again - supports['pcop']).abs()).max())
    assert identity_error < 1e-3, identity_error
    assert (supports['pcop'][state][:, STRUCTURAL] == atom0[STRUCTURAL]).all()
    assert ((supports['pcop'][:, SND] == atom1[SND]).sum(1) == 1).all()
    diagnostics = {}
    for s, x in supports.items():
        z = forward(calibrated, x); q = z.softmax(1)
        diagnostics[s] = dict(class_agreement=float((z.argmax(1) == labels).float().mean()), target_mass=q.mean(0).tolist(),
                              atom_rate={nm: float((x[:, index[nm]] == atom0[index[nm]]).float().mean())
                                         for nm in ('f_snd_spd', 'f_snd_vx', 'f_sess_spd_residual', 'f_sess_dspd')})
    log('generated', {s: round(d['class_agreement'], 3) for s, d in diagnostics.items()}, 'identity_error', identity_error)
    targets = {s: forward(calibrated, x).log_softmax(1) for s, x in supports.items()}
    evals = {split: read(split) for split in ('old', 'group')}
    out = dict(weights_sha=sha(a.weights), fixture_sha=sha(fx / 'manifest.json'), source_sha=sha(__file__), p32_sha=sha(a.p32),
               p33b_sha=sha(a.p33b), protocol='docs/research/methods/structured-support-generator-2026-10-22.md',
               scope='frozen backbone and BN; Project48 teacher; one-shot pooled tables (SecureSum sums); no official test',
               args=vars(a), table_bytes_per_client=sizes, identity_error=identity_error, generation_diagnostics=diagnostics, clients={})
    predictions, heads = {}, {}; path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    global_counts = torch.tensor([sum(manifest['counts'][f'train_{k}'][c] for k in range(20)) for c in range(C)], device=dev)
    pg = ((global_counts + 1) / (global_counts.sum() + C)).log(); picked = np.zeros(len(pool_lab), bool); picked[pick] = True
    owner_np = owner.cpu().numpy()
    for k in map(int, a.clients.split(',')):
        private = P.make('dagsnet').to(dev); private.load_state_dict(w['private'][k], strict=True); private.eval()
        private.requires_grad_(False)
        x, y, ht, zt = own[k]; _, hp, _ = forward(private, x, feats=True)
        features = {split: (yy, forward(private, xx, feats=True)[1].float()) for split, (xx, yy) in evals.items()}
        counts = torch.tensor(manifest['counts'][f'train_{k}'], device=dev)
        shift = ((counts + 1) / (counts.sum() + C)).log() - pg
        proxy_ce = F.cross_entropy(zt, y).detach()
        with torch.no_grad(): real_target = calibrated.head[5](ht).log_softmax(1)
        syn_h = {s: forward(private, xx, feats=True)[1].float() for s, xx in supports.items()}
        if k == 0:   # harness identity: the P32 Gaussian arm of client 0, bit for bit
            head = adapt(private.head[5], hp.float(), y, forward(private, gaussian, feats=True)[1].float(),
                         forward(calibrated, gaussian).log_softmax(1), real_target, proxy_ce, shift, a.steps, 2400)
            with torch.no_grad(): pred = head(features['group'][1]).argmax(1).cpu().numpy()
            assert np.array_equal(pred, saved['0_calibrated_gaussian_group'].astype('int64')); log('client 0 gaussian reproduces P32')
        # Offline mechanism: geometry in this private's feature space, P33 reference rows (other clients, seed [33, k]).
        other = per_class_rows(pool_lab, np.where((owner_np != k) & ~picked)[0], [33, k])
        _, h_pool, _ = forward(private, pool_x, feats=True)
        geo = {s: geometry(h_pool.float(), other, hs, labels) for s, hs in syn_h.items()}
        del h_pool
        out['clients'][k] = dict(geometry=geo)
        for s in ARMS:
            head = adapt(private.head[5], hp.float(), y, syn_h[s], targets[s], real_target, proxy_ce, shift, a.steps, 2400 + k)
            out['clients'][k][s] = {}
            for split, (yy, hh) in features.items():
                with torch.no_grad(): pred = head(hh).argmax(1)
                out['clients'][k][s][split] = report(yy, pred)
                predictions[f'{k}_{s}_{split}'] = pred.cpu().numpy().astype('uint8')
            heads[f'{k}_{s}'] = {nm: v.detach().cpu() for nm, v in head.state_dict().items()}
        path.write_text(json.dumps(out, indent=1) + '\n')
        log('client', k, {s: round(out['clients'][k][s]['group']['metrics']['f1_macro'], 4) for s in ARMS},
            'P32 gaussian/copula/real', [round(p32['clients'][str(k)][f'calibrated_{s}']['group']['metrics']['f1_macro'], 4)
                                          for s in ('gaussian', 'copula', 'real')])
        del private
    for split, (xx, yy) in evals.items(): predictions[split + '_y'] = yy.cpu().numpy().astype('uint8')
    np.savez_compressed(path.with_suffix('.npz'), **predictions); torch.save(heads, path.with_suffix('.pt'))
    saved_now = np.load(path.with_suffix('.npz')); checked = 0
    for k, arms_k in out['clients'].items():
        for s in ARMS:
            for split, rep in arms_k[s].items():
                yy, pred = saved_now[split + '_y'].astype('int64'), saved_now[f'{k}_{s}_{split}'].astype('int64')
                matrix = np.bincount(yy * C + pred, minlength=C * C).reshape(C, C)
                assert matrix.tolist() == rep['confusion'] and metrics_from_confusion(matrix) == rep['metrics']; checked += 1
    # Gates (locked in the protocol) and descriptive mechanism.
    res = {s: summary(out['clients'], s) for s in ARMS}
    anchors = {s: p32['summary']['calibrated_' + s] for s in ('gaussian', 'copula', 'real')}; anchors['own_bal'] = p33b['summary']['own_bal']
    gauss, best_prior = anchors['gaussian'], max(anchors['copula']['mean'], res['v4s']['mean'], res['g61']['mean'], anchors['own_bal']['mean'])
    pc = res['pcop']
    gates = dict(G1=dict(mean=pc['mean'] - gauss['mean'], benign=pc['benign'] - gauss['benign'], worst=pc['worst'] - gauss['worst']),
                 G2=pc['mean'] - best_prior)
    gates['G1']['pass'] = gates['G1']['mean'] >= .005 and gates['G1']['benign'] >= -.005 and gates['G1']['worst'] >= -.010
    gates['pass'] = bool(gates['G1']['pass'] and gates['G2'] >= .003)
    classes = manifest['classes']; key = [classes.index('positionMirroring'), classes.index('suddenStop')]
    pcf = lambda arm, c: float(np.mean([v[arm]['group']['per_class'][c] for v in out['clients'].values()]))
    p32f = lambda arm, c: float(np.mean([v[f'calibrated_{arm}']['group']['per_class'][c] for v in p32['clients'].values()]))
    per_class = {s: {classes[c]: pcf(s, c) - p32f('gaussian', c) for c in range(C)} for s in ARMS}
    per_class['real'] = {classes[c]: p32f('real', c) - p32f('gaussian', c) for c in range(C)}
    gates['G3_descriptive'] = dict(pcop=sum(per_class['pcop'][classes[c]] for c in key), half_real=sum(per_class['real'][classes[c]] for c in key) / 2)
    precision = {s: {classes[c]: float(np.nanmean([v['geometry'][s][c]['precision'] for v in out['clients'].values()])) for c in range(C)}
                 for s in ARMS}
    gates['mechanism_offline'] = dict(suddenStop=precision['pcop']['suddenStop'], positionMirroring=precision['pcop']['positionMirroring'],
                                      sign=precision['pcop']['suddenStop'] >= .75 and precision['pcop']['positionMirroring'] >= .60)
    geometry_mean = {s: {m: float(np.nanmean([[x[m] if x[m] is not None else np.nan for x in v['geometry'][s]] for v in out['clients'].values()]))
                         for m in ('precision', 'recall', 'landing')} for s in ARMS}
    out.update(summary=res, anchors=anchors, gates=gates, per_class_delta_vs_gaussian=per_class, precision_by_class=precision,
               geometry_mean=geometry_mean, predictions_sha=sha(path.with_suffix('.npz')), heads_sha=sha(path.with_suffix('.pt')),
               audit_checks=checked, seconds=time.monotonic() - t0)
    path.write_text(json.dumps(out, indent=1) + '\n'); log('saved', str(path), 'audit checks', checked, json.dumps(gates))


if __name__ == '__main__':
    main()
