"""P32: support realism x teacher calibration on frozen private backbones (the P24 harness), plus a realism ablation by feature group.

Copula = the round-18 `cop` generator (fd.make_copula/draw_copula): pooled per-class histograms of asinh(z) and normal-score
correlations, both additive under SecureSum; estimated from the same 20 x 16,384 sampled rows as the Gaussian control.
Real and hybrid supports are R3-ineligible diagnostics. Protocol and gates: docs/research/methods/synthetic-support-realism-2026-10-22.md.
No official test; old/group holdouts only.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fd_probe23 import C, basis, decode_stats, encode_stats, fd, fit, forward, P, report, sha, virtual
from fd_probe24 import adapt
from mh.metrics import metrics_from_confusion
from mh.v2_math import pack_vector, unpack

GROUPS = dict(
    snd_kin=['f_snd_pos_noise_x', 'f_snd_pos_noise_y', 'f_snd_spd', 'f_snd_spd_noise', 'f_snd_acl', 'f_snd_acl_noise', 'f_snd_hed_noise',
             'f_snd_hed_sin', 'f_snd_hed_cos', 'f_snd_vx', 'f_snd_vy', 'f_snd_noise_mag'],
    rcv_kin=['f_rcv_spd', 'f_rcv_acl', 'f_rcv_hed_sin', 'f_rcv_hed_cos', 'f_rcv_vx', 'f_rcv_vy'],
    geo=['f_snd_dist_road_edge', 'f_dx', 'f_dy', 'f_dist', 'f_bearing_sin', 'f_bearing_cos', 'f_hed_diff_cos', 'f_rel_speed',
         'f_closing_speed', 'f_spd_diff'],
    sess=['f_delay_s', 'f_first_in_session', 'f_sess_idx', 'f_sess_dt', 'f_sess_dt_send', 'f_sess_dt_skew', 'f_sess_dpos',
          'f_sess_implied_spd', 'f_sess_spd_residual', 'f_sess_dspd', 'f_sess_acl_residual', 'f_sess_dhed', 'f_sess_dmsgid',
          'f_sess_ddist', 'f_sess_ddre', 'f_sess_pos_pred_err'],
    rate=['f_alias_age_s', 'f_rx_rate_1s', 'f_rx_rate_5s', 'f_sender_rate_1s', 'f_sender_rate_5s', 'f_sender_share_5s'],
    prof=['f_snd_profile_normal', 'f_snd_profile_cautious', 'f_snd_profile_aggressive'])


def copula_fit(x, y, keep):
    """fd.make_copula on pooled rows (SecureSum totals equal the pooled sums); returns tables and the real packet sizes."""
    B, lim = fd.COP_BINS, fd.COP_LIM; d = len(keep); dev = x.device
    edges = torch.linspace(-lim, lim, B + 1, device=dev); width = float(edges[1] - edges[0])
    feature = torch.arange(d, device=dev); xs = x[:, keep].float()
    a = torch.asinh(xs).clamp(-lim, lim - 1e-6); b = torch.bucketize(a, edges[1:-1])
    counts = torch.bincount(((y[:, None] * d + feature) * B + b).flatten(), minlength=C * d * B).double()
    hist_packet = pack_vector('counts', counts, count=len(y)); counts = unpack(hist_packet, dev)['values'].double().view(C, d, B)
    n = counts[:, 0].sum(1); assert (n > 0).all()
    cum = torch.cat([torch.zeros(C, d, 1, device=dev, dtype=torch.float64), counts.cumsum(2)], 2) / n[:, None, None]
    flat = cum.float().flatten()
    at = ((y[:, None] * d + feature) * (B + 1) + b); lo, hi = flat[at], flat[at + 1]
    s = torch.special.ndtri((lo + (a - edges[b]) / width * (hi - lo)).clamp(1e-6, 1 - 1e-6)).double()
    tri = torch.triu_indices(d, d, device=dev)
    t1 = torch.stack([s[y == c].sum(0) for c in range(C)]); t2 = torch.stack([s[y == c].T @ s[y == c] for c in range(C)])
    score_packet = pack_vector('moments', torch.cat([t1, t2[:, tri[0], tri[1]]], 1).flatten(), count=len(y))
    total = unpack(score_packet, dev)['values'].double().view(C, -1)
    mean = total[:, :d] / n[:, None]; second = torch.zeros(C, d, d, dtype=torch.float64, device=dev)
    second[:, tri[0], tri[1]] = total[:, d:]; second = second + second.triu(1).transpose(1, 2)
    cov = (second / n[:, None, None] - mean[:, :, None] * mean[:, None, :]).cpu()
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
    return dict(cum=cum.float(), factor=low, keep=keep, edges=edges, width=width), dict(up=up, down=down)


def copula_draw(t, labels, seed):
    """fd.draw_copula, chunked; masked features stay 0."""
    g = torch.Generator(device=labels.device).manual_seed(seed); d = len(t['keep']); out = []
    for i in range(0, len(labels), 512):
        lab = labels[i:i + 512]
        z = torch.randn(len(lab), d, device=lab.device, generator=g)
        u = torch.special.ndtr((t['factor'][lab] @ z[:, :, None]).squeeze(-1)).clamp(1e-6, 1 - 1e-6)
        seq = t['cum'][lab]
        b = (torch.searchsorted(seq, u[:, :, None]).squeeze(-1) - 1).clamp(0, fd.COP_BINS - 1)
        lo, hi = seq.gather(2, b[:, :, None]).squeeze(-1), seq.gather(2, (b + 1)[:, :, None]).squeeze(-1)
        frac = ((u - lo) / (hi - lo).clamp_min(1e-12)).clamp(0, 1)
        x = torch.zeros(len(lab), 66, device=lab.device); x[:, t['keep']] = torch.sinh(t['edges'][b] + frac * t['width'])
        out.append(x)
    x = torch.cat(out); assert torch.isfinite(x).all() and torch.isfinite(x.half()).all()
    return x


def hybrid(rows, labels, group, active, mu, cov, seed):
    """Replace group G of real rows by the class Gaussian conditional on the remaining active features (same pooled moments)."""
    g = torch.Generator(device=rows.device).manual_seed(seed); out = rows.clone()
    rest = [i for i in active if i not in group]
    for c in range(C):
        idx = torch.where(labels == c)[0]
        if not len(idx): continue
        S = cov[c].double(); m = mu[c].double()
        Srr = S[rest][:, rest] + 1e-6 * torch.eye(len(rest), device=S.device, dtype=S.dtype)
        K = torch.linalg.solve(Srr, S[rest][:, group]).T            # Sigma_GR Sigma_RR^-1
        mean = m[group] + (rows[idx][:, rest].double() - m[rest]) @ K.T
        cond = S[group][:, group] - K @ S[rest][:, group]; cond = (cond + cond.T) / 2
        e, v = torch.linalg.eigh(cond); factor = v * e.clamp_min(0).sqrt()[None, :]
        z = torch.randn(len(idx), len(group), device=rows.device, generator=g, dtype=torch.float64)
        out[idx[:, None], torch.tensor(group, device=rows.device)[None, :]] = (mean + z @ factor.T).float()
    assert torch.isfinite(out).all() and torch.isfinite(out.half()).all()
    return out


def summary(clients, key):
    rows = [v[key]['group'] for v in clients.values()]
    f = [r['metrics']['f1_macro'] for r in rows]
    return dict(mean=float(np.mean(f)), benign=float(np.mean([r['per_class'][0] for r in rows])), worst=float(np.min(f)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True); p.add_argument('--out', required=True)
    p.add_argument('--clients', default=','.join(map(str, range(20)))); p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--bank', type=int, default=4096)
    a = p.parse_args(); t0 = time.monotonic(); log = lambda *v: print(round(time.monotonic() - t0, 1), *v, flush=True)
    fd.deterministic(); torch.set_num_threads(2); dev = torch.device('cuda')
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected
    assert manifest['group_overlap'] == 0 and manifest['train_frac'] == .025
    names = manifest['features']; zero = [names.index(n) for n in fd.BLOCK + fd.POSITION]
    active = [i for i in range(66) if i not in zero]
    groups = {g: [names.index(n) for n in v] for g, v in GROUPS.items()}
    assert sorted(sum(groups.values(), [])) == active, 'groups must partition the 53 active features'
    def read(name):
        x = torch.from_numpy(np.load(fx / f'{name}_X.npy').astype('float32')).to(dev); x[:, zero] = 0
        return x, torch.from_numpy(np.load(fx / f'{name}_y.npy').astype('int64')).to(dev)
    w = torch.load(a.weights, map_location=dev, weights_only=True)
    teacher = P.make('dagsnet').to(dev); teacher.load_state_dict(w['proxy'][0], strict=True); teacher.eval()
    teacher.requires_grad_(False); proj = basis(teacher.head[5].weight, 48)
    # P24 setup, unchanged: same row samples, Project48 statistics, virtual bank and head fit.
    gen = torch.Generator(device='cpu').manual_seed(22)
    own, total, input_n, input_s, input_q = {}, 0, torch.zeros(C, device=dev, dtype=torch.float64), \
        torch.zeros(C, 66, device=dev, dtype=torch.float64), torch.zeros(C, 66, 66, device=dev, dtype=torch.float64)
    for k in range(20):
        x, y = read(f'train_{k}'); idx = torch.randperm(len(y), generator=gen)[:16384].to(dev); x, y = x[idx], y[idx]
        z, h, _ = forward(teacher, x, feats=True)
        values, stat_bytes = encode_stats(h, y, 'project48', proj); total = total + values
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
    # Same pool for every other support: the 20 x 16,384 sampled rows behind the Gaussian moments.
    pool_x = torch.cat([own[k][0] for k in range(20)]); pool_y = torch.cat([own[k][1] for k in range(20)])
    rng = np.random.default_rng(27); pool_lab = pool_y.cpu().numpy(); lab = labels.cpu().numpy(); pick = np.empty(len(lab), dtype='int64')
    for c in range(C):
        slots = np.where(lab == c)[0]; pick[slots] = rng.choice(np.where(pool_lab == c)[0], len(slots), replace=True)
    real = pool_x[torch.from_numpy(pick).to(dev)].clone(); assert torch.equal(pool_y[torch.from_numpy(pick).to(dev)], labels)
    tables, copula_bytes = copula_fit(pool_x, pool_y, torch.tensor(active, device=dev))
    supports = dict(gaussian=gaussian, copula=copula_draw(tables, labels, 32), real=real)
    for i, (gname, cols) in enumerate(groups.items()):
        supports['hyb_' + gname] = hybrid(real, labels, cols, active, im, ic, 320 + i)
    for x in supports.values(): assert (x[:, zero] == 0).all()
    log('generated', a.bank, 'rows x', list(supports))
    teachers = dict(original=teacher, calibrated=calibrated)
    arms = [(t, s) for t in teachers for s in ('gaussian', 'copula', 'real')] + [('calibrated', 'hyb_' + g) for g in groups]
    targets, diagnostics = {}, {}
    for t, s in arms:
        z = forward(teachers[t], supports[s]); targets[t, s] = z.log_softmax(1); q = z.softmax(1)
        diagnostics[t + '_' + s] = dict(class_agreement=float((z.argmax(1) == labels).float().mean()),
                                        entropy=float(-(q * z.log_softmax(1)).sum(1).mean()), target_mass=q.mean(0).tolist())
    evals = {split: read(split) for split in ('old', 'group')}
    out = dict(weights_sha=sha(a.weights), fixture_sha=sha(fx / 'manifest.json'), source_sha=sha(__file__),
               protocol='docs/research/methods/synthetic-support-realism-2026-10-22.md',
               scope='frozen backbone and BN; own CE + real/synthetic KL; real/hyb supports are R3-ineligible diagnostics; no official test',
               args=vars(a), groups=GROUPS, copula_packet_bytes_per_client=copula_bytes, calibration_stats_up_bytes=stat_bytes,
               generation_diagnostics=diagnostics, clients={})
    predictions, heads = {}, {}; path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    global_counts = torch.tensor([sum(manifest['counts'][f'train_{k}'][c] for k in range(20)) for c in range(C)], device=dev)
    pg = ((global_counts + 1) / (global_counts.sum() + C)).log()
    for k in map(int, a.clients.split(',')):
        private = P.make('dagsnet').to(dev); private.load_state_dict(w['private'][k], strict=True); private.eval()
        private.requires_grad_(False)
        x, y, ht, zt = own[k]; _, hp, _ = forward(private, x, feats=True)
        syn_h = {s: forward(private, xx, feats=True)[1].float() for s, xx in supports.items()}
        features = {split: (yy, forward(private, xx, feats=True)[1].float()) for split, (xx, yy) in evals.items()}
        counts = torch.tensor(manifest['counts'][f'train_{k}'], device=dev)
        shift = ((counts + 1) / (counts.sum() + C)).log() - pg
        proxy_ce = F.cross_entropy(zt, y).detach()
        out['clients'][k] = dict(initial={split: report(yy, private.head[5](hh).argmax(1)) for split, (yy, hh) in features.items()})
        for t, s in arms:
            key = t + '_' + s
            with torch.no_grad(): real_target = teachers[t].head[5](ht).log_softmax(1)
            adapted = adapt(private.head[5], hp.float(), y, syn_h[s], targets[t, s], real_target, proxy_ce, shift, a.steps, 2400 + k)
            out['clients'][k][key] = {}
            for split, (yy, hh) in features.items():
                with torch.no_grad(): pred = adapted(hh).argmax(1)
                out['clients'][k][key][split] = report(yy, pred)
                predictions[f'{k}_{key}_{split}'] = pred.cpu().numpy().astype('uint8')
            heads[f'{k}_{key}'] = {nm: v.detach().cpu() for nm, v in adapted.state_dict().items()}
        path.write_text(json.dumps(out, indent=1) + '\n')
        log('client', k, {nm: round(v['group']['metrics']['f1_macro'], 4) for nm, v in out['clients'][k].items()})
        del private
    for split, (xx, yy) in evals.items(): predictions[split + '_y'] = yy.cpu().numpy().astype('uint8')
    np.savez_compressed(path.with_suffix('.npz'), **predictions); torch.save(heads, path.with_suffix('.pt'))
    out['summary'] = {key: summary(out['clients'], key) for key in ['initial'] + [t + '_' + s for t, s in arms]}
    # Audit: every stored metric recomputed from the saved predictions (predictions -> confusion -> 10 metrics).
    saved = np.load(path.with_suffix('.npz')); checked = 0
    for k, arms_k in out['clients'].items():
        for key, splits in arms_k.items():
            if key == 'initial': continue
            for split, rep in splits.items():
                yy, pred = saved[split + '_y'].astype('int64'), saved[f'{k}_{key}_{split}'].astype('int64')
                matrix = np.bincount(yy * C + pred, minlength=C * C).reshape(C, C)
                assert matrix.tolist() == rep['confusion'] and metrics_from_confusion(matrix) == rep['metrics']; checked += 1
    out.update(predictions_sha=sha(path.with_suffix('.npz')), heads_sha=sha(path.with_suffix('.pt')), audit_checks=checked,
               seconds=time.monotonic() - t0)
    path.write_text(json.dumps(out, indent=1) + '\n'); log('saved', str(path), 'audit checks', checked)


if __name__ == '__main__':
    main()
