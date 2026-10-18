"""Probe 14, no DAGSNet training (design §Probe 14): why the private gp F1 stalls near 0,73-0,75 on the group holdout while
the shared proxy of the same run reaches 0,84-0,85. Audited epoch-15 weights (m4b s42 / s43, gpool s42), m4b mask throughout.
Datasets are only read (the run's own fixture preparation); refits are measurements, not proposals.

T1 cross-client: every private k (and the proxy) on S sampled train rows of every client j -> confusion[k][j] (gp + head),
   private/proxy argmax agreement and KL(proxy || private) per source client.
T2/T7 group holdout: per-client confusions (head, gp) + per-class joint correctness (proxy x private) and the confusion of
   the rows the private loses (proxy right, private wrong), for the gap analysis by local class count.
T3 frozen backbone: last Linear on h256 (LBFGS) and head MLP on c544 (AdamW), fit on the pooled sample of all clients vs the
   client's own rows only (same steps); proxy as reference.
T4 oracle 16-class bias per model, coordinate ascent on `old`, scored on `group`: upper bound of decision-rule fixes.
T5 information: exact duplicate masked fp16 rows with conflicting labels over every train row (client spread, group rows
   found in train); kNN on own rows vs pooled rows of the same size, and a 10x pooled reference.

    python fd_probe14.py --weights <dir with *.pt + expected.json> --out probe14.json [--smoke]
"""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import fd
from fd import P
from mh.metrics import metrics_from_confusion

C = 16
SIZES = dict(full=dict(S=16384, Q=16384, own=100000, pooled=5000, big=50000, mlp_epochs=10),
             smoke=dict(S=256, Q=256, own=600, pooled=30, big=60, mlp_epochs=1))
GRID = torch.linspace(-4, 4, 33)


def cm(y, pred, n=1):
    """n stacked predictions (n, N) -> (n, 16, 16) confusions; a single (N,) prediction -> (16, 16)."""
    if pred.dim() == 1: return torch.bincount(y * C + pred, minlength=C * C).view(C, C)
    off = torch.arange(len(pred), device=y.device)[:, None] * C * C
    return torch.bincount((off + y[None] * C + pred).flatten(), minlength=len(pred) * C * C).view(len(pred), C, C)


def f1(c):
    """Macro F1 over all 16 classes, same as mh.metrics (2 tp / (support + predicted), 0 when both are 0)."""
    c = c.double(); tp = c.diagonal(dim1=-2, dim2=-1)
    return (2 * tp / (c.sum(-1) + c.sum(-2)).clamp_min(1)).mean(-1)


@torch.no_grad()
def forward(model, x, feats=False):
    """Same ops as Run.logits (fp16 autocast, 16384 rows/chunk) on masked rows; optionally the h256 / c544 features."""
    model.eval(); z, h, c = [], [], []
    for s in range(0, len(x), 16384):
        with torch.autocast('cuda', dtype=torch.float16):
            cc = model.combined(x[s:s + 16384]); hh = model.head[:4](cc); zz = model.head[4:](hh)
        z.append(zz.float())
        if feats: h.append(hh.half()); c.append(cc.half())
    return (torch.cat(z), torch.cat(h), torch.cat(c)) if feats else torch.cat(z)


def load(path, dev):
    w = torch.load(path, map_location=dev, weights_only=True)
    make = lambda s: (lambda m: (m.load_state_dict(s, strict=True), m.eval())[1])(P.make('dagsnet').to(dev))
    return [make(s) for s in w['private']], make(w['proxy'][0])


def fit_linear(h, y):
    """Multinomial logistic regression on frozen h (bias column), LBFGS full batch, tiny L2."""
    x = torch.cat([h.float(), torch.ones(len(h), 1, device=h.device)], 1)
    W = torch.zeros(x.shape[1], C, device=h.device, requires_grad=True)
    opt = torch.optim.LBFGS([W], lr=1, max_iter=200, history_size=20, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(x @ W, y) + 1e-4 * W[:-1].pow(2).sum(); loss.backward(); return loss
    with torch.enable_grad(): opt.step(closure)
    W = W.detach()
    return lambda f: torch.cat([f.float(), torch.ones(len(f), 1, device=f.device)], 1) @ W


def fit_mlp(c, y, steps, seed):
    """Fresh head MLP (LayerNorm, 544 -> 256 -> 16, no dropout) on frozen c544, AdamW 1e-3, batch 2048, fixed step count."""
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.LayerNorm(c.shape[1]), torch.nn.Linear(c.shape[1], 256), torch.nn.ReLU(),
                              torch.nn.Linear(256, C)).to(c.device)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=0); g = torch.Generator(device=c.device).manual_seed(seed)
    with torch.enable_grad():
        done = 0
        while done < steps:
            perm = torch.randperm(len(c), device=c.device, generator=g)
            for s in range(0, len(c), 2048):
                if done == steps: break
                b = perm[s:s + 2048]; loss = F.cross_entropy(net(c[b].float()), y[b])
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); done += 1
    net.eval()
    return torch.no_grad()(lambda f: torch.cat([net(f[s:s + 16384].float()) for s in range(0, len(f), 16384)]))


def oracle_bias(z_old, y_old, b0):
    """Coordinate ascent of a 16-d additive logit bias on `old` macro F1 (3 passes, +-4 grid); keeps b0 unless strictly better."""
    b = b0.clone(); best = float(f1(cm(y_old, (z_old + b).argmax(1)))); grid = GRID.to(z_old.device)
    for _ in range(3):
        for c in range(C):
            cand = b[None].repeat(len(grid), 1); cand[:, c] += grid
            scores = f1(cm(y_old, (z_old[None] + cand[:, None]).argmax(2)))
            i = int(scores.argmax())
            if float(scores[i]) > best + 1e-9: best, b = float(scores[i]), cand[i]
    return b, best


@torch.no_grad()
def knn(ref_x, ref_y, q, k=10):
    """Labels of the k nearest reference rows (squared Euclidean, fp32), chunked."""
    rn = (ref_x * ref_x).sum(1); out = []
    for a in range(0, len(q), 2048):
        qq = q[a:a + 2048]; bd = torch.full((len(qq), k), float('inf'), device=q.device)
        bi = torch.zeros(len(qq), k, dtype=torch.long, device=q.device)
        for s in range(0, len(ref_x), 131072):
            d = rn[s:s + 131072][None] - 2 * qq @ ref_x[s:s + 131072].T
            dd, ii = d.topk(min(k, d.shape[1]), dim=1, largest=False)
            bd, sel = torch.cat([bd, dd], 1).topk(k, dim=1, largest=False)
            bi = torch.cat([bi, ii + s], 1).gather(1, sel)
        out.append(ref_y[bi])
    return torch.cat(out)


def knn_scores(labels, y, log_w):
    """1-NN and 10-NN vote (raw, and votes reweighted by exp(log_w) per class) macro F1."""
    votes = F.one_hot(labels, C).float().sum(1)
    return dict(nn1=float(f1(cm(y, labels[:, 0]))), nn10=float(f1(cm(y, votes.argmax(1)))),
                nn10_prior=float(f1(cm(y, (votes * log_w.exp()).argmax(1)))))


@torch.no_grad()
def duplicates(R, log):
    """Exact duplicates of masked fp16 train rows (two independent integer hashes of the 16-bit patterns)."""
    dev = R.dev; g = torch.Generator(device='cpu').manual_seed(140)
    M1, M2 = (torch.randint(1, 2 ** 31 - 1, (P.D,), generator=g).to(dev) for _ in range(2))
    def keys(x):
        x = x.masked_fill(R.zero_cols, 0) + 0   # + 0 turns -0.0 into +0.0
        xi = x.contiguous().view(torch.int16).long() & 0xFFFF
        return (xi * M1).sum(1), (xi * M2).sum(1)
    h1, h2, ys, cl = [], [], [], []
    for k in range(R.K):
        for s in range(0, R.n[k], 1 << 20):
            a, b = keys(R.X[k][s:s + (1 << 20)]); h1.append(a); h2.append(b)
        ys.append(R.Y[k]); cl.append(torch.full((R.n[k],), k, device=dev))
    h1, h2, y, cl = torch.cat(h1), torch.cat(h2), torch.cat(ys), torch.cat(cl)
    u1, inv = torch.unique(h1, return_inverse=True); U = len(u1)
    collision = dict(unique_h1=U, unique_h2=len(torch.unique(h2)), unique_pair=len(torch.unique(h1 * 31 + h2)))
    del h2
    count = torch.bincount(inv, minlength=U)
    up, pinv, pcnt = torch.unique(inv * C + y, return_inverse=True, return_counts=True)
    pvec = up // C; labels = torch.bincount(pvec, minlength=U)
    major = torch.zeros(U, dtype=pcnt.dtype, device=dev).scatter_reduce(0, pvec, pcnt, 'amax')
    amb = labels[inv] > 1; unreach = pcnt[pinv] < major[inv]   # label not (one of) the majority labels of its vector
    # conflicts every client sees vs conflicts only visible across clients
    vc, vcinv = torch.unique(inv * 32 + cl, return_inverse=True)
    vcl = torch.unique(inv * (32 * C) + cl * C + y)
    labels_vc = torch.bincount(torch.searchsorted(vc, vcl // C), minlength=len(vc))
    local_conflict = torch.zeros(U, dtype=torch.bool, device=dev); local_conflict[(vc // 32)[labels_vc > 1]] = True
    per_class = lambda m: torch.bincount(y[m], minlength=C).tolist()
    out = dict(rows=len(y), vectors=U, collision=collision, rows_duplicated=int((count[inv] > 1).sum()), rows_ambiguous=int(amb.sum()),
               rows_unreachable=int(unreach.sum()), rows_cross_client_only=int((amb & ~local_conflict[inv]).sum()),
               class_rows=per_class(torch.ones_like(amb)), class_ambiguous=per_class(amb), class_unreachable=per_class(unreach),
               class_cross_client_only=per_class(amb & ~local_conflict[inv]))
    # group holdout rows whose exact vector occurs in train: found, and label differs from every train majority label
    gx, gy = R.H['group']; ga, _ = keys(gx); pos = torch.searchsorted(u1, ga).clamp_max(U - 1); found = u1[pos] == ga
    vid = pos[found]; key = vid * C + gy[found]; at = torch.searchsorted(up, key).clamp_max(len(up) - 1)
    has = up[at] == key; own_cnt = torch.where(has, pcnt[at], torch.zeros_like(key))
    out.update(group_found=torch.bincount(gy[found], minlength=C).tolist(),
               group_found_minority=torch.bincount(gy[found][own_cnt < major[vid]], minlength=C).tolist(),
               group_rows=torch.bincount(gy, minlength=C).tolist())
    log('T5 duplicates', {k: v for k, v in out.items() if not k.startswith(('class_', 'group_'))})
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--weights', required=True); p.add_argument('--out', required=True)
    p.add_argument('--smoke', action='store_true'); a = p.parse_args()
    t0 = time.time(); log = lambda *v: print(f'[{time.time() - t0:7.0f}s]', *v, flush=True)
    Z = SIZES['smoke' if a.smoke else 'full']; fd.deterministic()
    R = fd.Run(SimpleNamespace(arm='fd_dml_m4b', seed=42, rounds=50, batch=512, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=256,
                               aux_weight=1., backend='eager', smoke=a.smoke, wandb_online=False, workers=1, rank=0))
    dev, K = R.dev, R.K; log('setup rows', sum(R.n), 'per client', R.n)
    result = dict(smoke=a.smoke, sizes=Z, n=R.n, counts=R.counts.long().tolist(), log_pg=R.log_pg.tolist(), sets={})
    arrays = {}
    result['T5_duplicates'] = duplicates(R, log); torch.cuda.empty_cache()

    # fixed row samples (unsorted random permutations; seeds independent of the weights)
    g = torch.Generator(device='cpu').manual_seed(14)
    perm = [torch.randperm(R.n[k], generator=g).to(dev) for k in range(K)]
    idx = [pm[:Z['S']] for pm in perm]
    XS = torch.cat([R.mask(R.X[k][i].float()) for k, i in enumerate(idx)]); YS = torch.cat([R.Y[k][i] for k, i in enumerate(idx)])
    src = torch.cat([torch.full((len(i),), k, device=dev) for k, i in enumerate(idx)])
    Xg, yg = R.mask(R.H['group'][0].float()), R.H['group'][1]; Xo, yo = R.mask(R.H['old'][0].float()), R.H['old'][1]
    q_pool = torch.bincount(YS, minlength=C).double(); log_qpool = ((q_pool + 1) / (q_pool + 1).sum()).log().float()
    gp = [R.log_pg - R.log_pi[k] for k in range(K)]; pool_bias = R.log_pg - log_qpool

    # T5 kNN: query = random group rows; own (min(own, n_k) rows of k) vs pooled (same total, equal per client) vs 10x pooled
    gq = torch.randperm(len(yg), generator=g)[:Z['Q']].to(dev); Xq, yq = Xg[gq], yg[gq]
    pool_idx = [pm[:Z['pooled']] for pm in perm]; big_idx = [pm[:Z['big']] for pm in perm]
    rows = lambda ix: (torch.cat([R.mask(R.X[k][i].float()) for k, i in enumerate(ix)]), torch.cat([R.Y[k][i] for k, i in enumerate(ix)]))
    px, py = rows(pool_idx); pq = torch.bincount(py, minlength=C).double()
    tk = dict(pooled=knn_scores(knn(px, py, Xq), yq, R.log_pg - ((pq + 1) / (pq + 1).sum()).log().float()))
    del px; bx, by = rows(big_idx); bq = torch.bincount(by, minlength=C).double()
    tk['pooled_10x'] = knn_scores(knn(bx, by, Xq), yq, R.log_pg - ((bq + 1) / (bq + 1).sum()).log().float()); del bx
    tk['own'] = []
    for k in range(K):
        ox, oy = rows([perm[k][:Z['own']] if j == k else perm[j][:0] for j in range(K)])
        tk['own'].append(knn_scores(knn(ox, oy, Xq), yq, gp[k]))
    tk['own_mean'] = {m: float(np.mean([v[m] for v in tk['own']])) for m in tk['own'][0]}
    result['T5_knn'] = tk; log('T5 kNN pooled', tk['pooled'], '10x', tk['pooled_10x'], 'own mean', tk['own_mean'])
    torch.cuda.empty_cache()

    steps = Z['mlp_epochs'] * -(-len(YS) // 2048)
    expected = json.loads((Path(a.weights) / 'expected.json').read_text())
    for name, exp in expected.items():
        privates, proxy = load(Path(a.weights) / f'{name}.pt', dev); S = {}
        # proxy reference
        zpg, zpo = forward(proxy, Xg), forward(proxy, Xo); zps, hps, cps = forward(proxy, XS, True)
        _, hpg, cpg = forward(proxy, Xg, True)
        ppg, pps = zpg.argmax(1), zps.argmax(1)
        lin = fit_linear(hps, YS); mlp = fit_mlp(cps, YS, steps, 0)
        bpx, _ = oracle_bias(zpo, yo, torch.zeros(C, device=dev))
        S['proxy'] = dict(group=float(f1(cm(yg, ppg))), refit_linear_pool=float(f1(cm(yg, (lin(hpg) + pool_bias).argmax(1)))),
                          refit_mlp_pool=float(f1(cm(yg, (mlp(cpg) + pool_bias).argmax(1)))), oracle_bias=float(f1(cm(yg, (zpg + bpx).argmax(1)))))
        arrays[f'{name}/proxy_group_cm'] = cm(yg, ppg).cpu().numpy()
        arrays[f'{name}/proxy_source_cm'] = torch.stack([cm(YS[src == j], pps[src == j]) for j in range(K)]).cpu().numpy()
        log(name, 'proxy', S['proxy']); del hps, cps, hpg, cpg
        per = {key: [] for key in ('gp', 'head', 'oracle_bias', 'refit_linear_own', 'refit_linear_pool', 'refit_mlp_own', 'refit_mlp_pool',
                                   'agree_own', 'agree_other', 'agree_group', 'kl_own', 'kl_other', 'kl_group')}
        src_cm = np.zeros((2, K, K, C, C), np.int64); grp_cm = np.zeros((2, K, C, C), np.int64)
        joint = np.zeros((K, C, 4), np.int64); lost = np.zeros((C, C), np.int64); won = np.zeros((C, C), np.int64)
        lp_s, lp_g = F.log_softmax(zps, 1), F.log_softmax(zpg, 1)
        for k, model in enumerate(privates):
            zg, zo = forward(model, Xg), forward(model, Xo); zs, hs, cs = forward(model, XS, True); _, hg, cg = forward(model, Xg, True)
            pg_gp, pg_head = (zg + gp[k]).argmax(1), zg.argmax(1); ps_gp, ps_head = (zs + gp[k]).argmax(1), zs.argmax(1)
            grp_cm[0, k], grp_cm[1, k] = cm(yg, pg_gp).cpu().numpy(), cm(yg, pg_head).cpu().numpy()
            per['gp'].append(float(metrics_from_confusion(grp_cm[0, k])['f1_macro'])); per['head'].append(float(f1(cm(yg, pg_head))))
            for j in range(K):
                m = src == j; src_cm[0, k, j], src_cm[1, k, j] = cm(YS[m], ps_gp[m]).cpu().numpy(), cm(YS[m], ps_head[m]).cpu().numpy()
            own = src == k; agree = (ps_gp == pps).float(); kl = (lp_s.exp() * (lp_s - F.log_softmax(zs, 1))).sum(1)
            per['agree_own'].append(float(agree[own].mean())); per['agree_other'].append(float(agree[~own].mean()))
            per['agree_group'].append(float((pg_gp == ppg).float().mean()))
            per['kl_own'].append(float(kl[own].mean())); per['kl_other'].append(float(kl[~own].mean()))
            per['kl_group'].append(float((lp_g.exp() * (lp_g - F.log_softmax(zg, 1))).sum(1).mean()))
            ok_p, ok_q = ppg == yg, pg_gp == yg
            joint[k] = torch.stack([torch.bincount(yg[ok_p & ok_q], minlength=C), torch.bincount(yg[ok_p & ~ok_q], minlength=C),
                                    torch.bincount(yg[~ok_p & ok_q], minlength=C), torch.bincount(yg[~ok_p & ~ok_q], minlength=C)], 1).cpu().numpy()
            lost += cm(yg[ok_p & ~ok_q], pg_gp[ok_p & ~ok_q]).cpu().numpy(); won += cm(yg[~ok_p & ok_q], ppg[~ok_p & ok_q]).cpu().numpy()
            b, _ = oracle_bias(zo, yo, gp[k]); per['oracle_bias'].append(float(f1(cm(yg, (zg + b).argmax(1)))))
            lin = fit_linear(hs, YS); per['refit_linear_pool'].append(float(f1(cm(yg, (lin(hg) + pool_bias).argmax(1)))))
            lin = fit_linear(hs[own], YS[own]); per['refit_linear_own'].append(float(f1(cm(yg, (lin(hg) + gp[k]).argmax(1)))))
            mlp = fit_mlp(cs, YS, steps, k); per['refit_mlp_pool'].append(float(f1(cm(yg, (mlp(cg) + pool_bias).argmax(1)))))
            mlp = fit_mlp(cs[own], YS[own], steps, k); per['refit_mlp_own'].append(float(f1(cm(yg, (mlp(cg) + gp[k]).argmax(1)))))
            del hs, cs, hg, cg
            if k % 5 == 4 or a.smoke: log(name, 'client', k, {key: round(v[-1], 4) for key, v in per.items()})
        if not a.smoke: assert abs(np.mean(per['gp']) - exp) <= .002, ('gp does not reproduce the audit', name, np.mean(per['gp']), exp)
        S['private'] = {key: float(np.mean(v)) for key, v in per.items()}; S['private_worst'] = {key: float(min(v)) for key, v in per.items()}
        S['per_client'] = per; S['expected_gp'] = exp
        arrays.update({f'{name}/source_cm': src_cm, f'{name}/group_cm': grp_cm, f'{name}/joint': joint, f'{name}/lost': lost, f'{name}/won': won})
        result['sets'][name] = S; log(name, 'private mean', {key: round(v, 4) for key, v in S['private'].items()})
        del privates, proxy; torch.cuda.empty_cache()
    out = Path(a.out); np.savez_compressed(out.with_suffix('.npz'), **arrays)
    out.write_text(json.dumps(result, indent=1) + '\n'); log('done')


if __name__ == '__main__':
    main()
