"""ProxyFuse post-hoc probe: deploy private-gp and DML proxy log-prob average; no training, no extra bytes.

Contract: docs/methods/proxy-fusion/design.md. Reproduces the audited argmax of every single-model rule before
reporting fused rules; official test is never read.

    python proxyfuse.py --inputs DIR --out DIR [--seeds 42 43] [--clients N --rows N]   # last two: code test only
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

C = 16
RULES = ('private_head', 'private_gp', 'private_gp_legacy', 'proxy_head',
         'fuse_gp', 'fuse_gp_legacy', 'fuse_prob_gp')
PRIMARY, REFERENCES = 'fuse_gp', ('proxy_head', 'private_gp', 'private_gp_legacy')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def logits(model, X):
    """Same as mh.pilot.Run.logits: eval mode, 16384-row chunks, fp16 autocast, fp32 output."""
    model.eval(); out = []
    for i in range(0, len(X), 16384):
        with torch.autocast('cuda', dtype=torch.float16):
            out.append(model(X[i:i + 16384].float())[0].float())
    return torch.cat(out)


def confusion(y, pred):
    return torch.bincount(y * C + pred, minlength=C * C).view(C, C).cpu().numpy()


def summarize(cms, metrics, classes):
    mm = [metrics.metrics_from_confusion(c) for c in cms]
    pc = np.mean([[x['f1'] for x in metrics.per_class_from_confusion(c, classes)] for c in cms], axis=0)
    return dict(mean_metrics={k: float(np.mean([v[k] for v in mm])) for k in metrics.METRIC_KEYS},
                f1_macro=float(np.mean([v['f1_macro'] for v in mm])), benign_f1=float(pc[0]),
                f1_per_class=pc.tolist(), f1_macro_clients=[v['f1_macro'] for v in mm])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--inputs', required=True); p.add_argument('--out', required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43])
    p.add_argument('--clients', type=int, default=20); p.add_argument('--rows', type=int)
    a = p.parse_args()
    inp, out = Path(a.inputs), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((inp / 'proxyfuse_manifest.json').read_text())
    for rel, digest in manifest.items():
        assert sha(inp / rel) == digest, ('input sha mismatch', rel)
    sys.path.insert(0, str(inp / 'src'))
    import dagsnet, metrics
    classes = json.loads((inp / 'meta.json').read_text())['class_names']
    dev = torch.device('cuda')
    fx = inp / 'fixture'
    counts = torch.stack([torch.bincount(torch.from_numpy(np.load(fx / f'train_{k}_y.npy').astype(np.int64)),
                                         minlength=C) for k in range(20)]).double()
    log_pi_legacy = torch.log((counts + 1) / (counts + 1).sum(1, keepdim=True)).float().to(dev)
    tot = counts.sum(0)
    log_pg_fixture = torch.log((tot + 1) / (tot + 1).sum()).float().to(dev)
    holds = {}
    for h in ('group', 'old'):
        x, y = np.load(fx / f'{h}_X.npy'), np.load(fx / f'{h}_y.npy')
        if a.rows: x, y = x[:a.rows], y[:a.rows]
        holds[h] = (torch.from_numpy(x).to(dev), torch.from_numpy(y.astype(np.int64)).to(dev))
    report = dict(contract='docs/methods/proxy-fusion/design.md', inputs=manifest,
                  code_test=bool(a.rows or a.clients < 20), seeds={})
    for seed in a.seeds:
        ck = inp / f'ckpt/s{seed}'
        with torch.serialization.safe_globals([bytes]):
            w = torch.load(ck / 'weights_r020.pt', map_location='cpu', weights_only=True)
            st = torch.load(ck / 'resume_r020.pt', map_location='cpu', weights_only=True)
        saved = np.load(ck / 'predictions_r020.npz')
        assert w['round'] == st['round'] == 20 and w['fingerprint'] == st['fingerprint']
        log_pg = st['log_pg'].float().to(dev)
        assert torch.allclose(log_pg, log_pg_fixture, atol=1e-6), 'saved global prior differs from fixture counts'
        log_pi_gp = st['algorithm']['log_pi_effective'].float().to(dev)
        proxy_sd = w['proxy'][0]
        assert all(all(torch.equal(v, sd[k]) for k, v in proxy_sd.items()) for sd in w['proxy'])
        proxy = dagsnet.build_model(dict(dagsnet.CFG, n_features=66, num_classes=C), latent=True).to(dev)
        proxy.load_state_dict(proxy_sd, strict=True)
        private = dagsnet.build_model(dict(dagsnet.CFG, n_features=66, num_classes=C), latent=True).to(dev)
        res = {}
        for h, (X, y) in holds.items():
            zp = logits(proxy, X); assert torch.isfinite(zp).all()
            lp = F.log_softmax(zp, 1)
            cms = {r: [] for r in RULES}; preds = {r: [] for r in RULES}; mismatch = {}
            agree = dict(both_right=0, private_only=0, proxy_only=0, both_wrong=0)
            for k in range(a.clients):
                private.load_state_dict(w['private'][k], strict=True)
                z = logits(private, X); assert torch.isfinite(z).all()
                ls = F.log_softmax(z, 1)
                b_gp, b_legacy = log_pg - log_pi_gp[k], log_pg - log_pi_legacy[k]
                rule_pred = {
                    'private_head': z.argmax(1),
                    'private_gp': (z + b_gp).argmax(1),
                    'private_gp_legacy': (z + b_legacy).argmax(1),
                    'proxy_head': zp.argmax(1),
                    'fuse_gp': (0.5 * (ls + b_gp) + 0.5 * lp).argmax(1),
                    'fuse_gp_legacy': (0.5 * (ls + b_legacy) + 0.5 * lp).argmax(1),
                    'fuse_prob_gp': (0.5 * F.softmax(z + b_gp, 1) + 0.5 * F.softmax(zp, 1)).argmax(1)}
                for r, pr in rule_pred.items():
                    cms[r].append(confusion(y, pr)); preds[r].append(pr.cpu().numpy().astype('uint8'))
                for r, key in (('private_head', 'head'), ('private_gp', 'gp'), ('private_gp_legacy', 'gp_legacy')):
                    ref = saved[f'private_{h}_{key}'][k][:len(y)]
                    mismatch[f'{r}_{k}'] = int((preds[r][-1] != ref).sum())
                mismatch[f'proxy_head_{k}'] = int((preds['proxy_head'][-1] != saved[f'proxy_{h}_head'][k][:len(y)]).sum())
                pg, pp = rule_pred['private_gp'] == y, rule_pred['proxy_head'] == y
                agree['both_right'] += int((pg & pp).sum()); agree['private_only'] += int((pg & ~pp).sum())
                agree['proxy_only'] += int((~pg & pp).sum()); agree['both_wrong'] += int((~pg & ~pp).sum())
            res[h] = dict(rules={r: summarize(cms[r], metrics, classes) for r in RULES},
                          reproduction=dict(rows=int(len(y)) * a.clients, mismatched_rows=int(sum(mismatch.values())),
                                            max_client_mismatch=max(mismatch.values()), detail=mismatch),
                          complementarity_private_gp_vs_proxy=agree)
            np.savez_compressed(out / f'predictions_s{seed}_{h}.npz', y=y.cpu().numpy().astype('uint8'),
                                **{r: np.stack(v) for r, v in preds.items()})
        g = res['group']['rules']; best = max(REFERENCES, key=lambda r: g[r]['f1_macro'])
        gate = dict(primary=PRIMARY, strongest_reference=best,
                    delta_f1=g[PRIMARY]['f1_macro'] - g[best]['f1_macro'],
                    benign_delta=g[PRIMARY]['benign_f1'] - max(g[r]['benign_f1'] for r in REFERENCES),
                    reproduction_rows=res['group']['reproduction']['mismatched_rows'] + res['old']['reproduction']['mismatched_rows'])
        rows = res['group']['reproduction']['rows'] + res['old']['reproduction']['rows']
        gate['reproduction_pass'] = gate['reproduction_rows'] <= 1e-4 * rows
        gate['pass'] = gate['reproduction_pass'] and gate['delta_f1'] >= 0.010 and gate['benign_delta'] >= -0.005
        res['gate'] = gate; report['seeds'][seed] = res
        print('seed', seed, json.dumps(gate), flush=True)
        for h in ('group', 'old'):
            print(' ', h, {r: round(res[h]['rules'][r]['f1_macro'], 6) for r in RULES},
                  'mismatch', res[h]['reproduction']['mismatched_rows'], flush=True)
    report['pass_all_seeds'] = all(v['gate']['pass'] for v in report['seeds'].values())
    (out / 'proxyfuse_results.json').write_text(json.dumps(report, indent=1) + '\n')
    print('PROXYFUSE PASS all seeds', report['pass_all_seeds'], flush=True)


if __name__ == '__main__':
    main()
