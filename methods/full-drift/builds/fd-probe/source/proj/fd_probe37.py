"""P37: how the synthetic rows change the private and proxy weights online (full model: backbone, 31 BatchNorm layers, head).
Production Run/Method/private_epoch of source r31 on the pilot fixture from an audited xa checkpoint; arms only flip flags or patch
the synthetic draw / the private's forward on the synthetic rows. Protocol: docs/research/methods/synthetic-weight-dynamics-2026-10-23.md.
No official test.

    python fd_probe37.py --weights <xa round_006.pt> --fixture <pilot fixture> --out <json> [--clients 0,1] [--passes 4]
"""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
import fd
from mh.metrics import METRIC_KEYS, metrics_from_confusion

ARM = 'fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl'
ARMS = ['none', 'gauss', 'gauss_b', 'gauss_bn', 'gauss_sep', 'gauss_head', 'cop', 'real', 'real_sep', 'real_head']
GROUPS = ['stems', 'dense', 'google', 'alex', 'squeeze', 'bn_affine', 'bn_stats', 'head.0', 'head.2', 'head.5']
PAIRS = [('gauss', 'gauss_b'), ('gauss', 'real'), ('cop', 'real'), ('gauss_sep', 'real_sep'), ('gauss_head', 'real_head'), ('gauss', 'gauss_bn'),
         ('gauss_sep', 'gauss'), ('gauss_head', 'gauss')]
GRAD_BATCHES = 8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def groups_of(model):
    bn = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            for leaf in ('weight', 'bias'): bn[f'{name}.{leaf}'] = 'bn_affine'
            for leaf in ('running_mean', 'running_var'): bn[f'{name}.{leaf}'] = 'bn_stats'
    out = {}
    for name, value in model.state_dict().items():
        if not value.is_floating_point(): continue
        out[name] = bn.get(name) or next((g for g in ('head.0', 'head.2', 'head.5') if name.startswith(g + '.')), name.split('.')[0])
    assert set(out.values()) <= set(GROUPS), set(out.values())
    return out


def flat(state, names):
    return {g: torch.cat([state[n].float().flatten() for n in ns]) for g, ns in names.items()}


def report(y, pred):
    cm = torch.bincount(y * 16 + pred, minlength=256).view(16, 16).cpu().numpy()
    m = metrics_from_confusion(cm)
    tp = np.diag(cm).astype(float); f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
    return dict({k: float(m[k]) for k in METRIC_KEYS}, per_class=f1.tolist(), cm=cm.tolist())


class Probe:
    def __init__(self, a, dev):
        self.a, self.dev = a, dev
        fd.fixture_path = lambda args: Path(a.fixture)
        fd.check_full_manifest = lambda *v: None   # pilot fixture (2.5 %), not the full holdout manifest
        args = SimpleNamespace(arm=ARM, seed=42, rounds=50, batch=512, lr=1e-3, lr_min=1e-5, clip=1., aux_batch=256, aux_weight=1.,
                               backend='eager', smoke=False, wandb_online=False, fast=True, workers=1, rank=0)
        load = np.load   # the pilot fixture has no official test: Run's 'test' slot reads 'old' (as --smoke); never scored here
        np.load = lambda f, *v, **kw: load(Path(str(f).replace('/test_', '/old_')), *v, **kw)
        try: self.R = R = fd.Run(args)
        finally: np.load = load
        self.m = m = fd.Method(R); m.prior()
        R.a.arm = ARM + '_cop'; self.m_cop = fd.Method(R); R.a.arm = ARM   # copula tables (round 18) from the same rows
        assert self.m_cop.cop and not m.cop
        self.ck = torch.load(a.weights, map_location=dev, weights_only=True)
        self.round = int(self.ck['round']) + 1   # the next local epoch (LR of that round)
        self.base_gen = R.gen; self.pass_index = 0
        R.gen = lambda *key: self.base_gen(*key, 1000 + self.pass_index)   # a fresh row order / draw stream per pass
        self.names = {}
        for n, g in groups_of(R.init).items(): self.names.setdefault(g, []).append(n)
        self.bn = [mod for mod in R.work_private.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
        self.mode = None; self.split = None
        inner = R.forward
        def forward(model, x):
            if self.mode in ('sep', 'head') and model is R.work_private and model.training:   # always real + aux_batch synthetic rows
                n = len(x) - R.a.aux_batch
                zr, hr = inner(model, x[:n])   # real rows: batch statistics of real rows only
                for mod in self.bn: mod.eval()   # synthetic rows: running statistics, no update
                try:
                    if self.mode == 'sep': zs = inner(model, x[n:])[0]
                    else:
                        with torch.no_grad(): c = model.combined(x[n:])
                        zs = model.head(c)
                finally:
                    for mod in self.bn: mod.train()
                return torch.cat([zr, zs]), None
            return inner(model, x)
        R.forward = forward
        # real oracle: per class, rows of the 19 other clients (diagnostic only, R3-ineligible)
        self.Xall = torch.cat([R.X[k] for k in range(R.K)]); self.Yall = torch.cat(R.Y)
        self.Call = torch.cat([torch.full((R.n[k],), k, device=dev) for k in range(R.K)])
        self.pool = {}

    def real_draw(self, labels, g, k):
        if k not in self.pool:
            self.pool = {k: [torch.nonzero((self.Yall == c) & (self.Call != k)).squeeze(1) for c in range(16)]}
        idx = torch.empty(len(labels), dtype=torch.long, device=self.dev)
        for c in range(16):
            sel = torch.nonzero(labels == c).squeeze(1); cand = self.pool[k][c]
            if len(sel): idx[sel] = cand[torch.randint(len(cand), (len(sel),), device=self.dev, generator=g)]
        return self.Xall[idx].float()

    def reset(self, k):
        R, m = self.R, self.m
        R.priv[k].load_state_dict(self.ck['private'][k], strict=True)
        m.global_state = {n: v.clone() for n, v in self.ck['proxy'][0].items()}
        m.proxy[k].load_state_dict(self.ck['proxy'][0], strict=True); m.version = self.round - 1
        m.start_round()

    def run_arm(self, arm, k):
        R, m = self.R, self.m
        self.reset(k)
        saved = dict(syn=R.syn_teacher, paux=(R.paux, m.paux), gk2=R.gk2, GK2=fd.GK2, draw=m.draw, cop=m.cop)
        try:
            if arm == 'none': R.syn_teacher = None; R.paux = m.paux = False
            if arm == 'gauss_bn': R.gk2 = True; fd.GK2 = 0.
            if arm == 'cop': m.cop = True; m.copula = self.m_cop.copula
            if arm.startswith('real'): m.draw = self.real_draw
            if arm == 'gauss_b':   # noise floor: same real rows, another Gaussian draw stream
                draw = saved['draw']; m.draw = lambda labels, g, kk: draw(labels, self.alt, kk)
            self.mode = 'sep' if arm.endswith('_sep') else 'head' if arm.endswith('_head') else None
            logs = []
            for p in range(self.a.passes):
                self.pass_index = p; self.alt = self.base_gen(self.round, k, 4242, p); m.prepare(k, self.round)   # round start: proxy (and the synthetic teacher) = the snapshot
                logs.append(R.private_epoch(m, k, self.round))
        finally:
            R.syn_teacher = saved['syn']; R.paux, m.paux = saved['paux']; R.gk2 = saved['gk2']; fd.GK2 = saved['GK2']
            m.draw = saved['draw']; m.cop = saved['cop']; self.mode = None
        return ({n: v.clone() for n, v in R.priv[k].state_dict().items()},
                {n: v.clone() for n, v in m.proxy[k].state_dict().items()}, logs)

    @torch.no_grad()
    def evaluate(self, state, holds=('group', 'old')):
        model = self.R.eval_template; model.load_state_dict(state, strict=True); model.eval()
        out = {}
        for h in holds:
            x, y = self.R.H[h]; z = self.R.logits(model, x); out[h] = report(y, z.argmax(1))
        return out

    @torch.no_grad()
    def features(self, state, x):
        model = self.R.eval_template; model.load_state_dict(state, strict=True); model.eval()
        with torch.autocast('cuda', dtype=torch.float16): return model.combined(self.R.mask(x)).float()

    def gradients(self, k):
        """Initial state, fp32, mixed-BN forward as production: per-term private gradients by group (the adaptive KD weight applied)."""
        R, m = self.R, self.m
        self.reset(k)
        private = copy.deepcopy(R.priv[k]).train()
        for mod in private.modules():
            if isinstance(mod, torch.nn.Dropout): mod.eval()   # gradient differences come from BN batch statistics only
        proxy = copy.deepcopy(m.proxy[k]).eval()
        teacher = copy.deepcopy(R.priv[k]); teacher.load_state_dict(m.global_state, strict=True); teacher.eval()
        params = [(n, p) for n, p in private.named_parameters()]
        group = groups_of(private)
        g = self.base_gen(self.round, k, 9037); order = torch.randperm(R.n[k], device=self.dev, generator=g)
        labels = R.aux_labels[k]; acc = {}
        def grad(loss):
            gs = torch.autograd.grad(loss, [p for _, p in params], retain_graph=True)
            return {n: v.detach() for (n, _), v in zip(params, gs)}
        def add(key, gd):
            for n, v in gd.items(): acc.setdefault(key, {}).setdefault(n, torch.zeros_like(v)).add_(v)
        for b in range(GRAD_BATCHES):
            i = order[b * 512:(b + 1) * 512]; x, y = R.mask(R.X[k][i].float()), R.Y[k][i]
            syn = R.mask(m.draw(labels, g, k)); real = R.mask(self.real_draw(labels, g, k))
            with torch.no_grad():
                zt, zts, ztr = teacher(x)[0], teacher(syn)[0], teacher(real)[0]
                proxy_ce = F.cross_entropy(proxy(x)[0], y)
            z = private(torch.cat([x, syn]))[0]; zr = z[:len(y)]
            ce = F.cross_entropy(zr + R.la_shift[k], y); w = 1 / (ce + proxy_ce).detach().clamp_min(1e-6)
            kl = lambda zs, zt: F.kl_div(F.log_softmax(zs, 1), F.log_softmax(zt, 1), log_target=True, reduction='batchmean')
            add('ce', grad(ce)); add('kd', grad(w * kl(zr, zt))); add('syn', grad(w * kl(z[len(y):], zts)))
            z2 = private(torch.cat([x, real]))[0]
            add('real', grad(w * kl(z2[len(y):], ztr)))
            z3 = private(x)[0]
            add('ce_realbn', grad(F.cross_entropy(z3 + R.la_shift[k], y)))
        names = {}
        for n, gname in group.items():
            if n in acc['ce']: names.setdefault(gname, []).append(n)
        out = {}
        for gname, ns in list(names.items()) + [('all', [n for n, _ in params])]:
            v = {key: torch.cat([acc[key][n].flatten() for n in ns]) for key in acc}
            cos = lambda a, b: float(F.cosine_similarity(v[a], v[b], 0))
            out[gname] = dict({f'norm_{key}': float(t.norm()) for key, t in v.items()},
                              cos_syn_ce=cos('syn', 'ce'), cos_syn_kd=cos('syn', 'kd'), cos_syn_real=cos('syn', 'real'),
                              cos_real_ce=cos('real', 'ce'), cos_ce_bn=cos('ce', 'ce_realbn'),
                              rel_ce_bn=float((v['ce'] - v['ce_realbn']).norm() / v['ce_realbn'].norm().clamp_min(1e-12)))
        return out


def summary(clients, key):
    v = np.array([c[key] for c in clients]); return dict(mean=float(v.mean()), min=float(v.min()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True); p.add_argument('--fixture', required=True); p.add_argument('--out', required=True)
    p.add_argument('--clients', default=','.join(map(str, range(20)))); p.add_argument('--passes', type=int, default=4)
    a = p.parse_args(); t0 = time.monotonic(); log = lambda *v: print(round(time.monotonic() - t0, 1), *v, flush=True)
    fd.deterministic(); dev = torch.device('cuda')
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected
    pr = Probe(a, dev); R, m = pr.R, pr.m
    names = pr.names; clients = [int(c) for c in a.clients.split(',')]
    gx = R.H['group'][0][:4096]
    proxy_sum = {arm: None for arm in ARMS}; per = []
    # sanity: in the sep/head forward the real rows of a mixed batch get exactly the plain real-only forward
    pr.reset(0); R.work_private.load_state_dict(R.priv[0].state_dict()); R.work_private.train()
    xb = R.mask(R.X[0][:512].float()); sb = R.mask(m.draw(R.aux_labels[0], pr.base_gen(0, 0, 1), 0))
    with torch.autocast('cuda', dtype=torch.float16):
        torch.manual_seed(1); base = R.forward(R.work_private, xb)[0]
        for mode in ('sep', 'head'):
            pr.mode = mode; torch.manual_seed(1); mixed = R.forward(R.work_private, torch.cat([xb, sb]))[0]
            assert torch.equal(mixed[:512], base) and mixed.shape[0] == 768, mode
    pr.mode = None
    for k in clients:
        start = {n: v.clone() for n, v in pr.ck['private'][k].items()}
        states, proxies, logs = {}, {}, {}
        for arm in ARMS:
            states[arm], proxies[arm], logs[arm] = pr.run_arm(arm, k)
            for s in (states[arm], proxies[arm]):
                assert all(torch.isfinite(v).all() for v in s.values() if v.is_floating_point()), (arm, k)
            ps = proxies[arm]
            proxy_sum[arm] = {n: v.double().clone() if v.is_floating_point() else v.clone() for n, v in ps.items()} if proxy_sum[arm] is None else \
                {n: proxy_sum[arm][n] + v.double() if v.is_floating_point() else torch.maximum(proxy_sum[arm][n], v) for n, v in ps.items()}
            log(k, arm, 'trained')
        rec = dict(client=k, eval={}, swap={}, weights={}, proxy_weights={}, cos={}, proxy_cos={}, feature_shift={},
                   syn_accept={arm: [l.get('syn_accept') for l in logs[arm]] for arm in ARMS})
        f0, fn = flat(start, names), flat(states['none'], names)
        p0 = flat(pr.ck['proxy'][0], names); pn = flat(proxies['none'], names)
        feat_none = pr.features(states['none'], gx)
        for arm in ARMS:
            rec['eval'][arm] = pr.evaluate(states[arm])
            fa, pa = flat(states[arm], names), flat(proxies[arm], names)
            rec['weights'][arm] = {g: dict(update=float((fa[g] - f0[g]).norm() / f0[g].norm().clamp_min(1e-12)),
                                           vs_none=float((fa[g] - fn[g]).norm() / (fn[g] - f0[g]).norm().clamp_min(1e-12))) for g in names}
            rec['proxy_weights'][arm] = {g: float((pa[g] - pn[g]).norm() / (pn[g] - p0[g]).norm().clamp_min(1e-12)) for g in names}
            if arm != 'none':
                head = lambda s: {n: v for n, v in s.items() if n.startswith('head.')}
                rec['swap'][arm] = dict(backbone_arm=pr.evaluate({**states[arm], **head(states['none'])}, ('group',))['group']['f1_macro'],
                                        head_arm=pr.evaluate({**states['none'], **head(states[arm])}, ('group',))['group']['f1_macro'])
                fa_ = pr.features(states[arm], gx)
                rec['feature_shift'][arm] = float((fa_ - feat_none).norm() / feat_none.norm())
        for x_, y_ in PAIRS:
            dx, dy = flat(states[x_], names), flat(states[y_], names)
            qx, qy = flat(proxies[x_], names), flat(proxies[y_], names)
            rec['cos'][f'{x_}|{y_}'] = {g: float(F.cosine_similarity(dx[g] - fn[g], dy[g] - fn[g], 0)) for g in names}
            rec['proxy_cos'][f'{x_}|{y_}'] = {g: float(F.cosine_similarity(qx[g] - pn[g], qy[g] - pn[g], 0)) for g in names}
        rec['gradients'] = pr.gradients(k)
        per.append(rec); log(k, 'done', {arm: round(rec['eval'][arm]['group']['f1_macro'], 4) for arm in ARMS})
    proxy_eval = {}
    for arm in ARMS:
        s = {n: (v / len(clients)).to(pr.ck['proxy'][0][n].dtype) if v.is_floating_point() else v for n, v in proxy_sum[arm].items()}
        proxy_eval[arm] = pr.evaluate(s)
    out = dict(protocol='docs/research/methods/synthetic-weight-dynamics-2026-10-23.md', weights=str(Path(a.weights).resolve()),
               weights_sha=sha(a.weights), fixture_sha=sha(fx / 'manifest.json'), source_sha=sha(__file__), fd_sha=sha(fd.__file__),
               round=pr.round, passes=a.passes, clients=clients, arms=ARMS, groups=GROUPS, seconds=time.monotonic() - t0,
               proxy_eval=proxy_eval, per_client=per, summary={})
    for arm in ARMS:
        rows = [dict(f1=c['eval'][arm]['group']['f1_macro'], benign=c['eval'][arm]['group']['per_class'][0],
                     old=c['eval'][arm]['old']['f1_macro']) for c in per]
        out['summary'][arm] = dict(group=summary(rows, 'f1'), benign=summary(rows, 'benign'), old=summary(rows, 'old'),
                                   per_class=np.mean([c['eval'][arm]['group']['per_class'] for c in per], 0).tolist(),
                                   proxy_group=proxy_eval[arm]['group']['f1_macro'])
    Path(a.out).write_text(json.dumps(out))
    log('wrote', a.out, {arm: round(out['summary'][arm]['group']['mean'], 4) for arm in ARMS})


if __name__ == '__main__':
    main()
