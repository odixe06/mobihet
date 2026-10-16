"""AntiMem pilot: frozen v8 Gaussian/DML/Hybrid recipe + receiver-shortcut techniques, EMA deployment roles, QProxy.

Contract: docs/methods/anti-memorization/design.md. Arms `am_<base>` reproduce the v8 computation exactly; technique
arms only add an input hook. EMA roles are deployment-time averages evaluated inside every run (no training change).
QProxy arms quantize the DML proxy exchange (parameters per-row int-b, BN buffers fp16, integers raw).

    python antimem.py --arm am_dml_swap --seed 42 --wandb_online
"""
import copy
import io
import json
import os
import sys
import numpy as np
import torch
from mh import pilot as P
from mh import v2_run as driver
from mh import v8_hybrid as V8
from mh.v2_methods import calibrate
from mh.v6_run import Run as Base
from mh.metrics import METRIC_KEYS, metrics_from_confusion, per_class_from_confusion

BASES = ('gaussian', 'dml', 'hybrid')
TECHS = ('none', 'swap', 'm4b', 'swappos')
ARMS = [f'am_{b}' if t == 'none' else f'am_{b}_{t}' for b in BASES for t in TECHS] + ['am_dml_q8', 'am_dml_q4']
BLOCK = ['f_rcv_pos_noise_x', 'f_rcv_pos_noise_y', 'f_rcv_spd_noise', 'f_rcv_acl_noise', 'f_rcv_hed_noise',
         'f_rcv_noise_mag', 'f_rcv_profile_normal', 'f_rcv_profile_cautious', 'f_rcv_profile_aggressive']
POSITION = ['f_rcv_x_rel', 'f_rcv_y_rel', 'f_snd_x_rel', 'f_snd_y_rel']
SWAP_P, EMA_BETA = 0.5, 0.5 ** 0.5          # swap half of real rows; EMA half-life two rounds


def parse(arm):
    parts = arm.split('_')
    base, tech = parts[1], parts[2] if len(parts) > 2 else 'none'
    bits = int(tech[1:]) if tech.startswith('q') else None
    return base, ('none' if bits else tech), bits


def is_buffer(name):
    return 'running_' in name or 'num_batches_tracked' in name


def quantize(t, bits, generator=None):
    """Symmetric per-row (dim0) scale, fp16 scales; stochastic rounding when a generator is given."""
    levels = 2 ** (bits - 1) - 1
    rows = t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(1, -1)
    scale = (rows.abs().amax(1, keepdim=True) / levels).half().float()
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    v = rows / safe
    if generator is None:
        q = torch.round(v)
    else:
        q = torch.floor(v + torch.rand(v.shape, device=v.device, generator=generator))
    q = q.clamp(-levels, levels) * (scale > 0)
    nbytes = (q.numel() * bits + 7) // 8 + 2 * scale.numel()
    return (q * scale).reshape(t.shape), q.to(torch.int8), scale.half(), nbytes


class Method(V8.Method):
    def __init__(self, run, initialize=True):
        super().__init__(run, initialize)
        self.base, self.tech, self.bits = parse(self.arm)
        self.ema_private = None; self.ema_proxy = None; self.exact = None

    def prior(self):
        super().prior()
        if self.bits:
            self.exact = {k: v.detach().clone() for k, v in self.global_state.items()}

    def start_round(self):
        super().start_round()
        if self.bits:
            self.qsum, self.qseen = {}, set()

    def proxy_epoch(self, k, r):
        if not self.bits:
            return super().proxy_epoch(k, r)
        g = self.R.gen(r, k, 77); nbytes = 32
        for name, value in self.proxy[k].state_dict().items():
            if not value.is_floating_point():
                wire, nbytes = value.clone(), nbytes + 8 * value.numel()
                self.qsum[name] = wire if name not in self.qsum else torch.maximum(self.qsum[name], wire)
                continue
            if is_buffer(name):
                wire = value.detach().half().float(); nbytes += 2 * value.numel()
            else:
                wire, _, _, n = quantize(value.detach().float() - self.global_state[name].float(), self.bits, g)
                nbytes += n
            self.qsum[name] = wire.double() if name not in self.qsum else self.qsum[name] + wire.double()
        assert k not in self.qseen; self.qseen.add(k)
        self.R.bytes.append((r, k, 'up', 'proxy_quantized', nbytes))
        return None, self.logs[k]

    def aggregate(self, r):
        if self.bits:
            assert self.qseen == set(range(self.R.K))
            payload, nbytes, num, den = io.BytesIO(), 32, 0., 0.
            for name, total in self.qsum.items():
                if not total.is_floating_point():
                    self.exact[name] = total.clone(); new = total.clone(); nbytes += 8 * total.numel()
                elif is_buffer(name):
                    self.exact[name] = (total / self.R.K).float(); new = self.exact[name].half().float()
                    nbytes += 2 * total.numel()
                else:
                    self.exact[name] = self.exact[name] + (total / self.R.K).float()
                    decoded, q, scale, n = quantize(self.exact[name] - self.global_state[name], self.bits)
                    new = self.global_state[name] + decoded; nbytes += n
                    payload.write(q.cpu().numpy().tobytes()); payload.write(scale.cpu().numpy().tobytes())
                    num += float((self.exact[name] - new).double().pow(2).sum()); den += float(self.exact[name].double().pow(2).sum())
                assert torch.isfinite(new.float()).all(), ('quantized proxy nonfinite', name)
                self.global_state[name] = new.to(self.global_state[name].dtype)
            for k in range(self.R.K):
                self.R.bytes.append((r, k, 'down', 'proxy_quantized', nbytes))
                self.proxy[k].load_state_dict(self.global_state, strict=True)
            self.global_vector = self.layout.flatten(self.proxy[0]).clone()
            self.latest_packet = payload.getvalue()
            self.diag['q_residual_rel'] = (num / max(den, 1e-30)) ** 0.5
            self.diag['q_bytes_down'] = nbytes
            self.version += 1
        else:
            super().aggregate(r)
        self.update_ema()

    def update_ema(self):
        def blend(old, new):
            if old is None:
                return {k: v.detach().clone() for k, v in new.items()}
            return {k: (EMA_BETA * old[k] + (1 - EMA_BETA) * v.detach()) if v.is_floating_point() else v.detach().clone()
                    for k, v in new.items()}
        olds = self.ema_private or [None] * self.R.K
        self.ema_private = [blend(o, m.state_dict()) for o, m in zip(olds, self.R.priv)]
        if self.kd and not self.own:
            self.ema_proxy = blend(self.ema_proxy, self.global_state)

    def state(self):
        return dict(super().state(), ema_private=self.ema_private, ema_proxy=self.ema_proxy, exact=self.exact)


class Run(Base):
    def __init__(self, a):
        super().__init__(a)
        self.base, self.tech, self.bits = parse(a.arm)
        names = json.loads((P.ROOT / 'papers/pilot/fixture/manifest.json').read_text())['features']
        self.block = torch.tensor([names.index(n) for n in BLOCK], device=self.dev)
        zero = BLOCK + POSITION if self.tech == 'm4b' else POSITION if self.tech == 'swappos' else []
        self.zero = torch.tensor([names.index(n) for n in zero], device=self.dev, dtype=torch.long)
        self.swap = self.tech in ('swap', 'swappos')
        fx = P.ROOT / 'papers/pilot/fixture'
        self.G = []
        for k in range(self.K):
            g = np.load(fx / f'train_{k}_g.npy')[:len(self.Y[k])]
            self.G.append(torch.from_numpy(g.astype(np.int64)).to(self.dev))
        self.swap_stats = {}

    def mask(self, x):
        if len(self.zero):
            x = x.clone(); x[:, self.zero] = 0
        return x

    def transform(self, x, index, k, gen):
        if self.swap:
            n, G = len(index), self.G[k]
            apply = torch.rand(n, device=self.dev, generator=gen) < SWAP_P
            donor = torch.randint(0, len(G), (n,), device=self.dev, generator=gen)
            for _ in range(3):
                same = G[donor] == G[index]
                donor = torch.where(same, torch.randint(0, len(G), (n,), device=self.dev, generator=gen), donor)
            ok = apply & (G[donor] != G[index])
            rows = torch.nonzero(ok).squeeze(1)
            x = x.clone()
            x[rows[:, None], self.block[None, :]] = self.X[k][donor[rows]][:, self.block].float()
            s = self.swap_stats.setdefault(k, [0, 0]); s[0] += int(ok.sum()); s[1] += n
        return self.mask(x)

    @torch.no_grad()
    def logits(self, model, X):
        model.eval(); out = []
        for i in range(0, len(X), 16384):
            with torch.autocast('cuda', dtype=torch.float16):
                out.append(model(self.mask(X[i:i + 16384].float()))[0].float())
        return torch.cat(out)

    def private_epoch(self, method, k, r):
        return private_epoch(self, method, k, r)

    @torch.no_grad()
    def evaluate(self, m, out, r):
        arrays, evaluated = {}, {}
        roles = {'private': (self.priv, ('head', 'gp'))}
        if m.proxy:
            for k in range(self.K): m.prepare(k, r)
            roles['proxy'] = (m.proxy, ('head',))
        if m.ema_private is not None:
            models = []
            for k in range(self.K):
                model = copy.deepcopy(self.priv[k]); model.load_state_dict(m.ema_private[k], strict=True)
                calibrate(model, self.mask(self.X[k].float())); models.append(model)
            roles['private_ema'] = (models, ('head', 'gp'))
        if m.ema_proxy is not None:
            models = []
            for k in range(self.K):
                model = copy.deepcopy(m.proxy[k]); model.load_state_dict(m.ema_proxy, strict=True)
                calibrate(model, self.mask(self.X[k].float())); models.append(model)
            roles['proxy_ema'] = (models, ('head',))
        for role, (models, rules) in roles.items():
            results = {}
            for hold, (X, y) in self.H.items():
                arrays[hold + '_y'] = y.cpu().numpy().astype('uint8'); result = {}
                cms = {rule: [] for rule in rules}; preds = {rule: [] for rule in rules}
                for k, model in enumerate(models):
                    z = self.logits(model, X); assert torch.isfinite(z).all()
                    for rule in rules:
                        bias = self.log_pg - self.log_pi[k] if rule == 'gp' else 0.
                        pred = (z + bias).argmax(1)
                        cms[rule].append(torch.bincount(y * P.C + pred, minlength=P.C ** 2).view(P.C, P.C).cpu().numpy())
                        preds[rule].append(pred.cpu().numpy().astype('uint8'))
                for rule, cc in cms.items():
                    mm = [metrics_from_confusion(c) for c in cc]
                    pc = np.mean([[x['f1'] for x in per_class_from_confusion(c, P.CLASSES)] for c in cc], axis=0)
                    result[rule] = dict(mean_metrics={key: float(np.mean([v[key] for v in mm])) for key in METRIC_KEYS},
                                        f1_macro=float(np.mean([v['f1_macro'] for v in mm])), benign_f1=float(pc[0]),
                                        f1_per_class=pc.tolist(), confusion_clients=np.stack(cc).tolist())
                    arrays[f'{role}_{hold}_{rule}'] = np.stack(preds[rule])
                results[hold] = result
            evaluated[role] = results
        m.diag['swap_fraction'] = {k: v[0] / max(v[1], 1) for k, v in self.swap_stats.items()}
        pf = out / 'predictions' / f'round_{r:03d}.npz'; pf.parent.mkdir(exist_ok=True)
        tmp = pf.with_suffix('.tmp')
        with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
        os.replace(tmp, pf)
        return evaluated


def private_epoch(run, method, k, r):
    """mh.v8_hybrid.private_epoch with one input hook (`run.transform`/`run.mask`); identity for am_<base>."""
    import torch.nn.functional as F
    hooked = run.swap or len(run.zero) > 0
    swap_gen = run.gen(r, k, 55) if run.swap else None   # separate stream: order/synthetic draws stay paired
    private, proxy = run.priv[k], method.proxy[k]
    private.train(); proxy.train()
    optim = torch.optim.AdamW(list(private.parameters()) + list(proxy.parameters()),
                              lr=run.lr(r), weight_decay=0.)
    scaler = torch.amp.GradScaler()
    generator = run.gen(r, k, 0)
    torch.manual_seed(int(np.random.SeedSequence([run.a.seed, r, k, 7]).generate_state(1)[0]))
    x, y = run.X[k], run.Y[k]
    order = torch.randperm(len(y), device=run.dev, generator=generator)
    totals = torch.zeros(4, device=run.dev)
    applied = skipped = 0
    for start in range(0, len(y), run.a.batch):
        index = order[start:start + run.a.batch]
        real_x, real_y = x[index].float(), y[index]
        if hooked:
            real_x = run.transform(real_x, index, k, swap_gen)
        labels = torch.arange(run.a.aux_batch, device=run.dev) % P.C
        synthetic = method.draw(labels, generator, k) if method.aux else torch.randn(
            len(labels), P.D, device=run.dev, generator=generator)
        if hooked:
            synthetic = run.mask(synthetic)
        mixed = torch.cat([real_x, synthetic]) if method.aux else real_x
        with torch.autocast('cuda', dtype=torch.float16):
            zs, zr = private(mixed)[0].float(), proxy(mixed)[0].float()
            ce_s = F.cross_entropy(zs[:len(index)], real_y)
            ce_r = F.cross_entropy(zr[:len(index)], real_y)
            kd_s = kd_r = ce_s.detach() * 0
            if method.kd:
                ls, lr = F.log_softmax(zs[:len(index)], 1), F.log_softmax(zr[:len(index)], 1)
                weight = 1 / (ce_s + ce_r).detach().clamp_min(1e-6)
                kd_s = weight * F.kl_div(ls, lr.detach(), log_target=True, reduction='batchmean')
                kd_r = weight * F.kl_div(lr, ls.detach(), log_target=True, reduction='batchmean')
            aux_s = aux_r = ce_s.detach() * 0
            if method.aux:
                aux_s = F.cross_entropy(zs[len(index):], labels)
                aux_r = F.cross_entropy(zr[len(index):], labels)
            else:
                private.eval(); proxy.eval()
                dummy = (private(synthetic)[0].float().sum() + proxy(synthetic)[0].float().sum()) * 0
                private.train(); proxy.train()
            extra_s, extra_r = kd_s + run.a.aux_weight * aux_s, kd_r + run.a.aux_weight * aux_r
            loss = ce_s + ce_r + extra_s + extra_r
            if not method.aux:
                loss = loss + dummy
        assert torch.isfinite(loss), ('hybrid loss', k, r)
        scaler.scale(loss).backward(); scaler.unscale_(optim)
        for model in (private, proxy):
            torch.nn.utils.clip_grad_norm_(model.parameters(), run.a.clip)
        before = scaler.get_scale(); scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
        if scaler.get_scale() < before:
            skipped += 1
        else:
            applied += 1
            totals += torch.stack([ce_s.detach(), extra_s.detach(), ce_r.detach(), extra_r.detach()])
    assert applied > 0 and skipped <= 16
    V8.finite(private); V8.finite(proxy)
    ce_s, extra_s, ce_r, extra_r = (totals / applied).tolist()
    common = dict(client=k, rows=len(y), applied=applied, skipped=skipped)
    method.logs[k] = dict(common, loss=ce_r + extra_r, positive_ce=ce_r, negative_ce=extra_r)
    return dict(common, loss=ce_s + extra_s, real_ce=ce_s, auxiliary=extra_s)


if __name__ == '__main__':
    driver.ARMS, driver.Method, driver.Run = ARMS, Method, Run
    if '--out' not in sys.argv:
        sys.argv += ['--out', str(P.ROOT / 'papers/antimem/runs')]
    if '--wandb_project' not in sys.argv:
        sys.argv += ['--wandb_project', 'mobihet-antimem']
    driver.main()
