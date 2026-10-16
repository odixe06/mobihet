"""FullDrift probe: full-data internal holdout, epochs 1-6 of the fixed cosine-T50 full recipe, seeds 42/43.

Contract: docs/methods/full-drift/design.md. `fd_dml`/`fd_gaussian` are the frozen full recipe (full_run/full_method)
on the holdout fixture; technique arms only switch one hook. Decisions use the `group` holdout; official test is
evaluated once after the last epoch, descriptive only.

    python fd.py --arm fd_dml_sub4 --seed 42 --wandb_online
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mh import pilot as P
from mh.new_fl import digest, save
from mh.v2_methods import calibrate, finite
from mh.metrics import METRIC_KEYS, metrics_from_confusion, per_class_from_confusion
from mh.model_packet import Codec
from full_method import Method as FullMethod
from full_run import Run as FullRun

ARMS = ['fd_dml', 'fd_dml_sub4', 'fd_dml_prox', 'fd_dml_avgm', 'fd_fedavg', 'fd_fedavg_sub4', 'fd_gaussian', 'fd_gaussian_swappos',
        # round 2 (owner 15/10 ~22Z): QProxy q4 with frequent aggregation, sub16, sub4 + FedProx
        'fd_dml_sub4_q4', 'fd_fedavg_sub4_q4', 'fd_dml_sub16_q4', 'fd_fedavg_sub16_q4', 'fd_dml_sub4_prox', 'fd_fedavg_sub4_prox',
        # round 3 (owner 16/10: target = client private gp; zero extra bytes/contacts, architecture-agnostic)
        'fd_dml_m4b', 'fd_dml_swappos', 'fd_dml_ntd', 'fd_dml_q4',
        # round 4 (client): hidden-state KD; Gaussian with receiver/position features zeroed (AntiMem m4b)
        'fd_dml_hkd', 'fd_gaussian_m4b',
        # round 5 (owner 16/10 ~08:20Z): m4b with QProxy q4 (R1) and with sub4 + q4; m4b:15 horizon reuses fd_dml_m4b
        'fd_dml_m4b_q4', 'fd_dml_m4b_sub4_q4']
PARTS, MU, SWAP_P, EMA_BETA, CAL_ROWS, SERVER_BETA = 4, 0.01, 0.5, 0.5 ** 0.5, 65536, 0.9
BLOCK = ['f_rcv_pos_noise_x', 'f_rcv_pos_noise_y', 'f_rcv_spd_noise', 'f_rcv_acl_noise', 'f_rcv_hed_noise',
         'f_rcv_noise_mag', 'f_rcv_profile_normal', 'f_rcv_profile_cautious', 'f_rcv_profile_aggressive']
POSITION = ['f_rcv_x_rel', 'f_rcv_y_rel', 'f_snd_x_rel', 'f_snd_y_rel']
TRAIN_ROWS, TEST_ROWS = None, 10761343   # train rows are read from the holdout manifest


def is_buffer(name):
    return 'running_' in name or 'num_batches_tracked' in name


def quantize(t, bits, generator=None):
    """antimem.quantize: symmetric per-row (dim0) scale, fp16 scales; stochastic rounding when a generator is given."""
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


def not_true(z, y):
    """FedNTD: log-softmax over the not-true classes (true-class logit pushed to -1e4, so it carries no mass)."""
    return F.log_softmax(z.masked_fill(F.one_hot(y, z.shape[1]).bool(), -1e4), 1)


def check_full_manifest(manifest, n, test_rows):
    """Production coverage gate: full holdout fixture, every non-holdout train row loaded; counts are per-class lists."""
    assert manifest['protocol'].startswith('temporal2% whole partition') and manifest['train_cap'] is None
    assert manifest['group_overlap'] == 0 and manifest['eval_cap'] == 65536
    assert n == [sum(manifest['counts'][f'train_{k}']) for k in range(len(n))]
    assert manifest['source_rows'] == 43045415 and test_rows == TEST_ROWS, 'full holdout coverage gate'
    assert manifest['test_scaler'].startswith('scaler.json')


class Method(FullMethod):
    def __init__(self, run, initialize=True):
        super().__init__(run, initialize)
        self.ema_private = None
        self.fedavg = 'fedavg' in self.arm
        self.momentum = self.arm.endswith('_avgm'); self.velocity = {}
        self.bits = 4 if 'q4' in self.arm.split('_') else None; self.exact = None
        if self.fedavg:   # one shared model trained with CE only: the full DML proxy machinery without private/KD
            self.kd = True
            torch.manual_seed(run.a.seed + 1000); initial = P.make('dagsnet').to(run.dev)
            self.proxy = [copy.deepcopy(initial) for _ in range(run.K)]
            self.global_state = {name: value.clone() for name, value in initial.state_dict().items()}
            self.codec = Codec(initial, f'full:{self.arm}:{run.a.seed}')

    def prior(self):
        super().prior()
        if self.bits:
            self.exact = {k: v.detach().clone() for k, v in self.global_state.items()}

    def start_round(self):
        super().start_round()
        if self.bits:
            self.qsum, self.qseen = {}, set()

    def proxy_epoch(self, k, r):
        """QProxy (antimem): up = per-row int-b delta vs the replica, stochastic; BN buffers fp16; integers raw."""
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

    def aggregate_quantized(self, r):
        """Server keeps the exact model (error feedback); down = int-b nearest of exact − replica."""
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
        self.global_vector = self.layout.flatten(self.proxy[0]).clone(); self.latest_packet = payload.getvalue()
        self.diag['q_residual_rel'] = (num / max(den, 1e-30)) ** 0.5; self.diag['q_bytes_down'] = nbytes
        self.version = r

    def aggregate(self, r):
        if self.bits:
            return self.aggregate_quantized(r)
        if not self.momentum:
            return super().aggregate(r)
        # FedAvgM: server velocity on the averaged parameter delta; BN buffers/counters are averaged as usual.
        state = self.aggregate_sum.finish(self.R.K); norm = 0.
        for name, value in state.items():
            if not value.is_floating_point() or 'running_' in name: continue
            delta = value.float() - self.global_state[name].float()
            self.velocity[name] = SERVER_BETA * self.velocity.get(name, torch.zeros_like(delta)) + delta
            state[name] = (self.global_state[name].float() + self.velocity[name]).to(value.dtype)
            norm += float(self.velocity[name].square().sum())
        self.diag['server_velocity_norm'] = norm ** 0.5
        packet = self.codec.encode(state, r - 1, r)
        self.global_state, _, _ = self.codec.decode(packet, self.R.dev, r - 1)
        for k in range(self.R.K):
            self.record(packet, r, k, 'down', 'proxy_state')
            self.proxy[k].load_state_dict(self.global_state, strict=True)
        self.global_vector = self.layout.flatten(self.proxy[0]).clone(); self.latest_packet = packet
        self.version = r

    def update_ema(self):
        if self.fedavg: return
        def blend(old, new):
            if old is None:
                return {k: v.detach().clone() for k, v in new.items()}
            return {k: (EMA_BETA * old[k] + (1 - EMA_BETA) * v.detach()) if v.is_floating_point() else v.detach().clone()
                    for k, v in new.items()}
        olds = self.ema_private or [None] * self.R.K
        self.ema_private = [blend(o, m.state_dict()) for o, m in zip(olds, self.R.priv)]


class Run(FullRun):
    def __init__(self, args):
        self.a, self.dev, self.K, self.bytes = args, torch.device('cuda'), 20, []
        tokens = args.arm.split('_')
        self.parts = 16 if 'sub16' in tokens else PARTS if 'sub4' in tokens else 1
        self.mu = MU if 'prox' in tokens else 0.
        self.swap = 'swappos' in tokens; self.fedavg = 'fedavg' in args.arm; self.ntd = 'ntd' in tokens
        hidden = BLOCK + POSITION if 'm4b' in tokens else POSITION if self.swap else []
        fixture = P.ROOT / 'papers/pilot/fixture'; manifest = json.loads((fixture / 'manifest.json').read_text())
        names = manifest['features']
        self.block = torch.tensor([names.index(n) for n in BLOCK], device=self.dev)
        self.zero = torch.tensor([names.index(n) for n in hidden], device=self.dev, dtype=torch.long)
        self.X, self.Y, self.G = [], [], []
        for k in range(self.K):
            x = np.load(fixture / f'train_{k}_X.npy', mmap_mode='r'); y = np.load(fixture / f'train_{k}_y.npy', mmap_mode='r')
            g = np.load(fixture / f'train_{k}_g.npy', mmap_mode='r')
            if args.smoke:
                indices = np.random.default_rng(k).choice(len(y), 1024, replace=False); x, y, g = x[indices], y[indices], g[indices]
            self.X.append(torch.from_numpy(np.array(x, copy=True)).to(self.dev))
            self.Y.append(torch.from_numpy(np.array(y, dtype='int64', copy=True)).to(self.dev))
            if self.swap:
                self.G.append(torch.from_numpy(np.array(g, dtype='int64', copy=True)).to(self.dev))
        self.H = {}
        for hold in ('group', 'old', 'test'):
            name = 'old' if args.smoke and hold == 'test' else hold
            x = np.load(fixture / f'{name}_X.npy', mmap_mode='r'); y = np.load(fixture / f'{name}_y.npy', mmap_mode='r')
            if args.smoke: x, y = x[:256], y[:256]
            self.H[hold] = (torch.from_numpy(np.array(x, copy=True)).to(self.dev),
                            torch.from_numpy(np.array(y, dtype='int64', copy=True)).to(self.dev))
        self.n = [len(y) for y in self.Y]
        if not args.smoke: check_full_manifest(manifest, self.n, len(self.H['test'][1]))
        self.counts = torch.stack([torch.bincount(y, minlength=16) for y in self.Y]).double()
        self.log_pi = ((self.counts + 1) / (self.counts + 1).sum(1, keepdim=True)).log().float()
        self.log_pg = ((self.counts.sum(0) + 1) / (self.counts.sum() + 16)).log().float()
        torch.manual_seed(args.seed); self.init = P.make('dagsnet').to(self.dev)
        self.priv = [copy.deepcopy(self.init) for _ in range(self.K)]
        self.work_private = copy.deepcopy(self.init); self.work_proxy = copy.deepcopy(self.init)
        self.eval_template = copy.deepcopy(self.init).eval()
        self.compiled = {}; self.compile_gates = []
        if args.backend == 'compiled':
            arm = args.arm; args.arm = arm + '_dml' if self.fedavg else arm   # frozen preflight compiles the proxy only for 'dml'
            self.compile_preflight(); args.arm = arm
        self.optim_state, self.swap_stats = {}, [0, 0]
        # hkd (FedKD hidden-state loss): client-local projector private h -> proxy h, never uploaded (R1), any private width (R2)
        self.hkd = 'hkd' in tokens; self.proj = []
        for _ in range(self.K if self.hkd else 0):
            width = P.latent_dim(self.init)   # private width -> proxy width (both DAGSNet here)
            self.proj.append(torch.nn.Linear(width, width, bias=False).to(self.dev)); torch.nn.init.eye_(self.proj[-1].weight)
        cal = np.random.default_rng(4242)
        self.cal = [torch.from_numpy(np.sort(cal.choice(n, min(n, CAL_ROWS), replace=False))).to(self.dev) for n in self.n]

    def mask(self, x):
        if len(self.zero):
            x = x.clone(); x[:, self.zero] = 0
        return x

    def transform(self, x, index, k, gen):
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
        self.swap_stats[0] += int(ok.sum()); self.swap_stats[1] += n
        return self.mask(x)

    @torch.no_grad()
    def logits(self, model, x):
        return super().logits(model, x) if not len(self.zero) else super().logits(model, _Masked(x, self))

    def private_epoch(self, method, k, r, part=0):
        """full_run.Run.private_epoch + hooks: batch-aligned part of the epoch (sub4, optimizer kept within the epoch),
        FedProx on the proxy (prox), receiver swap + position mask (swappos), receiver+position features zeroed (m4b),
        private learns the proxy only on not-true classes (ntd, FedNTD), private hidden layer matches the proxy's through a
        client-local projector (hkd, FedKD). Identity for fd_dml/fd_gaussian."""
        started = time.time()
        original = self.priv[k]; self.work_private.load_state_dict(original.state_dict(), strict=True)
        private = self.work_private; private.train()
        models = [] if self.fedavg else [private]; proxy = None
        if method.kd:
            proxy = self.work_proxy; proxy.load_state_dict(method.proxy[k].state_dict(), strict=True); proxy.train(); models.append(proxy)
        local = list(self.proj[k].parameters()) if self.hkd else []
        optimizer = torch.optim.AdamW([p for m in models for p in m.parameters()] + local, lr=self.lr(r), weight_decay=0.)
        scaler = torch.amp.GradScaler(); g = self.gen(r, k, 0)
        if part:
            optimizer.load_state_dict(self.optim_state[k][0]); scaler.load_state_dict(self.optim_state[k][1])
        torch.manual_seed(int(np.random.SeedSequence([self.a.seed, r, k, 7] + ([part] if part else [])).generate_state(1)[0]))
        order = torch.randperm(self.n[k], device=self.dev, generator=g)
        batches = (self.n[k] + self.a.batch - 1) // self.a.batch
        first, last = part * batches // self.parts, (part + 1) * batches // self.parts
        swap_gen = self.gen(r, k, 55, part) if self.swap else None
        anchor = None
        if self.mu:
            anchor = [method.global_state[name].detach().float() for name, _ in proxy.named_parameters()]
        total = torch.zeros(5, device=self.dev); applied = skipped = 0
        for start in range(first * self.a.batch, min(last * self.a.batch, len(order)), self.a.batch):
            torch.compiler.cudagraph_mark_step_begin()
            index = order[start:start + self.a.batch]; x, y = self.X[k][index].float(), self.Y[k][index]
            if self.swap: x = self.transform(x, index, k, swap_gen)
            elif len(self.zero): x = self.mask(x)
            labels = torch.arange(self.a.aux_batch, device=self.dev) % 16
            if method.aux: x = torch.cat([x, self.mask(method.draw(labels, g, k))])
            with torch.autocast('cuda', dtype=torch.float16):
                if self.fedavg:
                    proxy_ce = F.cross_entropy(self.forward(proxy, x)[0].float(), y)
                    ce = extra = proxy_extra = proxy_ce.detach() * 0; loss = proxy_ce
                else:
                    if self.hkd: zs, hs = (v.float() for v in self.forward(private, x))
                    else: zs = self.forward(private, x)[0].float()
                    ce = F.cross_entropy(zs[:len(index)], y); extra = ce.detach() * 0
                    proxy_ce = proxy_extra = ce.detach() * 0
                    if method.aux: extra = F.cross_entropy(zs[len(index):], labels)
                    if proxy is not None:
                        if self.hkd: zr, hr = (v.float() for v in self.forward(proxy, x))
                        else: zr = self.forward(proxy, x)[0].float()
                        proxy_ce = F.cross_entropy(zr, y)
                        ls, lr = F.log_softmax(zs, 1), F.log_softmax(zr, 1)
                        weight = 1 / (ce + proxy_ce).detach().clamp_min(1e-6)
                        if self.ntd:
                            extra = weight * F.kl_div(not_true(zs, y), not_true(zr, y).detach(), log_target=True, reduction='batchmean')
                        else:
                            extra = weight * F.kl_div(ls, lr.detach(), log_target=True, reduction='batchmean')
                        if self.hkd:   # scale-free MSE to the proxy's 256-d hidden layer, same adaptive weight as the KL
                            target = hr.detach()
                            extra = extra + weight * (self.proj[k](hs).float() - target).square().mean() / target.square().mean().clamp_min(1e-6)
                        proxy_extra = weight * F.kl_div(lr, ls.detach(), log_target=True, reduction='batchmean')
                    loss = ce + extra + proxy_ce + proxy_extra
            prox = ce.detach() * 0
            if anchor is not None:
                prox = 0.5 * MU * sum((p.float() - a).square().sum() for p, a in zip(proxy.parameters(), anchor))
                loss = loss + prox
            assert torch.isfinite(loss), ('full loss', k, r)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            for model in models:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.a.clip)
                assert torch.isfinite(norm) or scaler.is_enabled()
            if local: torch.nn.utils.clip_grad_norm_(local, self.a.clip)
            before = scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale() < before: skipped += 1
            else: applied += 1; total += torch.stack([ce.detach(), extra.detach(), proxy_ce.detach(), proxy_extra.detach(), prox.detach()])
            if self.a.wandb_online and (start // self.a.batch + 1) % 1024 == 0:
                import wandb
                wandb.run.log({'progress/round_in_flight': r, 'progress/part': part, 'progress/client': k,
                               'progress/client_steps': start // self.a.batch + 1, 'progress/applied': applied,
                               'progress/skipped': skipped, 'progress/seconds': time.time() - started,
                               'progress/loss': float(loss.detach()), 'progress/grad_norm': float(norm)})
        assert applied > 0 and skipped <= 16, (k, r, part, applied, skipped)
        for model in models: finite(model)
        if self.parts > 1:
            self.optim_state[k] = (copy.deepcopy(optimizer.state_dict()), copy.deepcopy(scaler.state_dict()))
        if not self.fedavg: original.load_state_dict(private.state_dict(), strict=True)
        drift = None
        if proxy is not None:
            method.proxy[k].load_state_dict(proxy.state_dict(), strict=True)
            num = sum(float((p.detach().float() - method.global_state[n].float()).square().sum()) for n, p in proxy.named_parameters())
            den = sum(float(method.global_state[n].float().square().sum()) for n, _ in proxy.named_parameters())
            drift = (num / max(den, 1e-30)) ** 0.5
        ce, extra, proxy_ce, proxy_extra, prox = (total / applied).tolist()
        common = dict(client=k, rows=self.n[k], applied=applied, skipped=skipped, part=part)
        method.logs[k] = dict(common, loss=proxy_ce + proxy_extra, positive_ce=proxy_ce, negative_ce=proxy_extra,
                              prox=prox, drift=drift)
        return dict(common, loss=ce + extra, real_ce=ce, auxiliary=extra, seconds=time.time() - started)

    @torch.no_grad()
    def evaluate(self, m, out, r, holds):
        """Ten metrics per client; the shared DML proxy is one model after aggregation, so it is scored once."""
        arrays, evaluated = {}, {}
        roles = {} if self.fedavg else {'private': (self.priv, ('head', 'gp'))}
        if m.ema_private is not None:
            models = []
            for k in range(self.K):
                model = copy.deepcopy(self.priv[k]); model.load_state_dict(m.ema_private[k], strict=True)
                calibrate(model, self.mask(self.X[k][self.cal[k]].float())); models.append(model)
            roles['private_ema'] = (models, ('head', 'gp'))
        if m.kd:
            for k in range(self.K): m.prepare(k, r)
            for k in range(1, self.K):
                assert all(torch.equal(a, b) for a, b in zip(m.proxy[0].state_dict().values(), m.proxy[k].state_dict().values()))
            roles['proxy'] = ([m.proxy[0]], ('head',))
        for role, (models, rules) in roles.items():
            results = {}
            for hold in holds:
                X, y = self.H[hold]
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
                    cc = cc * self.K if len(cc) == 1 else cc     # shared proxy: identical confusion for every client
                    mm = [metrics_from_confusion(c) for c in cc]
                    pc = np.mean([[x['f1'] for x in per_class_from_confusion(c, P.CLASSES)] for c in cc], axis=0)
                    result[rule] = dict(mean_metrics={key: float(np.mean([v[key] for v in mm])) for key in METRIC_KEYS},
                                        f1_macro=float(np.mean([v['f1_macro'] for v in mm])), benign_f1=float(pc[0]),
                                        f1_per_class=pc.tolist(), confusion_clients=np.stack(cc).tolist())
                    arrays[f'{role}_{hold}_{rule}'] = np.stack(preds[rule])
                results[hold] = result
            evaluated[role] = results
        pf = out / 'predictions' / f'round_{r:03d}.npz'; pf.parent.mkdir(exist_ok=True)
        tmp = pf.with_suffix('.tmp')
        with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
        os.replace(tmp, pf)
        return evaluated


class _Masked:
    """Lazy masked view so the frozen chunked/compiled evaluation path stays unchanged."""
    def __init__(self, x, run): self.x, self.run = x, run
    def __len__(self): return len(self.x)
    def __getitem__(self, s): return self.run.mask(self.x[s].float())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=ARMS, required=True); p.add_argument('--seed', type=int, choices=(42, 43), required=True)
    p.add_argument('--rounds', type=int, default=50); p.add_argument('--stop_after', type=int, default=6)
    p.add_argument('--batch', type=int, default=512); p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr_min', type=float, default=1e-5); p.add_argument('--clip', type=float, default=1.)
    p.add_argument('--aux_batch', type=int, default=256); p.add_argument('--aux_weight', type=float, default=1.)
    p.add_argument('--backend', choices=['eager', 'compiled'], default='compiled')
    p.add_argument('--out', default=str(P.ROOT / 'papers/full-drift/runs')); p.add_argument('--smoke', action='store_true')
    p.add_argument('--wandb_project', default='mobihet-fulldrift'); p.add_argument('--wandb_online', action='store_true')
    p.add_argument('--max_hours', type=float, default=7.)
    a = p.parse_args()
    if a.smoke: a.stop_after = 2; a.aux_batch = 64
    else: assert a.rounds == 50 and a.stop_after in (6, 15) and a.wandb_online
    out = Path(a.out) / (f'{a.arm}_s{a.seed}' + ('' if a.smoke or a.stop_after == 6 else f'_e{a.stop_after}')); out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'config.json').exists(), 'single-session probe: no resume'
    config = {k: v for k, v in vars(a).items() if k not in ('out', 'wandb_online', 'wandb_project', 'max_hours')}
    config.update(parts=PARTS, mu=MU, swap_p=SWAP_P, ema_beta=EMA_BETA, cal_rows=CAL_ROWS)
    source = {f'mh/{f.name}': digest(f) for f in sorted((P.ROOT / 'mh').glob('*.py'))}
    source.update({f'proj/{f.name}': digest(f) for f in sorted(Path(__file__).parent.glob('*.py'))})
    fx = P.ROOT / 'papers/pilot/fixture'; manifest = json.loads((fx / 'manifest.json').read_text())
    for name, sha in manifest['files'].items(): assert digest(fx / name) == sha, ('fixture mismatch', name)
    identity = dict(config=config, source=source, fixture=digest(fx / 'manifest.json'))
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    save(out / 'config.json', dict(identity, fingerprint=fingerprint))
    wb = None
    if a.wandb_online:
        import wandb
        key = os.environ.get('WANDB_API_KEY'); assert key, 'W&B key required'
        assert wandb.login(key=key, anonymous='never', relogin=True, verify=True)
        wb = wandb.init(project=a.wandb_project, id=out.name + '-' + fingerprint[:8], resume='allow', name=out.name,
                        config=dict(config, fingerprint=fingerprint), mode='online', dir=str(out),
                        settings=wandb.Settings(save_code=False, disable_code=True))
        assert not wb.settings._offline
        wb.log({'preflight/online': 1}); print('WANDB', wb.url, flush=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    t0 = time.time(); R = Run(a); m = Method(R); m.prior()
    print('setup', a.arm, 'rows', sum(R.n), 'parts', R.parts, 'mu', R.mu, 'swap', R.swap, 'sec', round(time.time() - t0, 1), flush=True)
    hist, previous, step = [], None, 0
    for r in range(1, a.stop_after + 1):
        if time.time() - t0 + 1.15 * max([v['sec'] for v in hist] or [0]) + 60 > a.max_hours * 3600:
            print('budget stop before epoch', r, flush=True); break
        t1 = time.time(); private_logs, proxy_logs = [], []
        for part in range(R.parts):
            step += 1; m.start_round()
            for k in range(R.K):
                if m.kd: m.prepare(k, step)
                private_logs.append(R.private_epoch(m, k, r, part))
                if m.kd:
                    packet, log = m.proxy_epoch(k, step); proxy_logs.append(log)
                    if packet is not None: m.aggregate_sum.add(k, packet)
            m.aggregate(step)
        m.update_ema()
        algorithm_seconds = time.time() - t1
        diagnostics = dict(swap_fraction=R.swap_stats[0] / max(R.swap_stats[1], 1), aggregations=step,
                           **{k: v for k, v in m.diag.items() if isinstance(v, (int, float))})
        if proxy_logs:
            diagnostics.update(drift_mean=float(np.mean([v['drift'] for v in proxy_logs])),
                               prox_mean=float(np.mean([v['prox'] for v in proxy_logs])))
        row = dict(algorithm_seconds=algorithm_seconds, round=r, lr=R.lr(r), clients=private_logs, proxy_clients=proxy_logs,
                   version=m.version, fingerprint=fingerprint, prev_sha=previous, diagnostics=diagnostics,
                   loss=float(np.mean([v['loss'] for v in (proxy_logs if m.fedavg else private_logs)])))
        final = r == a.stop_after
        row['eval'] = R.evaluate(m, out, r, ('group', 'old', 'test') if final else ('group', 'old'))
        row['predictions_sha'] = digest(out / 'predictions' / f'round_{r:03d}.npz')
        weights = dict(private=[] if m.fedavg else [{k: v.detach().cpu() for k, v in model.state_dict().items()} for model in R.priv],
                       proxy=[{k: v.detach().cpu() for k, v in m.proxy[0].state_dict().items()}] if m.kd else [],
                       ema_private=[{k: v.detach().cpu() for k, v in s.items()} for s in (m.ema_private or [])],
                       architecture=dict(P.CFG, n_features=P.D, num_classes=P.C), round=r, fingerprint=fingerprint)
        wp = out / 'weights' / f'round_{r:03d}.pt'; save(wp, weights); row['weights_sha'] = digest(wp)
        row['sec'] = time.time() - t1; previous = row['weights_sha']
        save(out / 'complete' / f'round_{r:03d}.json', row); hist.append(row)   # atomic commit LAST
        role, rule = ('proxy', 'head') if m.kd else ('private', 'gp')
        print(f'r{r} {a.arm} s{a.seed} {role} group {rule} {row["eval"][role]["group"][rule]["f1_macro"]:.4f} '
              f'loss {row["loss"]:.4f} {row["sec"]:.0f}s', flush=True)
        if wb:
            metrics = dict(round=r, loss=row['loss'], lr=row['lr'], seconds=row['sec'], algorithm_seconds=algorithm_seconds,
                           **{'algorithm/' + k: v for k, v in diagnostics.items()})
            for model_role, holds in row['eval'].items():
                for hold, rules in holds.items():
                    for rule_name, ev in rules.items():
                        metrics.update({f'{model_role}/{hold}/{rule_name}/{key}': v for key, v in ev['mean_metrics'].items()})
            wb.log(metrics)
    result = dict(config=vars(a), fingerprint=fingerprint, history=hist, bytes=R.bytes, train_rows=R.n,
                  complete=len(hist) == a.stop_after, minutes=(time.time() - t0) / 60)
    save(out / 'result.json', result)
    with (out / 'metrics.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['round', 'role', 'holdout', 'rule', *METRIC_KEYS]); w.writeheader()
        for row in hist:
            for role, holds in row['eval'].items():
                for hold, rules in holds.items():
                    for rule, ev in rules.items():
                        w.writerow(dict(round=row['round'], role=role, holdout=hold, rule=rule, **ev['mean_metrics']))
    if wb: wb.summary['complete'] = result['complete']; wb.summary['rounds_completed'] = len(hist); wb.finish()
    print('done', a.arm, 'complete', result['complete'], flush=True)


if __name__ == '__main__':
    main()
