"""Private hard-label tail risk, with fresh Gaussian/DML references and audited mass."""
import hashlib
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.v2_methods import Method as PaidPrior, finite
from mh.v8_hybrid import Method as Reference
from tail_math import C, ALPHA, LAMBDA, EDGES, histogram, threshold, snapshot, probe, packet, weights

ARMS = ['ct_local', 'ct_fixed', 'ct_own', 'ct_fl', 'ct_shuffle', 'ct_classmass', 'ct_gaussian', 'ct_dml']


class Method:
    def __init__(self, run, initialize=True):
        self.R, self.arm = run, run.a.arm
        self.role, self.version, self.diag = 'private', 0, {}
        self.mass = torch.zeros(run.K, C, dtype=torch.float64, device=run.dev)
        self.log_pi_effective = run.log_pi.clone()
        self.proxy = []
        self.zeta = torch.zeros(run.K, C, device=run.dev)
        self.valid = torch.zeros(run.K, C, dtype=torch.bool, device=run.dev)
        self.class_weight = torch.ones(run.K, C, device=run.dev)
        self.identity = hashlib.sha256(EDGES.tobytes()).digest()[:16]
        self.reference = None
        if self.arm in ('ct_gaussian', 'ct_dml'):
            args = SimpleNamespace(**vars(run.a))
            args.arm = 'h_gaussian' if self.arm == 'ct_gaussian' else 'h_dml'
            holder = SimpleNamespace(**vars(run))
            holder.a = args
            self.reference = Reference(holder, initialize=initialize)
            self.reference.R = run
            self.proxy = self.reference.proxy

    def record(self, wire, r, k, direction, kind):
        self.R.bytes.append((r, k, direction, kind, len(wire)))

    def prior(self):
        PaidPrior.prior(self)
        if self.reference:
            self.reference.prior()
        if self.arm not in ('ct_local', 'ct_gaussian', 'ct_dml'):
            bootstrap = packet(3, 0, EDGES.tobytes(), self.identity)
            for k in range(self.R.K):
                self.record(bootstrap, 0, k, 'down', 'tail_dictionary')

    def start_round(self):
        self.logs = {}
        if self.reference:
            self.reference.start_round()
            if hasattr(self.reference, 'aggregate_sum'):
                self.aggregate_sum = self.reference.aggregate_sum

    def prepare(self, k, r):
        return self.reference.prepare(k, r)

    def proxy_epoch(self, k, r):
        return self.reference.proxy_epoch(k, r)

    @torch.no_grad()
    def scan(self, r):
        started = time.monotonic()
        losses, labels, local = [], [], []
        for k, model in enumerate(self.R.priv):
            y = self.R.Y[k]
            ce = F.cross_entropy(self.R.logits(model, self.R.X[k]), y, reduction='none').cpu().numpy()
            yy = y.cpu().numpy()
            losses.append(ce)
            labels.append(yy)
            local.append(histogram(ce, yy))
        zeta, valid, prior, ups, down = snapshot(local, r, self.identity)
        self.diag = probe(losses, labels, local, zeta, valid)
        self.diag['scan_seconds'] = time.monotonic()-started
        self.diag['threshold'] = zeta.tolist()
        self.diag['quantized_approximation'] = True
        self.latest_packet = down
        if self.arm != 'ct_local':
            for k, up in enumerate(ups):
                self.record(up, r, k, 'up', 'tail_histogram')
                self.record(down, r, k, 'down', 'tail_threshold')
        for k, (ce, y, hist) in enumerate(zip(losses, labels, local)):
            used, active = zeta.copy(), valid.copy()
            if self.arm == 'ct_own':
                used, active = threshold(hist, np.full(C, 5), minimum=32)
                active &= valid  # common cohort gate, no single-client fallback for pooled classes
            elif self.arm == 'ct_shuffle':
                perm = np.roll(np.arange(C), 1)  # fixed class derangement; not mass matched
                used, active = used[perm], active[perm]
            elif self.arm == 'ct_fixed':
                used = np.full(C, np.log(C), dtype='float32')
            self.zeta[k] = torch.from_numpy(used).to(self.R.dev)
            self.valid[k] = torch.from_numpy(active).to(self.R.dev)
            ww = weights(ce, y, zeta, valid)[0]
            mass = np.bincount(y, weights=ww, minlength=C)
            count = np.bincount(y, minlength=C)
            mean = np.divide(mass, count, out=np.ones(C), where=count>0)
            self.class_weight[k] = torch.from_numpy(mean.astype('float32')).to(self.R.dev)

    def aggregate(self, r):
        if self.reference:
            self.reference.aggregate(r)
            self.version = self.reference.version
            if hasattr(self.reference, 'latest_packet'):
                self.latest_packet = self.reference.latest_packet
        else:
            self.version += 1
            if r >= 2 and (r == 2 or self.arm != 'ct_local'):
                self.scan(r)

    def state(self):
        result = dict(version=self.version, mass=self.mass, log_pi_effective=self.log_pi_effective,
                      zeta=self.zeta, valid=self.valid, class_weight=self.class_weight,
                      diag={k:v for k,v in self.diag.items() if k != 'scan_seconds'})
        if self.reference:
            result['reference_state'] = self.reference.state()
        return result


def private_epoch(run, method, k, r):
    model = run.priv[k].train()
    ref = method.reference
    teacher = method.proxy[k].train() if ref else None
    mods = [model, teacher] if ref else [model]
    opt = torch.optim.AdamW([p for m in mods for p in m.parameters()], lr=run.lr(r), weight_decay=0.)
    scaler = torch.amp.GradScaler()
    torch.manual_seed(int(np.random.SeedSequence([run.a.seed, r, k, 7]).generate_state(1)[0]))
    generator = run.gen(r, k, 0)
    x, y = run.X[k], run.Y[k]
    order = torch.randperm(len(y), device=run.dev, generator=generator)
    mass = torch.zeros(C, dtype=torch.float64, device=run.dev)
    totals = torch.zeros(3, device=run.dev)
    applied = skipped = tail_rows = seen_rows = 0
    for start in range(0, len(y), run.a.batch):
        idx = order[start:start+run.a.batch]
        yy, xx = y[idx], x[idx].float()
        with torch.autocast('cuda', dtype=torch.float16):
            labels = torch.arange(run.a.aux_batch, device=run.dev) % C
            synthetic = ref.draw(labels, generator, k) if ref and ref.aux else None
            mixed = torch.cat([xx, synthetic]) if synthetic is not None else xx
            z = model(mixed)[0].float()
            ce = F.cross_entropy(z[:len(idx)], yy, reduction='none')
            target_mass = F.one_hot(yy, C).double()
            extra = ce.mean()*0
            tail = torch.zeros(len(yy), device=run.dev, dtype=torch.bool)
            if ref:
                tz = teacher(mixed)[0].float()
                teacher_loss = F.cross_entropy(tz[:len(idx)], yy)
                if ref.kd:
                    ls, lt = F.log_softmax(z[:len(idx)], 1), F.log_softmax(tz[:len(idx)], 1)
                    scale = 1/(ce.mean()+teacher_loss).detach().clamp_min(1e-6)
                    extra = scale*F.kl_div(ls, lt.detach(), log_target=True, reduction='batchmean')
                    teacher_loss += scale*F.kl_div(lt, ls.detach(), log_target=True, reduction='batchmean')
                    target_mass = target_mass.mean(0)+scale.double()*lt.detach().exp().double().mean(0)
                else:
                    target_mass = target_mass.mean(0)
                if ref.aux:
                    extra += run.a.aux_weight*F.cross_entropy(z[len(idx):], labels)
                    teacher_loss += run.a.aux_weight*F.cross_entropy(tz[len(idx):], labels)
                    target_mass += run.a.aux_weight*F.one_hot(labels, C).double().mean(0)
                loss = ce.mean()+extra+teacher_loss
            else:
                active = method.valid[k, yy] & (r > 2) & (method.arm != 'ct_local')
                tail = (ce.detach() > method.zeta[k, yy]) & active
                ww = torch.where(active, 1-LAMBDA+LAMBDA/ALPHA*tail.float(), 1.)
                if method.arm == 'ct_classmass' and r > 2:
                    ww = method.class_weight[k, yy]
                    loss = (ww*ce).mean()
                else:
                    zeta = method.zeta[k, yy]
                    cvar = zeta+(ce-zeta).relu()/ALPHA
                    loss = torch.where(active, (1-LAMBDA)*ce+LAMBDA*cvar, ce).mean()
                target_mass = (target_mass*ww[:, None].double()).mean(0)
            assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        for m in mods:
            torch.nn.utils.clip_grad_norm_(m.parameters(), run.a.clip)
        before = scaler.get_scale()
        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        if scaler.get_scale() < before:
            skipped += 1
        else:
            applied += 1
            mass += target_mass
            totals += torch.stack([loss.detach(), ce.detach().mean(), extra.detach()])
            tail_rows += int(tail.sum()); seen_rows += len(yy)
        if getattr(run, 'wb', None) is not None and (start//run.a.batch)%32 == 0:
            run.wb.log(dict(**{'progress/round':r,'progress/client':k,'progress/applied':applied,
                              'progress/skipped':skipped,'progress/real_ce':float(ce.detach().mean())}),
                       step=r, commit=False)
    assert applied > 0 and skipped <= 16
    for m in mods:
        finite(m)
    method.mass[k] = mass
    prior = mass/applied+1/max(len(y), 1)
    method.log_pi_effective[k] = (prior/prior.sum()).log().float()
    run.log_pi[k] = method.log_pi_effective[k]
    common = dict(client=k, rows=len(y), applied=applied, skipped=skipped)
    values = (totals/applied).tolist()
    if ref:
        ref.logs[k] = dict(common, loss=values[0]-values[1]-values[2])
    return dict(common, loss=values[0], real_ce=values[1], auxiliary=values[2],
                tail_rows=tail_rows, applied_rows=seen_rows)
