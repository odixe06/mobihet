"""Locked H1 x H2 factorial: mutual real-batch KD and optional mixed-batch Gaussian CE."""
import copy
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.v2_math import Layout
from mh.v2_methods import Method as Prior, finite
from mh.v7_nystrom import Method as Gaussian
from mh.model_packet import Codec, Sum

FAMILIES = {'h': ['h_local', 'h_gaussian', 'h_dml', 'h_hybrid',
                  'h_gaussian_own', 'h_dml_own', 'h_hybrid_own']}
ARMS = FAMILIES['h']


class Method(Prior):
    def __init__(self, run, initialize=True):
        self.R, self.arm, self.diag = run, run.a.arm, {}
        self.role, self.version = 'private', 0
        self.kd = 'dml' in self.arm or 'hybrid' in self.arm
        self.aux = 'gaussian' in self.arm or 'hybrid' in self.arm
        self.own = self.arm.endswith('_own')
        self.layout = Layout(run.init)
        self.global_vector = self.layout.flatten(run.init).clone()
        # Compute-matched control trains the second DAGSNet locally without KD.
        torch.manual_seed(run.a.seed + 1000)
        initial = P.make('dagsnet').to(run.dev)
        self.proxy = [copy.deepcopy(initial) for _ in range(run.K)]
        self.global_state = {name: value.detach().clone() for name, value in initial.state_dict().items()}
        self.codec = Codec(initial, f'v8:{self.arm}:{run.a.seed}')
        self.gaussian_state = None
        self.gaussian = None
        if self.aux:
            args = SimpleNamespace(arm='n_full', seed=run.a.seed)
            self.gaussian = Gaussian(SimpleNamespace(init=run.init, dev=run.dev, a=args), initialize=False)
            if initialize:
                self.gaussian.R = run
                self.gaussian.mode = 'own' if self.own else 'fl'
                self.gaussian.make_tables()
                self.gaussian_state = self.gaussian.state()

    def prepare(self, k, r):
        if self.kd and not self.own:
            self.proxy[k].load_state_dict(self.global_state, strict=True)
        return self.proxy[k]

    def prior(self):
        super().prior()
        if self.kd and not self.own:
            packet = self.codec.encode(self.global_state, 0, 0)
            self.global_state, _, _ = self.codec.decode(packet, self.R.dev)
            for k in range(self.R.K):
                self.record(packet, 0, k, 'down', 'proxy_bootstrap')

    def start_round(self):
        self.logs = {}
        if self.kd and not self.own:
            self.aggregate_sum = Sum(self.codec, self.version, self.R.dev)

    def proxy_epoch(self, k, r):
        packet = None
        if self.kd and not self.own:
            packet = self.codec.encode(self.proxy[k].state_dict(), self.version, self.version + 1)
            self.record(packet, r, k, 'up', 'proxy_state')
        return packet, self.logs[k]

    def aggregate(self, r):
        if self.kd and not self.own:
            state = self.aggregate_sum.finish(self.R.K)
            packet = self.codec.encode(state, self.version, self.version + 1)
            self.global_state, _, _ = self.codec.decode(packet, self.R.dev, self.version)
            for k in range(self.R.K):
                self.record(packet, r, k, 'down', 'proxy_state')
                self.proxy[k].load_state_dict(self.global_state, strict=True)
            self.global_vector = self.layout.flatten(self.proxy[0]).clone()
            self.latest_packet = packet
        self.version += 1

    def state(self):
        return dict(global_vector=self.global_vector, global_state=self.global_state,
                    version=self.version, gaussian_state=self.gaussian_state)

    def draw(self, labels, generator, k):
        self.gaussian.R = self.R
        self.gaussian.mode = 'own' if self.own else 'fl'
        for name, value in self.gaussian_state.items():
            setattr(self.gaussian, name, value)
        return self.gaussian.draw(labels, generator, k)


def private_epoch(run, method, k, r):
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
        labels = torch.arange(run.a.aux_batch, device=run.dev) % P.C
        synthetic = method.draw(labels, generator, k) if method.aux else torch.randn(
            len(labels), P.D, device=run.dev, generator=generator)
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
                # Equal number of model input rows/backward graphs; dummy noise must not alter BN.
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
    finite(private); finite(proxy)
    ce_s, extra_s, ce_r, extra_r = (totals / applied).tolist()
    common = dict(client=k, rows=len(y), applied=applied, skipped=skipped)
    method.logs[k] = dict(common, loss=ce_r + extra_r, positive_ce=ce_r, negative_ce=extra_r)
    return dict(common, loss=ce_s + extra_s, real_ce=ce_s, auxiliary=extra_s)
