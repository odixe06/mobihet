"""Core with a fixed 12-edge forest, Gaussian conditional and synthetic marginal transport.

Only pooled histogram counts and the existing compact copula statistics cross clients.
Calibration uses independent synthetic draws, never held-out rows. See screening-forest-t4.md.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from methods.mobihet_wstudy import WLive
from scripts.direct_generator import Generator as CPU, forest_order
from scripts.probe_pair_generator import PAIRS


def quantile_probabilities(source):
    grid = np.linspace(0, 1, source.shape[0])
    probability = np.empty_like(source)
    for j in range(source.shape[1]):
        _, first, inverse, count = np.unique(source[:, j], return_index=True, return_inverse=True, return_counts=True)
        probability[:, j] = ((grid[first] + grid[first + count - 1]) / 2)[inverse]
    return probability


class Sampler:
    def __init__(self, models, device):
        first = models[0]
        self.selected, self.rest = first.selected, first.rest.tolist()
        self.C, self.D = len(models), len(first.mu)
        self.roots = [self.selected.index(j) for j in first.roots]
        self.edges = [(self.selected.index(a), self.selected.index(b)) for a, b in first.edges]
        self.tensors = {}
        def put(name, values):
            self.tensors[name] = torch.as_tensor(np.stack(values), device=device, dtype=torch.float32).contiguous()
        put("cdf", [m.cdf for m in models])
        put("mass", [m.decoded_mass for m in models])
        put("mu", [m.mu for m in models])
        put("a", [m.a for m in models])
        put("chol", [m.conditional_chol for m in models])
        roots, conditionals, fine = [], [], []
        for m in models:
            single = m.single[self.selected].reshape(len(self.selected), 16, 16)
            coarse = single.sum(-1)
            roots.append((coarse[self.roots] / coarse[self.roots].sum(-1, keepdims=True)).cumsum(-1))
            fine.append((single / np.maximum(single.sum(-1, keepdims=True), 1)).cumsum(-1))
            tables = []
            for a, b in first.edges:
                table = m.joint[(a, b)] if (a, b) in m.joint else m.joint[(b, a)].T
                tables.append((table / np.maximum(table.sum(-1, keepdims=True), 1)).cumsum(-1))
            conditionals.append(np.stack(tables))
        put("roots", roots); put("conditional", conditionals); put("fine", fine)
        put("quantiles", [m.source_quantiles.T for m in models])
        put("probabilities", [quantile_probabilities(m.source_quantiles).T for m in models])
        # Supported rows end at exactly one; float rounding must not select a zero-count tail bin.
        for name in ("roots", "conditional", "fine"):
            self.tensors[name].clamp_(max=1)
        self.dev = device

    def decode(self, u, columns):
        # Class-major search avoids materializing one 4097/256-entry table per draw.
        cdf, mass = self.tensors["cdf"][:, columns], self.tensors["mass"][:, columns]
        value = u.transpose(1, 2).contiguous()
        j = torch.searchsorted(cdf.contiguous(), value).clamp(max=255)
        hi, m = cdf.gather(-1, j), mass.gather(-1, j)
        frac = ((value - hi + m) / m.clamp_min(1e-12)).clamp(0, 1)
        return (-4 + (j + frac) / 32).transpose(1, 2)

    def transport(self, x):
        q, p = self.tensors["quantiles"], self.tensors["probabilities"]
        value = x.transpose(1, 2).contiguous()
        j = torch.searchsorted(q, value).clamp(1, q.shape[-1] - 1)
        lo, hi = q.gather(-1, j-1), q.gather(-1, j)
        plo, phi = p.gather(-1, j-1), p.gather(-1, j)
        u = plo + ((value-lo) / (hi-lo).clamp_min(1e-20)).clamp(0, 1) * (phi-plo)
        u = torch.where(value < q[..., :1], 1e-6, torch.where(value > q[..., -1:], 1-1e-6, u))
        return self.decode(u.clamp(1e-6, 1-1e-6).transpose(1, 2), self.rest)

    def sample(self, per_class, generator):
        C, n, S = self.C, per_class, len(self.selected)
        t = self.tensors
        coarse = torch.zeros(C, n, S, device=self.dev, dtype=torch.long)
        u = torch.rand(C, len(self.roots), n, device=self.dev, generator=generator)
        coarse[:, :, self.roots] = torch.searchsorted(t["roots"].contiguous(), u.contiguous(), right=True).clamp(max=15).transpose(1, 2)
        classes = torch.arange(C, device=self.dev)[:, None]
        for k, (a, b) in enumerate(self.edges):
            cdf = t["conditional"][:, k][classes, coarse[:, :, a]]
            u = torch.rand(C, n, 1, device=self.dev, generator=generator)
            coarse[:, :, b] = (u >= cdf).sum(-1).clamp(max=15)
        features = torch.arange(S, device=self.dev)[None, :, None]
        groups = coarse.transpose(1, 2)
        fine_cdf = t["fine"][classes[:, :, None], features, groups]
        u = torch.rand(C, S, n, 1, device=self.dev, generator=generator)
        fine = (groups*16 + (u >= fine_cdf).sum(-1).clamp(max=15))
        fraction = torch.rand(C, S, n, device=self.dev, generator=generator)
        selected = (-4 + (fine + fraction) / 32).transpose(1, 2)
        cdf = t["cdf"][:, self.selected].gather(-1, fine)
        mass = t["mass"][:, self.selected].gather(-1, fine)
        z = torch.special.ndtri((cdf - mass*(1-fraction)).clamp(1e-6, 1-1e-6)).transpose(1, 2)
        mean = t["mu"][:, None, self.rest] + (z-t["mu"][:, None, self.selected]) @ t["a"].transpose(1, 2)
        noise = torch.randn(C, n, len(self.rest), device=self.dev, generator=generator)
        remaining = self.decode(torch.special.ndtr(mean + noise @ t["chol"].transpose(1, 2)).clamp(1e-6, 1-1e-6), self.rest)
        x = torch.empty(C, n, self.D, device=self.dev)
        x[:, :, self.selected], x[:, :, self.rest] = selected, self.transport(remaining)
        return x


class Method(WLive):
    def _copula_bootstrap(self, client, NB=256):
        started = time.monotonic()
        super()._copula_bootstrap(client, NB)
        assert self.synth == "copula" and self.w_boot == "compact" and self.variant == "stream"
        env, C, D = self.env, 16, self.cop_mu.shape[-1]
        order = json.loads((Path(__file__).resolve().parents[1]/"knowledge/input_order.json").read_text())["inputs"]
        pairs = [(order.index(a), order.index(b)) for a,b in PAIRS]
        selected, _, _ = forest_order(pairs)
        joints = torch.zeros(C, len(pairs), 16, 16, dtype=torch.float64, device=env.dev)
        singles = torch.zeros(C, len(selected), NB, dtype=torch.float64, device=env.dev)
        for start in range(0, len(env.train.Y), 1 << 18):
            x = env.train.X[start:start+(1<<18)].float()
            y = env.train.Y[start:start+(1<<18)].long()
            fine = ((x[:, selected]+4)*32).long().clamp(0, 255)
            f = torch.arange(len(selected), device=env.dev)
            singles.view(-1).index_add_(0, ((y[:,None]*len(selected)+f)*256+fine).reshape(-1), torch.ones(fine.numel(), dtype=torch.float64, device=env.dev))
            coarse = ((x+4)*2).long().clamp(0,15)
            for k,(a,b) in enumerate(pairs):
                code = ((y*len(pairs)+k)*16+coarse[:,a])*16+coarse[:,b]
                joints.view(-1).index_add_(0,code,torch.ones(len(y),dtype=torch.float64,device=env.dev))
        singles, joints = env.allreduce(singles), env.allreduce(joints)
        assert bool((singles >= 0).all()) and float(singles.max()) < 2**32 and float(joints.max()) < 2**32
        # uint32 wire roundtrip, exact integer sums; no row or learned model payload.
        single_np, joint_np = (v.cpu().numpy().astype(np.uint32).astype(np.float64) for v in (singles,joints))
        for k,(a,b) in enumerate(pairs):
            assert np.array_equal(joint_np[:,k].sum(-1),single_np[:,selected.index(a)].reshape(C,16,16).sum(-1))
            assert np.array_equal(joint_np[:,k].sum(-2),single_np[:,selected.index(b)].reshape(C,16,16).sum(-1))
        models=[]
        for c in range(C):
            mass = self.cop_mass[c].double().cpu().numpy()
            total = single_np[c,0].sum()
            single = mass*total
            single[selected] = single_np[c]
            joint = {pair:joint_np[c,k] for k,pair in enumerate(pairs)}
            m = CPU(single, joint, self.cop_mu[c].double().cpu().numpy(), self.cop_L[c].double().cpu().numpy(),pairs)
            m.cdf, m.decoded_mass = self.cop_cdf[c].double().cpu().numpy(), mass
            reference = CPU.sample(m,np.random.default_rng(9200),32768,direct=True)
            m.source_quantiles = np.quantile(reference[:,m.rest],np.linspace(0,1,4097),axis=0)
            models.append(m)
        self.forest = Sampler(models,env.dev)
        joint_bytes, single_bytes = C*len(pairs)*16*16*4, C*len(selected)*256*4
        self.bootstrap_bytes["up"] += joint_bytes
        self.bootstrap_bytes["down"] += joint_bytes+single_bytes
        digest=hashlib.sha256()
        for key,value in self.forest.tensors.items():
            assert torch.isfinite(value).all(), key
            digest.update(key.encode());digest.update(value.cpu().numpy().tobytes())
        self.forest_sha = digest.hexdigest()
        # Separate generator: benchmark must not consume the training stream.
        old_gen = self.gen_g
        timing = {}
        try:
            for groups in (1,32):
                ctx=SimpleNamespace(Gfix=groups,x=env.train.X[:1])
                for name,sample in (("copula",lambda:super(Method,self)._synth(ctx)),("forest",lambda:self._synth(ctx))):
                    self.gen_g=torch.Generator(device=env.dev).manual_seed(9300)
                    for _ in range(5):sample()
                    torch.cuda.synchronize();stamp=time.monotonic()
                    for _ in range(30):values,_=sample()
                    torch.cuda.synchronize()
                    assert torch.isfinite(values).all()
                    timing[f"{name}_G{groups}_ms"]=(time.monotonic()-stamp)*1000/30
        finally:
            self.gen_g=old_gen
        record=dict(rank=env.rank,pairs=pairs,selected=selected,extra_up=joint_bytes,extra_down=joint_bytes+single_bytes,
                    bootstrap_bytes=self.bootstrap_bytes,static_sha256=self.forest_sha,calibration_rows=32768,quantiles=4097,
                    seconds=time.monotonic()-started,counts_consistent=True,codec="uint32 joint + selected fine counts",sampler_timing=timing)
        folder=os.environ.get("MOBIHET_RECOVERY_DIR")
        if folder:
            path=Path(folder);path.mkdir(parents=True,exist_ok=True)
            (path/f"forest_rank{env.rank}.json").write_text(json.dumps(record,indent=2)+"\n")
        print("FOREST_BOOTSTRAP",json.dumps(record),flush=True)

    def _synth(self,ctx):
        n=ctx.Gfix*self.aux
        x=self.forest.sample(n//16,self.gen_g).transpose(0,1).reshape(ctx.Gfix,self.aux,-1)
        return x.to(ctx.x.dtype),self.labels[None].expand(ctx.Gfix,self.aux)

    def state(self):
        return super().state() | {"forest_sha256":self.forest_sha}

    def load_state(self,state):
        assert state["forest_sha256"]==self.forest_sha,"Bootstrap/calibration changed across resume"
        super().load_state(state)
