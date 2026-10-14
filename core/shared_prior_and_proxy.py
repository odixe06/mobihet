"""Approved V2 screen: proxy CE+KD, fixed-basis proxy, and smoothed density ratios."""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.v2_math import (Layout, CountSketch, SeedBasis, SecureSum,
                        pack_vector, unpack, ratio_weights)

ARMS = ['local', 'a_own', 'a_dense', 'a_small_noef', 'a_small_ef',
        'a_medium_noef', 'a_medium_ef', 'b_dense', 'b_own512', 'b_seed512',
        'b_own1024', 'b_seed1024', 'diag', 'c_own', 'c_fl', 'c_shuffle']


def finite(model):
    assert all(torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point())


@torch.no_grad()
def calibrate(model, x, batch=512):
    model.eval()
    bn = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    old = [m.momentum for m in bn]
    for m in bn:
        m.reset_running_stats(); m.momentum = None; m.train()
    for start in range(0, len(x), batch):
        with torch.autocast('cuda', dtype=torch.float16):
            model(x[start:start+batch].float())
    for m, momentum in zip(bn, old):
        m.momentum = momentum
    model.eval(); finite(model)


class Method:
    def __init__(self, run, initialize=True):
        self.R, self.arm, self.diag = run, run.a.arm, {}
        self.role = 'proxy' if self.arm.startswith('b_') else 'private'
        self.proxy = [copy.deepcopy(run.init) for _ in range(run.K)] if self.arm not in ('local','diag') else []
        self.layout = Layout(run.init); assert self.layout.size == 395024
        self.global_vector = self.layout.flatten(run.init).clone()
        self.coeff = None; self.basis = None; self.sketch = None; self.residual = None
        self.version = 0
        if self.arm.startswith('b_') and self.arm != 'b_dense':
            k = 1024 if self.arm.endswith('1024') else 512
            if run.a.smoke: k = 8  # formula/driver smoke only, never used as a feasibility result
            self.basis = SeedBasis(run.init, k, 1901, run.dev)
            self.coeff = torch.zeros(k, device=run.dev)
            self.own_coeff = [self.coeff.clone() for _ in range(run.K)]
        if 'small' in self.arm or 'medium' in self.arm:
            depth, width, self.topk = (3,256,128) if 'small' in self.arm else (5,2048,2048)
            self.sketch = CountSketch(self.layout.size, depth, width, 1701, run.dev)
            self.residual = torch.zeros(depth*width, device=run.dev)
        if initialize and (self.arm == 'diag' or self.arm.startswith('c_')):
            self.make_gaussian()

    def record(self, packet, r, client, direction, kind):
        self.R.bytes.append((r, client, direction, kind, len(packet)))

    def prior(self):
        R = self.R
        agg = SecureSum('counts', 0, P.C, R.dev)
        for k in range(R.K):
            packet = pack_vector('counts', R.counts[k], count=R.n[k])
            self.record(packet, 0, k, 'up', 'prior'); agg.add(k, packet)
        total, _ = agg.finish()
        packet = pack_vector('counts', total)
        pg = unpack(packet, R.dev)['values']; R.log_pg = ((pg+1)/(pg+1).sum()).log()
        R.ratio_pi = pg / pg.sum()
        for k in range(R.K): self.record(packet, 0, k, 'down', 'prior')

    def make_gaussian(self):
        R = self.R
        size = P.C*(1+2*P.D); agg = SecureSum('moments', 0, size, R.dev)
        for k, (X,Y) in enumerate(zip(R.X,R.Y)):
            moments = torch.zeros(P.C,1+2*P.D,device=R.dev)
            for y in range(P.C):
                x = X[Y==y].float(); moments[y,0] = len(x)
                moments[y,1:1+P.D] = x.sum(0); moments[y,1+P.D:] = x.square().sum(0)
            packet = pack_vector('moments', moments.flatten(), count=R.n[k])
            self.record(packet,0,k,'up','moments'); agg.add(k,packet)
        m, _ = agg.finish(); m = m.view(P.C,-1); n=m[:,0:1]
        assert (n>0).all(), 'fixture must cover all classes in cohort'
        mu=m[:,1:1+P.D]/n; var=(m[:,1+P.D:]/n-mu.square()).clamp_min(1e-4)
        packet=pack_vector('moments',torch.cat([mu.flatten(),var.sqrt().flatten()]),half=True)
        decoded=unpack(packet,R.dev)['values']; self.mu,self.sd=decoded.reshape(2,P.C,P.D)
        for k in range(R.K): self.record(packet,0,k,'down','gaussian')
        self.q_version = 0

    def draw(self, labels, g):
        return self.mu[labels] + self.sd[labels]*torch.randn(len(labels),P.D,device=self.R.dev,generator=g)

    def gaussian_calibration(self):
        n=256 if self.R.a.smoke else 4096
        y=torch.arange(n,device=self.R.dev)%P.C
        return self.draw(y,self.R.gen(991,0))

    def prepare(self, k, r):
        model=self.proxy[k]
        own='own' in self.arm
        if self.basis is not None:
            a=self.own_coeff[k] if own else self.coeff
            self.layout.load(model,self.basis.vector(a))
        elif not own:
            self.layout.load(model,self.global_vector)
        if self.arm.startswith('c_'):
            calibrate(model,self.gaussian_calibration())
        else:
            idx=torch.randperm(len(self.R.Y[k]),device=self.R.dev,generator=self.R.gen(881,k))[:4096]
            calibrate(model,self.R.X[k][idx])
        return model

    def proxy_epoch(self,k,r):
        R=self.R; model=self.prepare(k,r)
        base=self.layout.flatten(model).clone()
        own='own' in self.arm
        a=None
        if self.basis is not None:
            a=(self.own_coeff[k] if own else self.coeff).detach().clone().requires_grad_(True)
            opt=torch.optim.AdamW([a],lr=R.lr(r)*100,weight_decay=0.)
            model.eval()
        else:
            opt=torch.optim.AdamW(model.parameters(),lr=R.lr(r),weight_decay=0.)
            model.train() if self.arm.startswith('a_') else model.eval()
        scaler=torch.amp.GradScaler(); g=R.gen(r,k,101)
        torch.manual_seed(int(np.random.SeedSequence([R.a.seed,r,k,19]).generate_state(1)[0]))
        X,Y=R.X[k],R.Y[k]; order=torch.randperm(len(Y),device=R.dev,generator=g)
        total=torch.zeros(3,device=R.dev); applied=skipped=0
        for start in range(0,len(Y),R.a.batch):
            idx=order[start:start+R.a.batch]; x=X[idx].float(); y=Y[idx]
            with torch.autocast('cuda',dtype=torch.float16):
                if self.arm.startswith('c_'):
                    xp=x+R.a.tau*self.sd[y]*torch.randn(x.shape,device=R.dev,generator=g)
                    pos=model(xp)[0].float().gather(1,y[:,None]).squeeze(1)
                    neg=model(self.draw(y,g))[0].float().gather(1,y[:,None]).squeeze(1)
                    weight=1/(P.C*R.ratio_pi[y])
                    lp=(F.softplus(-pos)*weight).mean(); ln=(F.softplus(neg)*weight).mean()
                    loss=lp+ln; parts=torch.stack([loss.detach(),lp.detach(),ln.detach()])
                else:
                    z=self.basis.forward(model,a,x)[0] if a is not None else model(x)[0]
                    loss=F.cross_entropy(z.float(),y); parts=torch.stack([loss.detach(),loss.detach(),loss.detach()*0])
            assert torch.isfinite(loss), ('proxy loss',k,r)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([a] if a is not None else model.parameters(),R.a.clip)
            before=scaler.get_scale(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if scaler.get_scale()<before: skipped+=1
            else: applied+=1; total+=parts
        assert applied>0 and skipped<=16,(k,r,applied,skipped)
        if a is not None:
            assert torch.isfinite(a).all(); self.layout.load(model,self.basis.vector(a.detach()))
            if own: self.own_coeff[k]=a.detach()
            values=(a.detach()-(self.own_coeff[k] if own else self.coeff))*R.n[k]
            kind='coeff'
        else:
            values=(self.layout.flatten(model)-base)*R.n[k]; kind='dense'
            if self.sketch is not None: values=self.sketch.encode(values); kind='sketch'
        finite(model); packet=None
        if not own:
            packet=pack_vector(kind,values,base=self.version,version=self.version+1,count=R.n[k])
            self.record(packet,r,k,'up',kind)
        losses=(total/applied).tolist()
        return packet,dict(client=k,rows=len(Y),applied=applied,skipped=skipped,
                           loss=losses[0],positive_ce=losses[1],negative_ce=losses[2])

    def private_loss(self,k,x,y,g,r):
        R=self.R
        zreal=R.priv[k](x)[0].float()
        real=F.cross_entropy(zreal,y)
        extra=real.detach()*0
        if self.arm.startswith('a_') and r>1:
            teacher=self.teacher
            with torch.no_grad(): q=F.softmax(teacher(x)[0].float()/R.a.temperature,dim=1)
            extra=F.kl_div(F.log_softmax(zreal/R.a.temperature,dim=1),q,reduction='batchmean')*R.a.temperature**2
        elif self.arm=='diag' or self.arm.startswith('c_'):
            sy=torch.arange(R.a.aux_batch,device=R.dev)%P.C
            sx=self.draw(sy,g)
            w=torch.ones(len(sy),device=R.dev)
            if self.arm.startswith('c_') and r>1:
                with torch.no_grad():
                    z=self.teacher(sx)[0]
                    w,_=ratio_weights(z,sy,R.a.weight_cap,
                        R.gen(r,k,313) if self.arm=='c_shuffle' else None)
            extra=(F.cross_entropy(R.priv[k](sx)[0].float(),sy,reduction='none')*w).mean()
        loss=real+R.a.aux_weight*extra
        return loss,torch.stack([real.detach(),extra.detach()])

    def start_round(self):
        if not self.proxy: return
        if 'own' in self.arm: self.aggregate_sum=None; return
        kind='coeff' if self.basis is not None else ('sketch' if self.sketch is not None else 'dense')
        size=self.coeff.numel() if self.basis is not None else (self.residual.numel() if self.sketch is not None else self.layout.size)
        self.aggregate_sum=SecureSum(kind,self.version,size,self.R.dev)

    def aggregate(self,r):
        if not self.proxy or 'own' in self.arm: self.version+=int(bool(self.proxy)); return
        total,n=self.aggregate_sum.finish(); mean=total/n
        self.last_mean=mean.detach().clone()
        if self.basis is not None:
            packet=pack_vector('coeff',self.coeff+mean,base=self.version,version=self.version+1)
            self.coeff=unpack(packet,self.R.dev)['values']
            self.global_vector=self.basis.vector(self.coeff)
        elif self.sketch is not None:
            accumulated=mean+self.residual
            packet,delta=self.sketch.patch(accumulated,self.topk,self.version)
            self.global_vector+=delta
            self.residual=accumulated-self.sketch.encode(delta) if self.arm.endswith('_ef') else torch.zeros_like(accumulated)
            self.diag=dict(residual_norm=float(self.residual.norm()),update_norm=float(delta.norm()))
        else:
            packet=pack_vector('dense',self.global_vector+mean,base=self.version,version=self.version+1,half=True)
            self.global_vector=unpack(packet,self.R.dev)['values']
        assert torch.isfinite(self.global_vector).all()
        self.latest_packet=packet
        for k in range(self.R.K): self.record(packet,r,k,'down','proxy_update')
        self.version+=1

    @torch.no_grad()
    def ratio_diagnostics(self):
        models=self.proxy if 'own' in self.arm else self.proxy[:1]
        n=64 if self.R.a.smoke else 4096
        y=torch.arange(P.C,device=self.R.dev).repeat_interleave(n)
        x=self.draw(y,self.R.gen(771,self.version)); rows=[]
        for k,model in enumerate(models):
            self.prepare(k,self.version)
            logits=self.R.logits(model,x)
            _,raw=ratio_weights(logits,y,self.R.a.weight_cap)
            ess=[]; clipped=[]
            for c in range(P.C):
                w=raw[y==c]; ess.append(float(w.sum().square()/w.square().sum()/len(w)))
                clipped.append(float((w>=self.R.a.weight_cap).float().mean()))
            rows.append(dict(ess_fraction=ess,clip_fraction=clipped))
        return dict(models=rows,ess_fraction=np.mean([v['ess_fraction'] for v in rows],axis=0).tolist())

    def state(self):
        out=dict(global_vector=self.global_vector,version=self.version)
        if self.coeff is not None: out.update(coeff=self.coeff,own_coeff=self.own_coeff)
        if self.residual is not None: out['residual']=self.residual
        if hasattr(self,'last_mean'): out['last_mean']=self.last_mean
        if hasattr(self,'mu'): out.update(mu=self.mu,sd=self.sd,q_version=self.q_version)
        return out
