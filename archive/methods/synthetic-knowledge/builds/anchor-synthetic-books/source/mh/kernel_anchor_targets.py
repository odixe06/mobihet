"""Class-kernel aggregate soft targets on public noise anchors; local DAGSNet teachers."""
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.v2_math import Layout,SecureSum,pack_vector,unpack
from mh.v2_methods import Method as Base,calibrate,finite

WIDTHS=(1.,2.,4.,66**.5)
STEPS=20


class Method(Base):
    def __init__(self,run,initialize=True):
        self.R,self.arm,self.diag=run,run.a.arm,{}
        self.role='proxy'; self.proxy=[copy.deepcopy(run.init) for _ in range(run.K)]
        self.layout=Layout(run.init); self.global_vector=self.layout.flatten(run.init).clone()
        self.version=0; self.coeff=self.residual=None
        self.M=32 if 'small' in self.arm else 512
        if initialize:
            self.make_gaussian(); self.make_targets()

    def decode_targets(self,target):
        packet=pack_vector('moments',target.flatten(),half=True)
        decoded=unpack(packet,self.R.dev)['values'].reshape(self.M,P.C)
        assert torch.isfinite(decoded).all() and (decoded>=0).all()
        return packet,decoded/decoded.sum(1,keepdim=True).clamp_min(1e-12)

    def make_targets(self):
        R=self.R; self.labels=torch.arange(self.M,device=R.dev)%P.C
        noise=torch.from_numpy(np.random.default_rng(4902+self.M).standard_normal((self.M,P.D)).astype('float32')).to(R.dev)
        self.anchor_packet=pack_vector('dense',(self.mu[self.labels]+self.sd[self.labels]*noise).flatten(),half=True)
        self.anchors=unpack(self.anchor_packet,R.dev)['values'].reshape(self.M,P.D)
        for k in range(R.K): self.record(self.anchor_packet,0,k,'down','anchors')
        mean=self.mu.mean(0)
        self.scale=((self.sd.square()+self.mu.square()).mean(0)-mean.square()).clamp_min(1e-4).sqrt()
        b=self.anchors/self.scale; local_h=[]; coverage=SecureSum('moments',0,P.C*5,R.dev)
        for k,(X,Y) in enumerate(zip(R.X,R.Y)):
            counts=torch.bincount(Y,minlength=P.C).float(); covered=torch.zeros(4,P.C,device=R.dev)
            H=torch.zeros(4,self.M,P.C,device=R.dev)
            for start in range(0,len(Y),4096):
                x=X[start:start+4096].float()/self.scale; y=Y[start:start+4096]
                distance=(x.square().sum(1)[:,None]+b.square().sum(1)[None]-2*x@b.T).clamp_min(0)
                closest=distance.min(1).values
                for j,h in enumerate(WIDTHS):
                    mask=closest<=-2*h*h*math.log(1e-3); kernels=(-distance/(2*h*h)).exp()
                    for c in range(P.C):
                        covered[j,c]+=mask[y==c].sum(); H[j,:,c]+=kernels[y==c].sum(0)
            local_h.append(H)
            values=torch.cat([counts[None],covered],dim=0)
            packet=pack_vector('moments',values.flatten(),count=R.n[k])
            self.record(packet,0,k,'up','coverage'); coverage.add(k,packet)
        aggregate,_=coverage.finish(); aggregate=aggregate.reshape(5,P.C)
        counts=aggregate[0]; assert (counts>0).all()
        fractions=aggregate[1:]/counts
        choice=None
        for j,h in enumerate(WIDTHS):
            if aggregate[j+1].sum()/counts.sum()>=.9 and fractions[j].min()>=.8:
                choice=j; self.h=h; break
        assert choice is not None,'kernel coverage gate failed before teacher training'
        self.coverage_packet=pack_vector('moments',torch.cat([torch.tensor([self.h],device=R.dev),fractions[choice]]))
        for k in range(R.K): self.record(self.coverage_packet,0,k,'down','selected_bandwidth')
        self.coverage_per_class=fractions[choice]
        mode=self.arm.rsplit('_',1)[1]
        self.own_targets=[]; self.target_packet=None
        if mode=='own':
            for k,H in enumerate(local_h):
                values=H[choice]/R.counts[k].float().clamp_min(1)[None]
                total=values.sum(1,keepdim=True)
                target=torch.where(total>0,values/total.clamp_min(1e-30),torch.full_like(values,1/P.C))
                _,decoded=self.decode_targets(target); self.own_targets.append(decoded)
        else:
            if mode=='gauss': target=F.one_hot(self.labels,P.C).float()
            else:
                agg=SecureSum('moments',0,self.M*P.C,R.dev)
                for k,H in enumerate(local_h):
                    packet=pack_vector('moments',H[choice].flatten(),count=R.n[k])
                    self.record(packet,0,k,'up','label_kernel'); agg.add(k,packet)
                values,_=agg.finish(); values=values.reshape(self.M,P.C)/counts[None]
                total=values.sum(1,keepdim=True)
                target=torch.where(total>0,values/total.clamp_min(1e-30),torch.full_like(values,1/P.C))
                if mode=='shuffle':
                    perm=torch.randperm(self.M,generator=torch.Generator().manual_seed(4903+self.M)).to(R.dev)
                    target=target[perm]
            self.target_packet,self.target=self.decode_targets(target)
            if mode!='gauss':
                for k in range(R.K): self.record(self.target_packet,0,k,'down','soft_targets')
        self.diag=dict(anchors=self.M,bandwidth=self.h,coverage_per_class=self.coverage_per_class.tolist(),
            queried_real_rows=sum(R.n),target_mode=mode,teacher_steps_per_round=STEPS)

    def prepare(self,k,r):
        model=self.proxy[k]; calibrate(model,self.gaussian_calibration()); return model

    def start_round(self): self.aggregate_sum=None

    def proxy_epoch(self,k,r):
        R=self.R; model=self.prepare(k,r); model.eval()
        target=self.own_targets[k] if self.own_targets else self.target
        opt=torch.optim.AdamW(model.parameters(),lr=R.lr(r),weight_decay=0.)
        scaler=torch.amp.GradScaler(); total=0.; applied=skipped=0
        for _ in range(STEPS):
            with torch.autocast('cuda',dtype=torch.float16):
                logits=model(self.anchors)[0].float()
                loss=-(target*F.log_softmax(logits,dim=1)).sum(1).mean()
            assert torch.isfinite(loss),('proxy loss',k,r)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),R.a.clip)
            before=scaler.get_scale(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if scaler.get_scale()<before: skipped+=1
            else: applied+=1; total+=float(loss.detach())
        assert applied>0 and skipped<=16; finite(model)
        return None,dict(client=k,rows=STEPS*self.M,applied=applied,skipped=skipped,
            loss=total/applied,positive_ce=total/applied,negative_ce=0.,anchor_batch=self.M,real_stat_rows=R.n[k])

    def aggregate(self,r): pass  # one-shot statistics; no learned parameter synchronization

    def state(self):
        return dict(global_vector=self.global_vector,version=self.version,mu=self.mu,sd=self.sd,q_version=self.q_version,
            M=self.M,anchors=self.anchors,labels=self.labels,anchor_packet=self.anchor_packet,scale=self.scale,
            coverage_packet=self.coverage_packet,coverage_per_class=self.coverage_per_class,h=self.h,
            own_targets=self.own_targets,target_packet=self.target_packet,
            target=self.target if hasattr(self,'target') else None,diag=self.diag)
