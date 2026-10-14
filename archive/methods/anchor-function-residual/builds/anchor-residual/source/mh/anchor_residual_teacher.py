"""Common-probe residual fields on public noise anchors; DAGSNet teacher first."""
import math
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.teacher_base import Method as Base
from mh.v2_methods import calibrate, finite
from mh.v2_math import pack_vector, unpack

FAMILIES = {'a': ['r_dense_ce']+[f'r{m}_{mode}' for m in (64,512)
    for mode in ('fl','own','shuffle','classmean','hard','frozen')]}
ARMS = FAMILIES['a']
WIDTHS = (1.,2.,4.,66**.5)


class ResidualSum:
    def __init__(self,dimension,version,device,probe_hash):
        self.dimension,self.version,self.device=dimension,version,device
        self.seen=set(); self.total=torch.zeros(dimension,dtype=torch.float64,device=device)
        self.probe_hash=probe_hash

    def add(self,client,packet):
        assert client not in self.seen
        assert packet[:32]==self.probe_hash, 'residual common-probe hash mismatch'
        values=unpack(packet[32:],self.device,self.version)
        assert values['version']==self.version+1 and len(values['values'])==self.dimension
        self.total+=values['values'].double(); self.seen.add(client)

    def finish(self,clients):
        assert self.seen==set(range(clients)) and clients>=5
        return self.total


class Method(Base):
    def __init__(self,run,initialize=True):
        super().__init__(run,initialize)
        self.dense=self.arm=='r_dense_ce'
        if self.dense: return
        self.M=int(self.arm.split('_')[0][1:]); self.mode=self.arm.split('_')[1]
        if initialize:
            self.make_anchors()
            self.targets=[self.initial_target.clone() for _ in range(run.K if self.own else 1)]
            self.frozen_state={name:value.clone() for name,value in self.global_state.items()}

    def make_anchors(self):
        R=self.R; self.labels=torch.arange(self.M,device=R.dev)%P.C
        g=R.gen(9200,self.M)
        self.anchor_packet=pack_vector('dense',self.draw(self.labels,g).flatten(),half=True)
        self.anchors=unpack(self.anchor_packet,R.dev)['values'].view(self.M,P.D)
        for k in range(R.K): self.record(self.anchor_packet,0,k,'down','anchors')
        mean=self.mu.mean(0)
        self.scale=((self.sd.square()+self.mu.square()).mean(0)-mean.square()).clamp_min(1e-4).sqrt()
        covered=torch.zeros(4,P.C,dtype=torch.float64,device=R.dev)
        counts=R.counts.sum(0)
        for k,(X,Y) in enumerate(zip(R.X,R.Y)):
            local=torch.zeros_like(covered)
            for start in range(0,len(Y),4096):
                distance=self.distance(X[start:start+4096].float())
                nearest=distance.min(1).values
                y=Y[start:start+4096]
                for j,h in enumerate(WIDTHS):
                    mask=nearest<=-2*h*h*math.log(1e-3)
                    local[j]+=torch.bincount(y[mask],minlength=P.C)
            packet=pack_vector('moments',local.flatten()); self.record(packet,0,k,'up','coverage')
            covered+=unpack(packet,R.dev)['values'].view(4,P.C).double()
        fractions=covered/counts.clamp_min(1)[None]
        choice=next((j for j in range(4) if covered[j].sum()/counts.sum()>=.9 and fractions[j].min()>=.8),None)
        assert choice is not None,'kernel coverage gate failed before teacher training'
        self.h=WIDTHS[choice]
        packet=pack_vector('moments',torch.cat([torch.tensor([self.h],device=R.dev),fractions[choice].float()]))
        for k in range(R.K): self.record(packet,0,k,'down','selected_bandwidth')
        self.initial_target=(F.one_hot(self.labels,P.C).float()*.95+.05/P.C).log()
        self.initial_target-=self.initial_target.mean(1,keepdim=True)
        self.diag.update(anchors=self.M,bandwidth=self.h,coverage=fractions[choice].tolist(),
                         coverage_rows=sum(R.n),teacher_steps=20)

    def distance(self,x):
        a,b=x/self.scale,self.anchors/self.scale
        return (a.square().sum(1)[:,None]+b.square().sum(1)[None]-2*a@b.T).clamp_min(0)

    def fit(self,model,target,r):
        calibrate(model,self.gaussian_calibration()); model.eval()
        optimizer=torch.optim.AdamW(model.parameters(),lr=self.R.lr(r),weight_decay=0.)
        total=0.
        for _ in range(20):
            logits=model(self.anchors.float())[0].float()
            centered=logits-logits.mean(1,keepdim=True)
            loss=F.mse_loss(centered,target.float())
            assert torch.isfinite(loss),'nonfinite residual teacher fit'
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); optimizer.zero_grad(set_to_none=True); total+=float(loss.detach())
        finite(model)
        return total/20

    def prior(self):
        if self.dense: super().prior(); return
        # Initial public fit is independent of local labels/records, common across all controls.
        self.fit(self.proxy[0],self.initial_target,1)
        self.global_state={name:value.clone() for name,value in self.proxy[0].state_dict().items()}
        super().prior()
        for model in self.proxy: model.load_state_dict(self.global_state,strict=True)
        self.frozen_state={name:value.clone() for name,value in self.global_state.items()}

    def start_round(self):
        if self.dense: super().start_round(); return
        self.own_residual={}
        state=self.frozen_state if self.mode=='frozen' else self.global_state
        self.probe_hash=hashlib.sha256(self.codec.encode(state,self.version,self.version,half=True)).digest()
        self.aggregate_sum=ResidualSum(self.M*(P.C+1),self.version,self.R.dev,self.probe_hash)

    def proxy_epoch(self,k,r):
        if self.dense: return self.ce_epoch(k,r)
        R=self.R; model=self.prepare(k,r); model.eval()
        if self.mode=='frozen': model.load_state_dict(self.frozen_state,strict=True)
        G=torch.zeros(self.M,P.C,dtype=torch.float64,device=R.dev)
        B=torch.zeros(self.M,dtype=torch.float64,device=R.dev)
        population=R.counts.sum(0).float().clamp_min(1)
        with torch.no_grad():
            for start in range(0,R.n[k],4096):
                x=R.X[k][start:start+4096].float(); y=R.Y[k][start:start+4096]
                kernels=torch.exp(-self.distance(x)/(2*self.h*self.h))
                weighted=kernels/population[y,None]
                residual=F.softmax(model(x)[0].float(),1)-F.one_hot(y,P.C)
                G+=(weighted.T@residual).double(); B+=weighted.sum(0).double()
        values=torch.cat([G.flatten(),B])
        packet=pack_vector('moments',values.float(),base=r-1,version=r,count=R.n[k])
        decoded=unpack(packet,R.dev,r-1)['values'].double()
        probe_hash=hashlib.sha256(self.codec.encode(model.state_dict(),r-1,r-1,half=True)).digest()
        packet=probe_hash+packet
        if not self.own: assert probe_hash==self.probe_hash
        assert torch.isfinite(decoded).all()
        if self.own: self.own_residual[k]=decoded; packet=None
        elif self.mode=='hard': packet=None
        else: self.record(packet,r,k,'up','residual_sum')
        return packet,dict(client=k,rows=R.n[k],applied=0,skipped=0,loss=0.,positive_ce=0.,negative_ce=0.,
            residual_rows=R.n[k],residual_norm=float(G.norm()))

    def update_target(self,target,values):
        G=values[:self.M*P.C].view(self.M,P.C); B=values[self.M*P.C:]
        residual=G/B.clamp_min(1e-12)[:,None]
        if self.mode=='shuffle':
            order=torch.randperm(self.M,generator=torch.Generator().manual_seed(9201+self.M)).to(self.R.dev)
            residual=residual[order]
        elif self.mode=='classmean': residual=residual.mean(0,keepdim=True).expand_as(residual)
        norm=residual.norm(dim=1,keepdim=True)
        delta=-residual*(.25/(norm+1e-12)).clamp_max(1)
        candidate=(target.double()+delta).clamp(-8,8)
        candidate-=candidate.mean(1,keepdim=True)
        self.diag.update(zero_mass=int((B==0).sum()),residual_norm=float(residual.norm()),
                         delta_max_norm=float(delta.norm(dim=1).max()),target_max=float(candidate.abs().max()))
        packet=pack_vector('moments',candidate.float().flatten(),half=True)
        decoded=unpack(packet,self.R.dev)['values'].view(self.M,P.C)
        decoded-=decoded.mean(1,keepdim=True)
        return packet,decoded

    def aggregate(self,r):
        if self.dense: super().aggregate(r); return
        groups=range(self.R.K) if self.own else range(1)
        fit_losses=[]
        for k in groups:
            values=self.own_residual[k] if self.own else (self.aggregate_sum.finish(self.R.K) if self.mode!='hard' else None)
            target=self.targets[k]
            if values is not None:
                packet,target=self.update_target(target,values)
                if not self.own:
                    for cid in range(self.R.K): self.record(packet,r,cid,'down','target_logits')
            self.targets[k]=target
            model=self.prepare(k,r)
            fit_losses.append(self.fit(model,target,r))
            snapshot,decoded=self.finish_snapshot(model,r)
            if not self.own:
                self.global_state=decoded; self.latest_packet=snapshot
                for cid in range(self.R.K):
                    self.record(snapshot,r,cid,'down','teacher_state')
                    self.proxy[cid].load_state_dict(decoded,strict=True)
                self.global_vector=self.layout.flatten(model).clone()
        self.version=r; self.diag['teacher_fit_mse']=float(np.mean(fit_losses))
        self.diag['teacher_applied']=20*len(fit_losses)

    def state(self):
        state=super().state()
        if not self.dense:
            state.update(M=self.M,mode=self.mode,anchors=self.anchors,labels=self.labels,scale=self.scale,
                anchor_packet=self.anchor_packet,initial_target=self.initial_target,targets=self.targets,
                h=self.h,frozen_state=self.frozen_state)
        return state
