"""Teacher-only dense FL plumbing with paid coherent DAGSNet/BN snapshots."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.v2_methods import Method as Prior, calibrate, finite
from mh.v2_math import Layout
from mh.model_packet import Codec, Sum


class Method(Prior):
    def __init__(self, run, initialize=True):
        self.R, self.arm, self.diag = run, run.a.arm, {}
        self.role, self.version = 'proxy', 0
        self.own = self.arm.endswith('_own')
        self.proxy = [copy.deepcopy(run.init) for _ in range(run.K)]
        self.layout = Layout(run.init)
        self.global_vector = self.layout.flatten(run.init).clone()
        self.global_state = {name:value.clone() for name,value in run.init.state_dict().items()}
        self.codec = Codec(run.init,f'teacher:{self.arm}:{run.a.seed}')
        if initialize:
            assert ((run.counts>0).sum(0)>=5).all(), 'teacher population class cohort gate'
            self.make_gaussian()

    def prior(self):
        super().prior()
        packet = self.codec.encode(self.global_state,0,0,half=True)
        self.global_state,_,_ = self.codec.decode(packet,self.R.dev)
        if not self.own:
            for k in range(self.R.K):
                self.record(packet,0,k,'down','teacher_bootstrap')

    def prepare(self,k,r):
        if not self.own:
            self.proxy[k].load_state_dict(self.global_state,strict=True)
        return self.proxy[k]

    def start_round(self):
        if not self.own:
            self.aggregate_sum = Sum(self.codec,self.version,self.R.dev)

    def finish_snapshot(self,model,r):
        finite(model)
        packet = self.codec.encode(model.state_dict(),r-1,r,half=True)
        decoded,_,_ = self.codec.decode(packet,self.R.dev,r-1)
        model.load_state_dict(decoded,strict=True)
        return packet,decoded

    def aggregate(self,r):
        if not self.own:
            state = self.aggregate_sum.finish(self.R.K)
            packet = self.codec.encode(state,r-1,r,half=True)
            self.global_state,_,_ = self.codec.decode(packet,self.R.dev,r-1)
            for k in range(self.R.K):
                self.record(packet,r,k,'down','teacher_state')
                self.proxy[k].load_state_dict(self.global_state,strict=True)
            self.global_vector = self.layout.flatten(self.proxy[0]).clone()
            self.latest_packet = packet
        else:
            for model in self.proxy:
                self.finish_snapshot(model,r)
        self.version = r

    def state(self):
        return dict(global_vector=self.global_vector,global_state=self.global_state,version=self.version,
                    mu=self.mu,sd=self.sd,q_version=self.q_version,diag=self.diag)

    def ce_epoch(self,k,r,aug=False):
        R = self.R; model = self.prepare(k,r)
        calibrate(model,self.gaussian_calibration()); model.eval()
        optimizer = torch.optim.AdamW(model.parameters(),lr=R.lr(r),weight_decay=0.)
        scaler = torch.amp.GradScaler(); generator = R.gen(r,k,510)
        order = torch.randperm(R.n[k],device=R.dev,generator=generator)
        total = torch.zeros(3,device=R.dev); applied = skipped = 0
        for start in range(0,len(order),R.a.batch):
            index = order[start:start+R.a.batch]; x,y = R.X[k][index].float(),R.Y[k][index]
            with torch.autocast('cuda',dtype=torch.float16):
                ce = F.cross_entropy(model(x)[0].float(),y)
                extra = ce.detach()*0
                if aug:
                    smooth = x+.05*self.sd[y]*torch.randn(x.shape,device=R.dev,generator=generator)
                    extra = .1*F.cross_entropy(model(smooth)[0].float(),y)
                loss = ce+extra
            assert torch.isfinite(loss),('teacher CE loss',k,r)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),R.a.clip)
            before=scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale()<before: skipped+=1
            else: applied+=1; total+=torch.stack([loss.detach(),ce.detach(),extra.detach()])
        assert applied>0 and skipped<=16; finite(model)
        packet=self.codec.encode(model.state_dict(),r-1,r)
        if not self.own: self.record(packet,r,k,'up','teacher_state')
        values=(total/applied).tolist()
        return None if self.own else packet,dict(client=k,rows=R.n[k],applied=applied,skipped=skipped,
            loss=values[0],positive_ce=values[1],negative_ce=values[2])
