"""Normalizable DAGSNet joint energy with a common Gaussian mixture base and MH."""
import numpy as np
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.teacher_base import Method as Base
from mh.v2_methods import calibrate, finite

FAMILIES={'e':['e_ce_raw','e_ce_matched','e_ce_aug','e_fl','e_own','e_shuffle','e_q']}
ARMS=FAMILIES['e']


class Method(Base):
    def __init__(self,run,initialize=True):
        super().__init__(run,initialize)
        self.permutation=torch.from_numpy(np.random.default_rng(10102).permutation(P.D)).to(run.dev)
        self.chains={}; self.sampler={}

    def proposal(self,n,g):
        component=torch.randint(P.C,(n,),device=self.R.dev,generator=g)
        return self.draw(component,g)

    def coordinates(self,x):
        return x[:,self.permutation] if self.arm=='e_shuffle' else x

    @torch.no_grad()
    def transition(self,model,current,g):
        proposal=self.proposal(len(current),g)
        # Both energies use current parameters, including after an optimizer update.
        old=torch.logsumexp(model(self.coordinates(current).float())[0].float(),1)
        new=torch.logsumexp(model(self.coordinates(proposal).float())[0].float(),1)
        assert torch.isfinite(old).all() and torch.isfinite(new).all(),'nonfinite MH energy'
        accepted=torch.rand(len(current),device=self.R.dev,generator=g).log()<torch.minimum(new-old,torch.zeros_like(old))
        result=torch.where(accepted[:,None],proposal,current)
        jump=(result-current).square().sum(1).sqrt()
        return result,accepted,jump

    def start_round(self):
        super().start_round(); self.sampler={}

    def proxy_epoch(self,k,r):
        if self.arm=='e_ce_raw': return self.ce_epoch(k,r)
        R=self.R; model=self.prepare(k,r)
        calibrate(model,self.gaussian_calibration()); model.eval()
        optimizer=torch.optim.AdamW(model.parameters(),lr=R.lr(r),weight_decay=0.)
        scaler=torch.amp.GradScaler(); g=R.gen(r,k,10001)
        order=torch.randperm(R.n[k],device=R.dev,generator=g)
        chains=self.proposal(256,g); dwell=torch.zeros(256,dtype=torch.int64,device=R.dev)
        accepted_count=transitions=0; jump_sum=0.; max_dwell=0
        if self.arm!='e_q':
            for _ in range(20):
                chains,accepted,jump=self.transition(model,chains,g)
                dwell=torch.where(accepted,torch.zeros_like(dwell),dwell+1)
                accepted_count+=int(accepted.sum()); transitions+=len(chains)
                jump_sum+=float(jump.sum()); max_dwell=max(max_dwell,int(dwell.max()))
        totals=torch.zeros(4,device=R.dev); applied=skipped=0
        for start in range(0,len(order),R.a.batch):
            index=order[start:start+R.a.batch]; x,y=R.X[k][index].float(),R.Y[k][index]
            if self.arm=='e_q': chains=self.proposal(256,g)
            else:
                chains,accepted,jump=self.transition(model,chains,g)
                dwell=torch.where(accepted,torch.zeros_like(dwell),dwell+1)
                accepted_count+=int(accepted.sum()); transitions+=len(chains)
                jump_sum+=float(jump.sum()); max_dwell=max(max_dwell,int(dwell.max()))
            smooth=x+.05*self.sd[y]*torch.randn(x.shape,device=R.dev,generator=g)
            assert torch.isfinite(smooth.half()).all() and torch.isfinite(chains.half()).all(),'energy input fp16 overflow'
            with torch.autocast('cuda',dtype=torch.float16):
                ce=F.cross_entropy(model(x)[0].float(),y)
            positive=model(self.coordinates(smooth))[0].float()
            negative=model(self.coordinates(chains.detach()))[0].float()
            positive_energy=torch.logsumexp(positive,1).mean()
            negative_energy=torch.logsumexp(negative,1).mean()
            if self.arm=='e_ce_aug':
                extra=.1*F.cross_entropy(positive,y)+negative_energy*0
            elif self.arm=='e_ce_matched':
                extra=(positive_energy+negative_energy)*0
            else:
                extra=.1*(-positive_energy+negative_energy)
            loss=ce+extra
            assert torch.isfinite(loss),('joint energy loss',k,r)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),R.a.clip)
            before=scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale()<before: skipped+=1
            else:
                assert torch.isfinite(norm),'nonfinite applied energy gradient'
                applied+=1; totals+=torch.stack([loss.detach(),ce.detach(),positive_energy.detach(),negative_energy.detach()])
        assert applied>0 and skipped<=16; finite(model)
        self.chains[k]=chains.detach()
        self.sampler[k]=dict(acceptance=accepted_count/max(transitions,1),movement=accepted_count/max(transitions,1),
                            jump_distance=jump_sum/max(transitions,1),max_dwell=max_dwell,transitions=transitions)
        packet=self.codec.encode(model.state_dict(),r-1,r)
        if not self.own: self.record(packet,r,k,'up','teacher_state')
        values=(totals/applied).tolist()
        return None if self.own else packet,dict(client=k,rows=R.n[k],applied=applied,skipped=skipped,
            loss=values[0],positive_ce=values[1],negative_ce=values[0]-values[1],
            energy_positive=values[2],energy_negative=values[3],sampler=self.sampler[k])

    def aggregate(self,r):
        super().aggregate(r)
        if self.sampler:
            self.diag.update(sampler_acceptance=float(np.mean([v['acceptance'] for v in self.sampler.values()])),
                sampler_jump=float(np.mean([v['jump_distance'] for v in self.sampler.values()])),
                sampler_max_dwell=max(v['max_dwell'] for v in self.sampler.values()))
        if r in (1,5,10,20) and self.arm in ('e_fl','e_shuffle'):
            self.diag['sampler_probe']=self.probe(r)

    @torch.no_grad()
    def probe(self,r):
        model=self.prepare(0,r); model.eval(); series=[]; moved=[]
        for replica in range(2):
            g=self.R.gen(10002,r,replica); x=self.proposal(64,g)
            history=[]; movement=torch.zeros(64,dtype=torch.int64,device=self.R.dev)
            for _ in range(100):
                x,accepted,_=self.transition(model,x,g); movement+=accepted
                history.append(torch.logsumexp(model(self.coordinates(x))[0].float(),1))
            series.append(torch.stack(history).cpu().numpy()); moved.append(movement.cpu().numpy())
        pooled=np.concatenate([a.flatten() for a in series]); sd=float(pooled.std())
        difference=abs(float(series[0].mean()-series[1].mean()))
        normalized=difference/max(sd,1e-12)
        trajectory=np.concatenate(series,axis=1); centered=trajectory-trajectory.mean(0,keepdims=True)
        variance=float(np.mean(centered**2)); correlations={}
        for lag in (1,10,20):
            correlations[str(lag)]=float(np.mean(centered[:-lag]*centered[lag:])/max(variance,1e-12))
        positive=[]
        for lag in range(1,21):
            rho=float(np.mean(centered[:-lag]*centered[lag:])/max(variance,1e-12))
            if rho<=0: break
            positive.append(rho)
        ess=100/(1+2*sum(positive))
        movement=float(np.concatenate(moved).sum()/(128*100)); frozen=int((np.concatenate(moved)==0).sum())
        passed=movement>=.05 and frozen==0 and normalized<=.1
        return dict(two_noise_init=2,chains_per_init=64,transitions=100,movement=movement,
                    frozen_chains=frozen,mean_energy_difference_sd=normalized,autocorrelation=correlations,
                    energy_ess_per_chain=ess,proposed_gate=bool(passed),limited_finite_diagnostic=True)

    def state(self):
        state=super().state(); state.update(permutation=self.permutation,chains=self.chains,sampler=self.sampler)
        return state
