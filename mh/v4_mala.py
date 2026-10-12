"""Local Gaussian-base Metropolis adjusted transport; no records leave client."""
import math
import torch
import torch.nn.functional as F
from mh.v2_methods import Method as Base
from mh import pilot as P

ETA=.01
BURN=20


def transition(z,logp,score,generator,eta=ETA):
    proposal=z+eta*score+math.sqrt(2*eta)*torch.randn(z.shape,device=z.device,generator=generator)
    return proposal


def accept(z,proposal,logp,score,new_logp,new_score,generator,eta=ETA):
    forward=(proposal.double()-z.double()-eta*score.double()).square().sum(1)
    reverse=(z.double()-proposal.double()-eta*new_score.double()).square().sum(1)
    loga=new_logp.double()-logp.double()+(forward-reverse)/(4*eta)
    take=torch.rand(len(z),device=z.device,generator=generator).double().clamp_min(1e-30).log()<loga
    return (torch.where(take[:,None],proposal,z).detach(),
            torch.where(take,new_logp,logp).detach(),
            torch.where(take[:,None],new_score,score).detach(),take)


def potential(model,z,labels,mu,sd,cap=10.,permutation=None):
    """FP32 input derivative, frozen DAGSNet parameters/BN/dropout, Gaussian base."""
    assert not model.training
    previous=[p.requires_grad for p in model.parameters()]
    try:
        model.requires_grad_(False)
        with torch.enable_grad(),torch.autocast('cuda',enabled=False):
            variable=z.detach().float().requires_grad_(True)
            addressed=variable if permutation is None else variable[:,permutation]
            x=mu[labels]+sd[labels]*addressed
            ratio=model(x)[0].float().gather(1,labels[:,None]).squeeze(1).clamp(-10.,math.log(cap))
            logp=-.5*variable.square().sum(1)+ratio
            score,=torch.autograd.grad(logp.sum(),variable)
        assert torch.isfinite(logp).all() and torch.isfinite(score).all()
        return logp.detach(),score.detach()
    finally:
        for parameter,required in zip(model.parameters(),previous): parameter.requires_grad_(required)


class Method(Base):
    def __init__(self,run,initialize=True):
        super().__init__(run,initialize)
        self.chains={}; self.chain_round={}; self.mala_stats={}
        self.permutation=torch.randperm(P.D,device='cpu',generator=torch.Generator().manual_seed(4602)).to(run.dev)

    def start_round(self):
        super().start_round()
        self.mala_stats={}

    def step(self,z,labels,generator):
        perm=self.permutation if self.arm=='c_mala_shuffle' else None
        lp,grad=potential(self.teacher,z,labels,self.mu,self.sd,self.R.a.weight_cap,perm)
        proposal=transition(z,lp,grad,generator)
        nlp,ngrad=potential(self.teacher,proposal,labels,self.mu,self.sd,self.R.a.weight_cap,perm)
        next_z,_,_,taken=accept(z,proposal,lp,grad,nlp,ngrad,generator)
        return next_z,int(taken.sum()),float((next_z-z).square().sum(1).mean())

    def private_loss(self,k,x,y,g,r):
        if r==1: return super().private_loss(k,x,y,g,r)
        R=self.R; labels=torch.arange(R.a.aux_batch,device=R.dev)%P.C
        accepted=attempted=0; esjd=0.
        with torch.autocast('cuda',enabled=False):
            if self.chain_round.get(k)!=r:
                z=torch.randn(len(labels),P.D,device=R.dev,generator=g)
                for _ in range(BURN):
                    z,taken,jump=self.step(z,labels,g); accepted+=taken; attempted+=len(z); esjd+=jump
                self.chain_round[k]=r
            else: z=self.chains[k]
            z,taken,jump=self.step(z,labels,g); accepted+=taken; attempted+=len(z); esjd+=jump
            self.chains[k]=z.detach()
            synthetic=(self.mu[labels]+self.sd[labels]*z).detach()
        stats=self.mala_stats.setdefault(k,dict(accepted=0,attempted=0,transitions=0,esjd_sum=0.))
        stats['accepted']+=accepted; stats['attempted']+=attempted
        stats['transitions']+=attempted//len(labels); stats['esjd_sum']+=esjd
        real=F.cross_entropy(R.priv[k](x)[0].float(),y)
        extra=F.cross_entropy(R.priv[k](synthetic)[0].float(),labels)
        return real+R.a.aux_weight*extra,torch.stack([real.detach(),extra.detach()])

    def aggregate(self,r):
        super().aggregate(r)
        self.diag.update(mala=dict(eta=ETA,burn_in=BURN,transitions_per_batch=1,
            clients=[dict(client=k,**v,acceptance=v['accepted']/v['attempted'],
                esjd=v['esjd_sum']/v['transitions']) for k,v in sorted(self.mala_stats.items())]))

    def state(self):
        return dict(super().state(),chains=self.chains,chain_round=self.chain_round,permutation=self.permutation)
