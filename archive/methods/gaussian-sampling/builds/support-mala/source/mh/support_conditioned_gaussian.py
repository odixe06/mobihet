"""One-shot aggregate support moments, fresh local analytic draws; DAGSNet IDS."""
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.v2_math import Layout,SecureSum,pack_vector,unpack
from mh.v2_methods import Method as Base
from mh.v4_schema import ATOM0,ATOM1,FIRST,PROFILES,STRUCTURAL


class Method(Base):
    def __init__(self,run,initialize=True):
        self.R,self.arm,self.diag=run,run.a.arm,{}
        self.proxy=[]; self.role='private'; self.layout=Layout(run.init)
        self.global_vector=self.layout.flatten(run.init).clone(); self.version=0
        self.coeff=self.residual=None
        # Fixture stores standardized values fp16; equality must use those same atoms.
        self.atom0=torch.tensor(ATOM0,device=run.dev,dtype=torch.float64).half().float()
        self.atom1=torch.tensor(ATOM1,device=run.dev,dtype=torch.float64).half().float()
        if initialize: self.statistics()

    def local_stats(self,X,Y):
        x=X.float(); zero=x==self.atom0; one=x==self.atom1
        binary=[FIRST]+sum(PROFILES,[])
        assert (zero[:,binary]|one[:,binary]).all(),'binary schema violation'
        assert zero[one[:,FIRST]][:,STRUCTURAL].all(),'first-session structural schema violation'
        for indices in PROFILES: assert (one[:,indices].sum(1)<=1).all(),'profile multi-hot'
        result=torch.zeros(P.C,1+3*P.D,device=self.R.dev)
        for y in range(P.C):
            values=x[Y==y]; iszero=zero[Y==y]; other=values.masked_fill(iszero,0)
            result[y,0]=len(values); result[y,1:1+P.D]=iszero.sum(0)
            result[y,1+P.D:1+2*P.D]=other.sum(0)
            result[y,1+2*P.D:]=other.square().sum(0)
        return result

    def parameters(self,m):
        n=m[:,0:1]; atoms=m[:,1:1+P.D]; count=(n-atoms).clamp_min(1)
        sums=m[:,1+P.D:1+2*P.D]; squares=m[:,1+2*P.D:]
        nonmu=sums/count; nonsd=(squares/count-nonmu.square()).clamp_min(1e-4).sqrt()
        mu=(sums+atoms*self.atom0)/n.clamp_min(1)
        sd=((squares+atoms*self.atom0.square())/n.clamp_min(1)-mu.square()).clamp_min(1e-4).sqrt()
        rho=atoms/n.clamp_min(1)
        return torch.stack([mu,sd,rho,nonmu,nonsd])

    def statistics(self):
        R=self.R; agg=SecureSum('moments',0,P.C*(1+3*P.D),R.dev); own=[]
        for k,(X,Y) in enumerate(zip(R.X,R.Y)):
            stats=self.local_stats(X,Y); own.append(self.parameters(stats).half().float())
            packet=pack_vector('moments',stats.flatten(),count=R.n[k])
            if self.arm!='s_own': self.record(packet,0,k,'up','support_moments'); agg.add(k,packet)
        if self.arm=='s_own':
            self.tables=torch.stack(own); return
        m,_=agg.finish(); table=self.parameters(m.reshape(P.C,-1))
        packet=pack_vector('moments',table.flatten(),half=True)
        self.support_packet=packet
        self.tables=unpack(packet,R.dev)['values'].reshape(5,P.C,P.D)
        for k in range(R.K): self.record(packet,0,k,'down','support_table')

    def draw_support(self,labels,g,k):
        table=self.tables[k] if self.arm=='s_own' else self.tables
        mu,sd,rho,nonmu,nonsd=table
        if self.arm=='s_diag': return mu[labels]+sd[labels]*torch.randn(len(labels),P.D,device=self.R.dev,generator=g)
        rho=rho.clone()
        if self.arm=='s_shuffle':
            binary=[FIRST]+sum(PROFILES,[])
            other=[j for j in range(P.D) if j not in binary+STRUCTURAL]
            for indices in (STRUCTURAL,other,*PROFILES):
                rho[:,indices]=rho[:,indices[1:]+indices[:1]]
        x=nonmu[labels]+nonsd[labels]*torch.randn(len(labels),P.D,device=self.R.dev,generator=g)
        mask=torch.rand(x.shape,device=self.R.dev,generator=g)<rho[labels]
        x=torch.where(mask,self.atom0,x)
        first=torch.rand(len(labels),device=self.R.dev,generator=g)<(1-rho[labels,FIRST])
        x[:,FIRST]=torch.where(first,self.atom1[FIRST],self.atom0[FIRST])
        for indices in PROFILES:
            probs=(1-rho[:,indices]).clamp(0,1)
            unknown=(1-probs.sum(1,keepdim=True)).clamp_min(0)
            p=torch.cat([unknown,probs],dim=1); p=p/p.sum(1,keepdim=True).clamp_min(1e-8)
            category=torch.multinomial(p[labels],1,generator=g).squeeze(1)
            x[:,indices]=self.atom0[indices]
            for offset,j in enumerate(indices,1): x[:,j]=torch.where(category==offset,self.atom1[j],self.atom0[j])
        if self.arm!='s_marginal':
            pfirst=1-rho[:,FIRST:FIRST+1]
            conditional=((rho[:,STRUCTURAL]-pfirst)/(1-pfirst).clamp_min(1e-8)).clamp(0,1)
            mask=torch.rand(len(labels),len(STRUCTURAL),device=self.R.dev,generator=g)<conditional[labels]
            fresh=nonmu[labels][:,STRUCTURAL]+nonsd[labels][:,STRUCTURAL]*torch.randn(mask.shape,device=self.R.dev,generator=g)
            x[:,STRUCTURAL]=torch.where(mask|first[:,None],self.atom0[STRUCTURAL],fresh)
        assert torch.isfinite(x).all()
        return x

    def private_loss(self,k,x,y,g,r):
        R=self.R; real=F.cross_entropy(R.priv[k](x)[0].float(),y)
        labels=torch.arange(R.a.aux_batch,device=R.dev)%P.C
        synthetic=self.draw_support(labels,g,k)
        extra=F.cross_entropy(R.priv[k](synthetic)[0].float(),labels)
        return real+R.a.aux_weight*extra,torch.stack([real.detach(),extra.detach()])

    def state(self):
        result=dict(global_vector=self.global_vector,version=self.version,tables=self.tables,
                    atom0=self.atom0,atom1=self.atom1)
        if hasattr(self,'support_packet'): result['support_packet']=self.support_packet
        return result
