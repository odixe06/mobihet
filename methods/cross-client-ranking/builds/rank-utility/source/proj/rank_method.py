"""Locked block ranking surrogate, cohort moments and orthogonal NNLS class-CE control."""
import struct
import time
import zlib
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from mh.v2_methods import Method as Prior,finite
from mh.v8_hybrid import Method as Reference,private_epoch as reference_epoch
from rank_math import C,encode_up,decode_up,aggregate,decode_down,gradient

ARMS=['cr_local','cr_own','cr_fl','cr_shuffle','cr_fixed','cr_classweight','cr_gaussian','cr_dml']
FRAME=struct.Struct('<4sII16sI')


def frame(value,r,identity):return FRAME.pack(b'CRU1',r,len(value),identity,zlib.crc32(value))+value


def unframe(wire,r,identity):
    magic,round_,n,key,crc=FRAME.unpack_from(wire);value=wire[FRAME.size:]
    assert (magic,round_,key)==(b'CRU1',r,identity) and len(value)==n and zlib.crc32(value)==crc
    return value


def nnls_class(prob,y,target):
    # True-class CE columns have disjoint row blocks, hence orthogonal NNLS is exact.
    ce=prob-np.eye(C)[y];beta=np.zeros(C)
    for c in range(C):
        x=ce[y==c];t=target[y==c];denom=float((x*x).sum())
        if denom>0:beta[c]=max(0.,float((x*t).sum())/denom)
    return np.clip(beta,0,4)


def rank_loss(prob,y,means,coefficient):
    positive=F.one_hot(y,C).bool()
    square=torch.where(positive,(prob-1-means[1]).square(),(prob+1-means[0]).square())
    return (square*torch.where(positive,coefficient[0],coefficient[1])).sum(1).mean()


class Method:
    def __init__(self,run,initialize=True):
        self.R=run;self.arm=run.a.arm;self.role='private';self.version=0;self.diag={};self.proxy=[];self.reference=None
        self.identity=bytes.fromhex(run.fingerprint)[:16]
        self.means=torch.zeros(run.K,2,C,device=run.dev);self.coefficient=torch.zeros_like(self.means)
        self.support=torch.zeros(run.K,C,dtype=torch.bool,device=run.dev)
        self.beta=torch.zeros(run.K,C,device=run.dev);self.up=[];self.wire=b''
        if self.arm in ('cr_gaussian','cr_dml'):
            holder=SimpleNamespace(**vars(run));holder.a=SimpleNamespace(**vars(run.a))
            holder.a.arm='h_gaussian' if self.arm=='cr_gaussian' else 'h_dml'
            self.reference=Reference(holder,initialize=initialize);self.reference.R=run;self.proxy=self.reference.proxy

    def record(self,wire,r,k,direction,kind):self.R.bytes.append((r,k,direction,kind,len(wire)))

    def prior(self):
        if self.reference:self.reference.prior()
        else:Prior.prior(self)
        self.R.log_pi=self.R.true_log_pi.clone()
        if self.arm not in ('cr_local','cr_gaussian','cr_dml'):
            dictionary=frame(b'softmaxT1;margin1;lambda.25;K16;common32;global128;contributors5;sample1729;NNLSclip0,4',0,self.identity)
            for k in range(self.R.K):self.record(dictionary,0,k,'down','rank_dictionary')

    def start_round(self):
        if self.reference:
            self.reference.start_round()
            if hasattr(self.reference,'aggregate_sum'):self.aggregate_sum=self.reference.aggregate_sum

    def prepare(self,k,r):return self.reference.prepare(k,r)

    def proxy_epoch(self,k,r):return self.reference.proxy_epoch(k,r)

    @torch.no_grad()
    def scan(self,r):
        started=time.monotonic();up=[]
        for k,model in enumerate(self.R.priv):
            model.eval();x,y=self.R.X[k],self.R.Y[k];n=np.bincount(y.cpu().numpy(),minlength=C).astype('uint64')
            counts=np.stack([n,len(y)-n]);sums=np.zeros((4,C))
            for start in range(0,len(y),4096):
                p=model(x[start:start+4096].float())[0].float().softmax(1).double().cpu().numpy()
                yy=y[start:start+4096].cpu().numpy();positive=np.arange(C)[None,:]==yy[:,None]
                sums[0]+=(p*positive).sum(0);sums[1]+=(p*~positive).sum(0)
                sums[2]+=(p*p*positive).sum(0);sums[3]+=(p*p*~positive).sum(0)
            framed=frame(encode_up(counts,sums,self.identity),r,self.identity)
            up.append(unframe(framed,r,self.identity));self.record(framed,r,k,'up','rank_moments')
        inner,cohort=aggregate(up,self.identity);wire=frame(inner,r,self.identity)
        pooled,n_global,eligible,total=decode_down(unframe(wire,r,self.identity),self.identity)
        for k in range(self.R.K):
            n,s,_=decode_up(up[k],self.identity);valid=eligible&(n>=32).all(0)
            means=np.divide(s[:2],n,out=np.zeros((2,C)),where=n>0) if self.arm=='cr_own' else pooled.copy()
            counts,N=(n,int(n[0].sum())) if self.arm=='cr_own' else (n_global,total)
            if self.arm=='cr_shuffle':means=np.roll(means,1,axis=1)
            coef=np.divide(.25*N/C,counts,out=np.zeros((2,C)),where=counts>0)*valid
            self.means[k]=torch.from_numpy(means).to(self.R.dev,dtype=torch.float32)
            self.coefficient[k]=torch.from_numpy(coef).to(self.R.dev,dtype=torch.float32)
            self.support[k]=torch.from_numpy(valid).to(self.R.dev)
            self.record(wire,r,k,'down','rank_snapshot')
            if self.arm=='cr_classweight':
                indexes=[];yy=self.R.Y[k].cpu().numpy()
                for c in np.flatnonzero(valid):
                    available=np.flatnonzero(yy==c);rng=np.random.default_rng(np.random.SeedSequence([1729,k,int(c)]))
                    chosen=rng.choice(available,min(256,len(available)),replace=False);indexes.extend(chosen[:len(chosen)//2])
                if indexes:
                    idx=torch.as_tensor(np.asarray(indexes),device=self.R.dev);labels=yy[indexes]
                    prob=self.R.priv[k].eval()(self.R.X[k][idx].float())[0].float().softmax(1).double().cpu().numpy()
                    target=gradient(prob,labels,pooled,n_global,total,valid)
                    self.beta[k]=torch.from_numpy(nnls_class(prob,labels,target)).to(self.R.dev,dtype=torch.float32)
        self.up=up;self.wire=wire;self.latest_packet=wire
        self.diag=dict(scan_seconds=time.monotonic()-started,eligible=eligible.tolist(),support_clients=self.support.sum(0).tolist(),
            snapshot_round=r,nnls_beta_max=float(self.beta.max()),cohort=cohort)

    def aggregate(self,r):
        if self.reference:
            self.reference.aggregate(r);self.version=self.reference.version
            if hasattr(self.reference,'latest_packet'):self.latest_packet=self.reference.latest_packet
        else:
            self.version=r
            if self.arm not in ('cr_local','cr_fixed') and r>=2:self.scan(r)

    def state(self):
        state=dict(version=self.version,means=self.means,coefficient=self.coefficient,support=self.support,beta=self.beta,
            up=self.up,wire=self.wire,diag={k:v for k,v in self.diag.items() if k!='scan_seconds'})
        if self.reference:state['reference_state']=self.reference.state()
        if hasattr(self,'latest_packet'):state['latest_packet']=self.latest_packet
        return state


def private_epoch(run,method,k,r):
    if method.reference:return reference_epoch(run,method.reference,k,r)
    model=run.priv[k].train();opt=torch.optim.AdamW(model.parameters(),lr=run.lr(r),weight_decay=0.)
    scaler=torch.amp.GradScaler();torch.manual_seed(int(np.random.SeedSequence([run.a.seed,r,k,7]).generate_state(1)[0]))
    order=torch.randperm(len(run.Y[k]),device=run.dev,generator=run.gen(r,k,0));totals=torch.zeros(3,device=run.dev)
    applied=skipped=0
    for start in range(0,len(order),run.a.batch):
        idx=order[start:start+run.a.batch];y=run.Y[k][idx]
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(run.X[k][idx].float())[0].float();ce=F.cross_entropy(logits,y,reduction='none');aux=ce.mean()*0
            if method.arm=='cr_classweight':aux=(method.beta[k,y]*ce).mean()
            elif method.arm!='cr_local':aux=rank_loss(logits.softmax(1),y,method.means[k],method.coefficient[k])
            loss=ce.mean()+aux
        assert torch.isfinite(loss),('rank nonfinite loss',k,r)
        scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),run.a.clip)
        before=scaler.get_scale();scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
        if scaler.get_scale()<before:skipped+=1
        else:applied+=1;totals+=torch.stack([loss.detach(),ce.detach().mean(),aux.detach()])
        if getattr(run,'wb',None) is not None and start//run.a.batch%32==0:
            run.wb.log({'progress/round':r,'progress/client':k,'progress/applied':applied,'progress/skipped':skipped,
                        'progress/real_ce':float(ce.detach().mean())},step=r,commit=False)
    assert applied>0 and skipped<=16;finite(model)
    loss,ce,aux=(totals/applied).tolist()
    return dict(client=k,rows=len(run.Y[k]),applied=applied,skipped=skipped,loss=loss,real_ce=ce,auxiliary=aux)
