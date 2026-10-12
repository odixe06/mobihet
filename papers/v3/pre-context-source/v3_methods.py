"""V3 private CE+KD with four real fp32 proxy carriers, local EF, explicit PowerSGD stages."""
import copy
import torch
from mh.v2_methods import Method as V2Method,calibrate
from mh.v2_math import Layout,SecureSum,pack_vector,unpack
from mh.v3_math import coordinates,basis,orthogonal,matrix,unpad


class Method(V2Method):
    def __init__(self,run,initialize=True):
        self.R,self.arm,self.diag=run,run.a.arm,{}
        self.role='private'; self.proxy=[copy.deepcopy(run.init) for _ in range(run.K)]
        self.layout=Layout(run.init); self.global_vector=self.layout.flatten(run.init).clone()
        self.coeff=self.basis=self.sketch=self.residual=None; self.own_coeff=[]; self.version=0
        self.family=self.arm.split('_')[1]; self.own='own' in self.arm; self.ef=self.arm.endswith('_ef')
        self.k=512 if 'small' in self.arm else 24576
        self.rank=8 if self.family=='p' or 'rank8' in self.arm else (1 if 'small' in self.arm else 32)
        self.client_residual=[torch.zeros_like(self.global_vector) for _ in range(run.K)]
        self.own_vectors=[self.global_vector.clone() for _ in range(run.K)] if self.own else []

    def prepare(self,k,r):
        model=self.proxy[k]
        self.layout.load(model,self.own_vectors[k] if self.own else self.global_vector)
        idx=torch.randperm(len(self.R.Y[k]),device=self.R.dev,generator=self.R.gen(881,k))[:4096]
        calibrate(model,self.R.X[k][idx]); return model

    def start_round(self):
        r=self.version+1
        self.mask=coordinates(self.layout,self.k,r,self.family=='t').to(self.R.dev) if self.family in ('r','t') else None
        self.q=basis(self.rank,r,self.R.dev) if self.family in ('o','p') else None
        self.local_delta=[]; self.client_coeff=[]; self.staged=[]; self.own_left=[]
        size=self.k if self.mask is not None else 512*self.rank
        self.aggregate_sum=None if self.own else SecureSum('coeff',self.version,size,self.R.dev)

    def project(self,target,left=None):
        if self.mask is not None: return target[self.mask]
        return (matrix(target)@self.q if left is None else matrix(target).T@left).reshape(-1)

    def reconstruct(self,coeff,left=None):
        if self.mask is not None:
            value=torch.zeros_like(self.global_vector); value[self.mask]=coeff; return value
        if left is None: return unpad(coeff.reshape(512,self.rank)@self.q.T)
        return unpad(left@coeff.reshape(772,self.rank).T)

    def packet(self,coeff,k,r,stage):
        count=self.R.n[k]
        packet=pack_vector('coeff',coeff*count,base=self.version,version=r,count=count)
        decoded=unpack(packet,self.R.dev)['values']/count
        if not self.own: self.record(packet,r,k,'up',stage)
        return packet,decoded

    def proxy_epoch(self,k,r):
        base=self.own_vectors[k] if self.own else self.global_vector
        hooks=[]
        if self.family=='t' and self.arm.endswith('_train'):
            allowed=torch.zeros_like(base); allowed[self.mask]=1
            masks=self.layout.mapping(allowed)
            for name,p in self.proxy[k].named_parameters():
                mask=masks[name]
                hooks.append(p.register_hook(lambda gradient,mask=mask:gradient*mask))
        original=self.arm
        try:
            # Reuse exact A CE optimizer/BN/dropout loop; suppress its dense transport.
            self.arm='a_own'
            _,log=super().proxy_epoch(k,r)
        finally:
            self.arm=original
            for hook in hooks: hook.remove()
        delta=self.layout.flatten(self.proxy[k])-base; self.local_delta.append(delta.detach().clone())
        if self.family=='t' and self.arm.endswith('_train'):
            outside=torch.ones(len(delta),device=delta.device,dtype=torch.bool); outside[self.mask]=False
            assert torch.count_nonzero(delta[outside])==0,'optimizer changed unpaid coordinates'
        target=delta+(self.client_residual[k] if self.ef else 0)
        packet,decoded=self.packet(self.project(target),k,r,'P' if self.family=='p' else 'proxy_update')
        if self.family=='p':
            self.staged.append(target.detach().clone())
            if self.own:
                left=orthogonal(decoded.reshape(512,self.rank))
                self.own_left.append(left)
                _,second=self.packet(self.project(target,left),k,r,'Q')
                restored=self.reconstruct(second,left); self.client_coeff.append(second)
                self.own_vectors[k]=base+restored
                self.client_residual[k]=target-restored if self.ef else torch.zeros_like(target)
        else:
            restored=self.reconstruct(decoded); self.client_coeff.append(decoded)
            self.client_residual[k]=target-restored if self.ef else torch.zeros_like(target)
            if self.own: self.own_vectors[k]=base+restored
        return (None if self.own else packet),log

    def aggregate(self,r):
        if self.own:
            self.version=r; self.diag=self.diagnostics(); return
        total,n=self.aggregate_sum.finish(); mean=total/n
        if self.family=='p':
            left_packet=pack_vector('coeff',orthogonal(mean.reshape(512,self.rank)).reshape(-1),base=self.version,version=r)
            self.left_factor=unpack(left_packet,self.R.dev)['values'].reshape(512,self.rank)
            second=SecureSum('coeff',self.version,772*self.rank,self.R.dev)
            for k,target in enumerate(self.staged):
                self.record(left_packet,r,k,'down','P_basis')
                packet,decoded=self.packet(self.project(target,self.left_factor),k,r,'Q')
                second.add(k,packet); self.client_coeff.append(decoded)
                restored=self.reconstruct(decoded,self.left_factor)
                self.client_residual[k]=target-restored if self.ef else torch.zeros_like(target)
            total,n=second.finish(); mean=total/n
            # Preserve the actual first broadcast bytes, hashed through resume state.
            self.left_packet=list(left_packet)
        packet=pack_vector('coeff',mean,base=self.version,version=r)
        self.coeff=unpack(packet,self.R.dev)['values']
        restored=self.reconstruct(self.coeff,self.left_factor if self.family=='p' else None)
        self.global_vector+=restored; assert torch.isfinite(self.global_vector).all()
        self.latest_packet=packet
        for k in range(self.R.K): self.record(packet,r,k,'down','Q' if self.family=='p' else 'proxy_update')
        self.version=r; self.diag=self.diagnostics()

    def diagnostics(self):
        return dict(residual_norms=[float(v.norm()) for v in self.client_residual],
                    delta_norms=[float(v.norm()) for v in self.local_delta],
                    parameter_max=float(self.global_vector.abs().max()))

    def state(self):
        result=super().state()
        result.update(client_residual=self.client_residual,own_vectors=self.own_vectors,
            local_delta=self.local_delta,client_coeff=self.client_coeff,mask=self.mask,q=self.q,own_left=self.own_left)
        if hasattr(self,'left_factor'): result.update(left_factor=self.left_factor,left_packet=self.left_packet)
        return result
