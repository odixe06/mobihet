"""Noise-init dynamic condensation at producer; only synthetic books cross clients."""
import copy
import hashlib
import time
import torch
import torch.nn.functional as F
from mh import pilot as P
from mh.v2_math import Layout,pack_vector,unpack
from mh.v2_methods import Method as Base,calibrate


class Method(Base):
    def __init__(self,run,initialize=True):
        self.R,self.arm,self.diag=run,run.a.arm,{}
        self.role='private'; self.proxy=[]; self.layout=Layout(run.init)
        self.global_vector=self.layout.flatten(run.init).clone(); self.version=0
        self.coeff=self.residual=None; self.ipc=1 if self.arm.startswith('g1_') else 4
        self.mode=self.arm.split('_',1)[1]; self.round_no=0; self.phase=0
        self.books=[]; self.book_packets=[]; self.peers=[]
        if initialize: self.make_gaussian()

    def condense(self,k,r,steps=None):
        R=self.R; steps=steps or (16 if R.a.smoke else 80)
        g=R.gen(r,k,350)
        noise=torch.randn(P.C,self.ipc,P.D,device=R.dev,generator=g)
        if self.mode=='noise': return self.mu[:,None]+self.sd[:,None]*noise,dict(slots=0,applied=0,fit=[])
        current=self.mode in ('grad_trained','mean_trained','own')
        gradient=self.mode.startswith('grad') or self.mode=='own'
        encoder=copy.deepcopy(R.priv[k] if current else R.init)
        idx=torch.randperm(len(R.Y[k]),device=R.dev,generator=R.gen(r,k,351))[:4096]
        calibrate(encoder,R.X[k][idx]); encoder.eval()
        encoder.requires_grad_(gradient); params=list(encoder.parameters())
        target=[]; batches=[]
        with torch.autocast('cuda',enabled=False):
            for c in range(P.C):
                indices=torch.where(R.Y[k]==c)[0]
                indices=indices[torch.randperm(len(indices),device=R.dev,generator=g)[:32]]
                x=R.X[k][indices].float(); batches.append(len(x))
                if not len(x): target.append(None); continue
                if gradient:
                    label=torch.full((len(x),),c,device=R.dev,dtype=torch.long)
                    grads=torch.autograd.grad(F.cross_entropy(encoder(x)[0],label),params)
                    target.append([v.detach() for v in grads])
                else:
                    with torch.no_grad(): target.append(encoder(x)[1].mean(0).detach())
            variables=[noise[c].detach().clone().requires_grad_(True) for c in range(P.C)]
            opts=[torch.optim.Adam([v],lr=.03) for v in variables]
            fit=[]; applied=0
            for step in range(steps):
                c=step%P.C
                if target[c] is None: continue  # no invented real gradient for an absent local class
                u=variables[c]; synthetic=self.mu[c]+self.sd[c]*u
                if gradient:
                    label=torch.full((self.ipc,),c,device=R.dev,dtype=torch.long)
                    grads=torch.autograd.grad(F.cross_entropy(encoder(synthetic)[0],label),params,create_graph=True)
                    terms=[1-F.cosine_similarity(a.flatten(),b.flatten(),dim=0,eps=1e-8)
                           for a,b in zip(grads,target[c]) if b.ndim>=2]
                    loss=torch.stack(terms).mean()
                else:
                    feature=encoder(synthetic)[1].mean(0); truth=target[c]
                    loss=(feature-truth).square().mean()/truth.square().mean().clamp_min(1e-8)
                loss=loss+.01*u.mean(0).square().mean()
                grad,=torch.autograd.grad(loss,u)
                assert torch.isfinite(loss) and torch.isfinite(grad).all(),'nonfinite condensation objective'
                opts[c].zero_grad(set_to_none=True); u.grad=grad; opts[c].step(); applied+=1
                fit.append(float(loss.detach()))
            book=torch.stack([self.mu[c]+self.sd[c]*u.detach() for c,u in enumerate(variables)])
        assert torch.isfinite(book).all()
        hash_=hashlib.sha256()
        param_hash=hashlib.sha256()
        for name,value in sorted(encoder.named_parameters()):
            param_hash.update(name.encode()); param_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
        for name,value in sorted(encoder.state_dict().items()):
            hash_.update(name.encode()); hash_.update(value.detach().cpu().contiguous().numpy().tobytes())
        return book,dict(slots=steps,applied=applied,fit=fit,real_batch_rows=batches,
            encoder_sha256=hash_.hexdigest(),encoder_parameter_sha256=param_hash.hexdigest(),
            encoder_current=current,objective='gradient_matrix_tensor_cosine' if gradient else 'normalized_mean_latent')

    def make_books(self,r):
        R=self.R; self.phase+=1; books=[]; packets=[]; diagnostics=[]
        start=time.time()
        for k in range(R.K):
            book,diag=self.condense(k,r)
            packet=pack_vector('dense',book.flatten(),base=self.phase-1,version=self.phase,half=True)
            decoded=unpack(packet,R.dev)['values'].reshape(P.C,self.ipc,P.D)
            # Exact fixture-precision duplicate guard, checked locally before individual synthetic upload.
            points=decoded.reshape(-1,P.D).half(); X=R.X[k]
            for offset in range(0,len(X),2048):
                same=(X[offset:offset+2048].half()[:,None,:]==points[None,:,:]).all(2)
                assert not same.any(),'synthetic book exactly equals a local raw record'
            if self.mode!='own': self.record(packet,r,k,'up','synthetic_book')
            books.append(decoded); packets.append(packet); diagnostics.append(diag)
        self.books=torch.stack(books); self.book_packets=packets
        self.peers=[]
        for k in range(R.K):
            peers=[k] if self.mode=='own' else [(k+1+((self.phase-1)*4+j)%(R.K-1))%R.K for j in range(4)]
            assert self.mode=='own' or k not in peers
            self.peers.append(peers)
            if self.mode!='own':
                for producer in peers: self.record(packets[producer],r,k,'down','synthetic_book')
        self.version=self.phase
        self.diag=dict(phase=self.phase,refresh_round=r,condensation_seconds=time.time()-start,
            producers=diagnostics,points_per_class=self.ipc,peers_per_client=1 if self.mode=='own' else 4,
            exact_raw_duplicate_guard=True)

    def start_round(self):
        period=2 if self.R.a.smoke else 5
        if self.mode!='diag' and self.round_no%period==0: self.make_books(self.round_no+1)

    def private_loss(self,k,x,y,g,r):
        R=self.R; real=F.cross_entropy(R.priv[k](x)[0].float(),y)
        labels=torch.arange(R.a.aux_batch,device=R.dev)%P.C
        if self.mode=='diag': synthetic=self.draw(labels,g)
        else:
            peers=torch.tensor(self.peers[k],device=R.dev)
            producer=peers[torch.randint(len(peers),(len(labels),),device=R.dev,generator=g)]
            point=torch.randint(self.ipc,(len(labels),),device=R.dev,generator=g)
            synthetic=self.books[producer,labels,point]
        extra=F.cross_entropy(R.priv[k](synthetic)[0].float(),labels)
        return real+R.a.aux_weight*extra,torch.stack([real.detach(),extra.detach()])

    def aggregate(self,r): self.round_no=r

    def state(self):
        return dict(global_vector=self.global_vector,version=self.version,mu=self.mu,sd=self.sd,q_version=self.q_version,
            round_no=self.round_no,phase=self.phase,books=self.books,book_packets=self.book_packets,peers=self.peers,
            diag={key:value for key,value in self.diag.items() if key!='condensation_seconds'})
