"""Full train80% and raw-test20% private-model validation, one arm per GPU."""
import copy
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from mh import pilot as P
from mh.v2_run import Run as Base
from mh.v2_methods import finite
from full_method import Method,ARMS
import full_driver as driver


class Run(Base):
    def __init__(self,args):
        self.a,self.dev,self.K,self.bytes=args,torch.device('cuda'),20,[]
        fixture=P.ROOT/'papers/pilot/fixture';manifest=json.loads((fixture/'manifest.json').read_text())
        self.X,self.Y=[],[]
        for k in range(self.K):
            x=np.load(fixture/f'train_{k}_X.npy',mmap_mode='r')
            y=np.load(fixture/f'train_{k}_y.npy',mmap_mode='r')
            if args.smoke:
                indices=np.random.default_rng(k).choice(len(y),1024,replace=False);x,y=x[indices],y[indices]
            self.X.append(torch.from_numpy(np.array(x,copy=True)).to(self.dev))
            self.Y.append(torch.from_numpy(np.array(y,dtype='int64',copy=True)).to(self.dev))
        hold='group' if args.smoke else 'test'
        x=np.load(fixture/f'{hold}_X.npy',mmap_mode='r');y=np.load(fixture/f'{hold}_y.npy',mmap_mode='r')
        if args.smoke:x,y=x[:256],y[:256]
        self.H={'test':(torch.from_numpy(np.array(x,copy=True)).to(self.dev),
                        torch.from_numpy(np.array(y,dtype='int64',copy=True)).to(self.dev))}
        self.n=[len(y) for y in self.Y]
        if not args.smoke:
            assert sum(self.n)==43045415 and len(self.H['test'][1])==10761343,'full data coverage gate'
            assert manifest['protocol'].startswith('final: every train80%')
            assert manifest['test_scaler'].startswith('scaler.json')
        self.counts=torch.stack([torch.bincount(y,minlength=16) for y in self.Y]).double()
        self.log_pi=((self.counts+1)/(self.counts+1).sum(1,keepdim=True)).log().float()
        self.log_pg=((self.counts.sum(0)+1)/(self.counts.sum()+16)).log().float()
        torch.manual_seed(args.seed);self.init=P.make('dagsnet').to(self.dev)
        self.priv=[copy.deepcopy(self.init) for _ in range(self.K)]
        self.work_private=copy.deepcopy(self.init);self.work_proxy=copy.deepcopy(self.init)
        self.eval_template=copy.deepcopy(self.init).eval()
        self.compiled={};self.compile_gates=[]
        if args.backend=='compiled':self.compile_preflight()

    def compile_preflight(self):
        torch._dynamo.config.recompile_limit=64
        size=self.a.batch+(self.a.aux_batch if 'gaussian' in self.a.arm else 0)
        # Numerical probe uses actual train rows, identical dropout-off on both copies; restore all BN.
        for role,model in [('private',self.work_private),('proxy',self.work_proxy),('eval',self.eval_template)]:
            if role=='proxy' and 'dml' not in self.a.arm:continue
            training=role!='eval';model.train(training)
            batch=size if training else 16384
            x=self.X[0][:batch].float()
            if len(x)<batch:x=x.repeat((batch+len(x)-1)//len(x),1)[:batch]
            saved={k:v.clone() for k,v in model.state_dict().items()};rng=torch.cuda.get_rng_state()
            dropout=[(m,m.p) for m in model.modules() if isinstance(m,torch.nn.Dropout)]
            for m,_ in dropout:m.p=0.
            reference=copy.deepcopy(model)
            compiled=torch.compile(model,mode='reduce-overhead',dynamic=False)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                torch.compiler.cudagraph_mark_step_begin();expected=reference(x)[0].float()
                actual=compiled(x)[0].float().clone()
            error=float((actual-expected).abs().max());margin=expected.topk(2,dim=1).values.diff(dim=1).abs().squeeze(1)
            decisive=margin>.02
            assert error<=.005 and torch.equal(actual.argmax(1)[decisive],expected.argmax(1)[decisive]),('compile gate',role,error)
            model.load_state_dict(saved,strict=True);torch.cuda.set_rng_state(rng)
            for m,p in dropout:m.p=p
            self.compiled[id(model)]=(compiled,batch)
            self.compile_gates.append(dict(role=role,rows=batch,max_error=error,decisive=int(decisive.sum())))
            del reference,actual,expected,x;torch.cuda.empty_cache()
        print('COMPILE PREFLIGHT PASS',self.compile_gates,flush=True)

    def forward(self,model,x):
        cached=self.compiled.get(id(model))
        return cached[0](x) if cached and len(x)==cached[1] else model(x)

    @torch.no_grad()
    def logits(self,model,x):
        self.eval_template.load_state_dict(model.state_dict(),strict=True);self.eval_template.eval()
        result=[]
        for start in range(0,len(x),16384):
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast('cuda',dtype=torch.float16):
                result.append(self.forward(self.eval_template,x[start:start+16384].float())[0].float().clone())
        return torch.cat(result)

    def private_epoch(self,method,k,r):
        started=time.time()
        original=self.priv[k];self.work_private.load_state_dict(original.state_dict(),strict=True)
        private=self.work_private;private.train()
        models=[private];proxy=None
        if method.kd:
            proxy=self.work_proxy;proxy.load_state_dict(method.proxy[k].state_dict(),strict=True);proxy.train();models.append(proxy)
        optimizer=torch.optim.AdamW([p for m in models for p in m.parameters()],lr=self.lr(r),weight_decay=0.)
        scaler=torch.amp.GradScaler();g=self.gen(r,k,0)
        torch.manual_seed(int(np.random.SeedSequence([self.a.seed,r,k,7]).generate_state(1)[0]))
        order=torch.randperm(self.n[k],device=self.dev,generator=g)
        total=torch.zeros(4,device=self.dev);applied=skipped=0
        for start in range(0,len(order),self.a.batch):
            torch.compiler.cudagraph_mark_step_begin()
            index=order[start:start+self.a.batch];x,y=self.X[k][index].float(),self.Y[k][index]
            labels=torch.arange(self.a.aux_batch,device=self.dev)%16
            if method.aux:x=torch.cat([x,method.draw(labels,g,k)])
            with torch.autocast('cuda',dtype=torch.float16):
                zs=self.forward(private,x)[0].float()
                ce=F.cross_entropy(zs[:len(index)],y);extra=ce.detach()*0
                proxy_ce=proxy_extra=ce.detach()*0
                if method.aux:extra=F.cross_entropy(zs[len(index):],labels)
                if proxy is not None:
                    zr=self.forward(proxy,x)[0].float();proxy_ce=F.cross_entropy(zr,y)
                    ls,lr=F.log_softmax(zs,1),F.log_softmax(zr,1)
                    weight=1/(ce+proxy_ce).detach().clamp_min(1e-6)
                    extra=weight*F.kl_div(ls,lr.detach(),log_target=True,reduction='batchmean')
                    proxy_extra=weight*F.kl_div(lr,ls.detach(),log_target=True,reduction='batchmean')
                loss=ce+extra+proxy_ce+proxy_extra
            assert torch.isfinite(loss),('full loss',k,r)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            for model in models:
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),self.a.clip)
                assert torch.isfinite(norm) or scaler.is_enabled()
            before=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
            if scaler.get_scale()<before:skipped+=1
            else:applied+=1;total+=torch.stack([ce.detach(),extra.detach(),proxy_ce.detach(),proxy_extra.detach()])
            if self.a.wandb_online and (start//self.a.batch+1)%1024==0:
                import wandb
                wandb.run.log({'progress/round_in_flight':r,'progress/client':k,
                    'progress/client_steps':start//self.a.batch+1,'progress/applied':applied,
                    'progress/skipped':skipped,'progress/seconds':time.time()-started,
                    'progress/loss':float(loss.detach()),'progress/grad_norm':float(norm),
                    'progress/max_logit':float(zs.detach().abs().max()),
                    'progress/softmax_entropy':float(-(F.softmax(zs.detach(),1)*F.log_softmax(zs.detach(),1)).sum(1).mean())})
        assert applied>0 and skipped<=16,(k,r,applied,skipped)
        for model in models:finite(model)
        original.load_state_dict(private.state_dict(),strict=True)
        if proxy is not None:method.proxy[k].load_state_dict(proxy.state_dict(),strict=True)
        ce,extra,proxy_ce,proxy_extra=(total/applied).tolist()
        common=dict(client=k,rows=self.n[k],applied=applied,skipped=skipped)
        method.logs[k]=dict(common,loss=proxy_ce+proxy_extra,positive_ce=proxy_ce,negative_ce=proxy_extra)
        return dict(common,loss=ce+extra,real_ce=ce,auxiliary=extra,seconds=time.time()-started)


if __name__=='__main__':
    driver.ARMS,driver.Method,driver.Run=ARMS,Method,Run
    driver.main()
