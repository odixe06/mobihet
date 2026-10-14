"""V2 unlimited-channel feasibility. Contact runs are opened only after paired U gates."""
import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from mh import pilot as P
from mh.new_fl import digest, save
from mh.v2_methods import ARMS, Method, finite
from mh.metrics import METRIC_KEYS, metrics_from_confusion, per_class_from_confusion


class Run(P.Run):
    def __init__(self,a):
        super().__init__(a); self.bytes=[]
        if a.smoke:
            self.X=[x[:256] for x in self.X]; self.Y=[y[:256] for y in self.Y]
            self.H={h:(x[:256],y[:256]) for h,(x,y) in self.H.items()}
            self.n=[len(y) for y in self.Y]
            self.counts=torch.stack([torch.bincount(y,minlength=P.C) for y in self.Y]).double()
            self.log_pi=((self.counts+1)/(self.counts+1).sum(1,keepdim=True)).log().float()

    def private_epoch(self,m,k,r):
        model=self.priv[k]; model.train()
        opt=torch.optim.AdamW(model.parameters(),lr=self.lr(r),weight_decay=0.)
        scaler=torch.amp.GradScaler(); g=self.gen(r,k,0)
        torch.manual_seed(int(np.random.SeedSequence([self.a.seed,r,k,7]).generate_state(1)[0]))
        X,Y=self.X[k],self.Y[k]; order=torch.randperm(len(Y),device=self.dev,generator=g)
        total=torch.zeros(2,device=self.dev); applied=skipped=0
        for start in range(0,len(Y),self.a.batch):
            idx=order[start:start+self.a.batch]
            with torch.autocast('cuda',dtype=torch.float16):
                loss,parts=m.private_loss(k,X[idx].float(),Y[idx],g,r)
            assert torch.isfinite(loss),(k,r,'private loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(),self.a.clip)
            before=scaler.get_scale(); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            if scaler.get_scale()<before: skipped+=1
            else: applied+=1; total+=parts
        assert applied>0 and skipped<=16,(k,r,applied,skipped)
        finite(model); real,aux=(total/applied).tolist()
        return dict(client=k,rows=len(Y),applied=applied,skipped=skipped,loss=real+self.a.aux_weight*aux,
                    real_ce=real,auxiliary=aux)

    @torch.no_grad()
    def evaluate(self,m,out,r):
        arrays={}; evaluated={}
        roles={'private':self.priv} if m.role=='private' else {}
        if m.proxy and not self.a.arm.startswith('c_'):
            for k in range(self.K): m.prepare(k,r)
            roles['proxy']=m.proxy
        for role,models in roles.items():
            results={}
            for hold,(X,y) in self.H.items():
                arrays[hold+'_y']=y.cpu().numpy().astype('uint8'); result={}
                rules=('head','gp') if role=='private' else ('head',)
                cms={rule:[] for rule in rules}; preds={rule:[] for rule in rules}
                for k,model in enumerate(models):
                    z=self.logits(model,X); assert torch.isfinite(z).all()
                    for rule in rules:
                        bias=self.log_pg-self.log_pi[k] if rule=='gp' else 0.
                        pred=(z+bias).argmax(1)
                        cms[rule].append(torch.bincount(y*P.C+pred,minlength=P.C**2).view(P.C,P.C).cpu().numpy())
                        preds[rule].append(pred.cpu().numpy().astype('uint8'))
                for rule,cc in cms.items():
                    mm=[metrics_from_confusion(c) for c in cc]
                    pc=np.mean([[x['f1'] for x in per_class_from_confusion(c,P.CLASSES)] for c in cc],axis=0)
                    result[rule]=dict(mean_metrics={key:float(np.mean([v[key] for v in mm])) for key in METRIC_KEYS},
                        f1_macro=float(np.mean([v['f1_macro'] for v in mm])),benign_f1=float(pc[0]),
                        f1_per_class=pc.tolist(),confusion_clients=np.stack(cc).tolist())
                    arrays[f'{role}_{hold}_{rule}']=np.stack(preds[rule])
                results[hold]=result
            evaluated[role]=results
        pf=out/'predictions'/f'round_{r:03d}.npz'; pf.parent.mkdir(exist_ok=True)
        tmp=pf.with_suffix('.tmp')
        with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
        os.replace(tmp,pf)
        return evaluated


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',choices=ARMS,required=True); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--rounds',type=int,default=20); p.add_argument('--batch',type=int,default=512)
    p.add_argument('--private',default='dagsnet'); p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--lr_min',type=float,default=1e-5); p.add_argument('--clip',type=float,default=1.)
    p.add_argument('--eval_every',type=int,default=5); p.add_argument('--aux_batch',type=int,default=256)
    p.add_argument('--aux_weight',type=float,default=1.); p.add_argument('--temperature',type=float,default=2.)
    p.add_argument('--tau',type=float,default=.05); p.add_argument('--weight_cap',type=float,default=10.)
    p.add_argument('--out',default=str(P.ROOT/'papers/v2/runs')); p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true'); p.add_argument('--stop_after',type=int)
    p.add_argument('--wandb_project',default='mobihet-v2'); p.add_argument('--wandb_online',action='store_true')
    p.add_argument('--max_hours',type=float,default=4.)
    a=p.parse_args(); assert a.private=='dagsnet'
    if a.smoke: a.rounds=3; a.eval_every=1; a.aux_batch=64
    assert a.smoke or a.wandb_online, 'production requires W&B online before data/model preparation'
    out=Path(a.out)/f'{a.arm}_s{a.seed}'; out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(a).items() if k not in ('out','resume','stop_after','wandb_online','wandb_project','max_hours')}
    source={f.name:digest(f) for f in sorted((P.ROOT/'mh').glob('*.py'))}
    fx=P.ROOT/'papers/pilot/fixture'; manifest=json.loads((fx/'manifest.json').read_text())
    for name,sha in manifest['files'].items(): assert digest(fx/name)==sha,('fixture mismatch',name)
    identity=dict(config=config,source=source,fixture=digest(fx/'manifest.json'))
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if (out/'config.json').exists():
        assert a.resume and json.loads((out/'config.json').read_text())['fingerprint']==fingerprint
    else:
        assert not a.resume,'resume requested without run'; save(out/'config.json',dict(identity,fingerprint=fingerprint))
    wb=None
    if a.wandb_online:
        import wandb
        key=os.environ.get('WANDB_API_KEY'); assert key,'W&B key required'
        assert wandb.login(key=key,anonymous='never',relogin=True,verify=True)
        wb=wandb.init(project=a.wandb_project,id=out.name+'-'+fingerprint[:8],resume='allow',
                      name=out.name,config=dict(config,fingerprint=fingerprint),mode='online',dir=str(out),
                      settings=wandb.Settings(save_code=False,disable_code=True))
        assert not wb.settings._offline
        wb.log({'preflight/online':1}); print('WANDB',wb.url,flush=True)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(1); torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    t0=time.time(); R=Run(a); m=Method(R,initialize=not a.resume)
    if not a.resume: m.prior()
    hist=[]; previous=None
    if a.resume:
        markers=sorted((out/'complete').glob('round_*.json'))
        assert markers,'resume missing committed rounds'
        for marker in markers:
            row=json.loads(marker.read_text()); r=row['round']; assert r==len(hist)+1 and row['prev_sha']==previous
            assert row['fingerprint']==fingerprint
            assert digest(out/'weights'/f'round_{r:03d}.pt')==row['weights_sha']
            assert digest(out/'resume'/f'round_{r:03d}.pt')==row['resume_sha']
            hist.append(row); previous=row['weights_sha']
        r=len(hist)
        weights=torch.load(out/'weights'/f'round_{r:03d}.pt',map_location=R.dev,weights_only=True)
        for model,sd in zip(R.priv,weights['private']): model.load_state_dict(sd,strict=True)
        for model,sd in zip(m.proxy,weights['proxy']): model.load_state_dict(sd,strict=True)
        resume=torch.load(out/'resume'/f'round_{r:03d}.pt',map_location=R.dev,weights_only=True)
        for key,value in resume['algorithm'].items(): setattr(m,key,value)
        R.bytes=resume['bytes']
        R.log_pg=resume['log_pg']; R.ratio_pi=resume['ratio_pi']
    print('setup',a.arm,'role',m.role,'rows',sum(R.n),'sec',round(time.time()-t0,1),flush=True)
    for r in range(len(hist)+1,a.rounds+1):
        if time.time()-t0+1.15*max([v['sec'] for v in hist] or [0])+60>a.max_hours*3600:
            print('budget stop before round',r,flush=True); break
        t1=time.time(); private_logs=[]; proxy_logs=[]; m.start_round()
        for k in range(R.K):
            if m.proxy and m.role=='private': m.teacher=m.prepare(k,r)
            if m.role=='private': private_logs.append(R.private_epoch(m,k,r))
            if m.proxy:
                packet,log=m.proxy_epoch(k,r); proxy_logs.append(log)
                if packet is not None: m.aggregate_sum.add(k,packet)
        m.aggregate(r)
        row=dict(round=r,lr=R.lr(r),clients=private_logs,proxy_clients=proxy_logs,version=m.version,
                 fingerprint=fingerprint,prev_sha=previous,diagnostics=dict(m.diag),
                 loss=float(np.mean([v['loss'] for v in private_logs or proxy_logs])))
        if hasattr(m,'latest_packet'):
            packet_path=out/'knowledge'/f'round_{r:03d}.bin'; packet_path.parent.mkdir(exist_ok=True)
            tmp=packet_path.with_suffix('.tmp'); tmp.write_bytes(m.latest_packet); os.replace(tmp,packet_path)
            row['payload_sha']=digest(packet_path)
        if r%a.eval_every==0 or r==a.rounds:
            row['eval']=R.evaluate(m,out,r)
            row['predictions_sha']=digest(out/'predictions'/f'round_{r:03d}.npz')
            if a.arm.startswith('c_'): row['ratio']=m.ratio_diagnostics()
        weights=dict(private=[{k:v.detach().cpu() for k,v in model.state_dict().items()} for model in R.priv],
                     proxy=[{k:v.detach().cpu() for k,v in model.state_dict().items()} for model in m.proxy],
                     architecture=dict(P.CFG,n_features=P.D,num_classes=P.C),round=r,fingerprint=fingerprint)
        wp=out/'weights'/f'round_{r:03d}.pt'; save(wp,weights); row['weights_sha']=digest(wp)
        rp=out/'resume'/f'round_{r:03d}.pt'
        save(rp,dict(algorithm=m.state(),bytes=R.bytes,log_pg=R.log_pg,ratio_pi=R.ratio_pi,
                     round=r,fingerprint=fingerprint))
        row['resume_sha']=digest(rp); row['sec']=time.time()-t1; previous=row['weights_sha']
        save(out/'complete'/f'round_{r:03d}.json',row); hist.append(row)  # atomic commit LAST
        role=m.role; rule='head' if role=='proxy' else 'gp'
        score=row.get('eval',{}).get(role,{}).get('group',{}).get(rule,{}).get('f1_macro')
        print(f'r{r} {a.arm} s{a.seed} {role} group {rule} {score} loss {row["loss"]:.4f} {row["sec"]:.0f}s',flush=True)
        if wb:
            metrics=dict(round=r,loss=row['loss'],lr=row['lr'],seconds=row['sec'],version=m.version,
                         applied=sum(v['applied'] for v in private_logs+proxy_logs),
                         skipped=sum(v['skipped'] for v in private_logs+proxy_logs))
            for model_role,holds in row.get('eval',{}).items():
                for hold,rules in holds.items():
                    for rule,ev in rules.items():
                        metrics.update({f'{model_role}/{hold}/{rule}/{key}':v for key,v in ev['mean_metrics'].items()})
            for group,logs in (('private',private_logs),('proxy',proxy_logs)):
                if logs:
                    for key in ('real_ce','auxiliary','positive_ce','negative_ce'):
                        if key in logs[0]: metrics[f'{group}/{key}']=float(np.mean([v[key] for v in logs]))
            wb.log(metrics,step=r)
        if a.stop_after and r>=a.stop_after: break
    result=dict(config=vars(a),fingerprint=fingerprint,role=m.role,history=hist,bytes=R.bytes,
                train_rows=R.n,complete=len(hist)==a.rounds,minutes=(time.time()-t0)/60)
    save(out/'result.json',result)
    with (out/'metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['round','role','holdout','rule',*METRIC_KEYS]); w.writeheader()
        for row in hist:
            for role,holds in row.get('eval',{}).items():
                for hold,rules in holds.items():
                    for rule,ev in rules.items():
                        w.writerow(dict(round=row['round'],role=role,holdout=hold,rule=rule,**ev['mean_metrics']))
    if wb: wb.summary['complete']=result['complete']; wb.summary['rounds_completed']=len(hist); wb.finish()
    print('done',a.arm,'complete',result['complete'],flush=True)


if __name__=='__main__': main()
