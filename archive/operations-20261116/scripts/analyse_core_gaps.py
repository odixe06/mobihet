"""Core r50 gaps from audited CMs: class F1 vs baselines, error budget, cohorts, seed noise, weak clients. No training; reads ../fl_dagsnet_veremi_v2."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/odixe/project/mobihet'); sys.path.insert(0,str(ROOT))
from core.metrics import ten
from scripts.research_metrics import client_f1
lab=json.load(open(ROOT/'knowledge/label_mapping.json')); 
names=[None]*16
try:
    for k,v in (lab.get('label_to_id') or lab).items():
        if isinstance(v,int): names[v]=k
except Exception as e: print(e)
B=Path('/home/odixe/project/fl_dagsnet_veremi_v2/runs/pulls')
counts=np.load(ROOT/'evidence/research-data/train_counts.npy'); n=counts.sum(1)
F={'F2':ROOT/'artifacts/f50/mh-f2-core-s42-r50/mh-f2-core-s42-r50/cms_r050.npz','F3':ROOT/'artifacts/f50/mh-f3-core-s43-r50/mh-f3-core-s43-r50/cms_r050.npz'}
cm={k:np.load(v)['private'].astype(np.int64) for k,v in F.items()}
def load(p,key):
    d=np.load(p); return d[key].astype(np.int64),d.files
for name,p in [('apfed',B/'fl2-apfed-r2/fl2-apfed-r2/cms_r050.npz'),('fedavg',B/'fl2-fedavg/fl2-fedavg/cms_r050.npz'),('scaffold',B/'fl2-scaffold/fl2-scaffold/cms_r050.npz'),('canperfl',B/'fl2-canperfl/fl2-canperfl/cms_r010.npz')]:
    d=np.load(p); print(name,{k:d[k].shape for k in d.files})
    key=[k for k in ('private','client','global') if k in d.files]
    cm[name]=d[key[0]].astype(np.int64)
    if name=='canperfl': cm['canp_g']=d['global'].astype(np.int64)
print(sorted(p.name for p in (B/'fl2-central-w1/fl2-central-w1').iterdir())[:20])
cc=sorted((B/'fl2-central-w1/fl2-central-w1').glob('cm_e*.npy'))
print(cc[-1]); x=np.load(cc[-1]).astype(np.int64); print(x.shape); cm['central']=x
print('== per class F1')
pc={k:ten(v.sum(0) if v.ndim==3 else v)['f1_per_class'] for k,v in cm.items()}
ks=list(pc); print(' '*30,*[f'{k:>8s}' for k in ks])
tot=cm['F3'].sum(0)
for c in range(16):
    print(f'{c:2d} {str(names[c])[:26]:26s}',*[f'{pc[k][c]:8.4f}' for k in ks], f'sup {tot[c].sum()}')
print('== F3 pooled error budget, top confusions')
E_=tot.copy(); np.fill_diagonal(E_,0); T=E_.sum(); print('errors',T,'of',tot.sum())
idx=np.argsort(E_.ravel())[::-1][:12]
for i in idx:
    a,b=divmod(i,16); print(f'{names[a]}->{names[b]}: {E_[a,b]} {E_[a,b]/T:.3f}')
print('row err share', [(names[c],round(E_[c].sum()/T,3)) for c in np.argsort(E_.sum(1))[::-1][:6]])
print('col err share', [(names[c],round(E_[:,c].sum()/T,3)) for c in np.argsort(E_.sum(0))[::-1][:6]])
print('== cohorts client F1 (by train rows)')
bins=[(0,256),(256,1024),(1024,4096),(4096,16384),(16384,10**9)]
keys=[k for k in cm if cm[k].ndim==3]
print('cohort n',*keys)
fs={k:client_f1(cm[k]) for k in keys}
for lo,hi in bins:
    m=(n>=lo)&(n<hi); print(lo,hi,m.sum(),*[f'{fs[k][m].mean():.4f}' for k in keys], 'test rows share %.3f'%(cm['F3'][m].sum()/cm['F3'].sum()))
print('== seed noise'); d=fs['F2']-fs['F3']; print('std diff',d.std(),'abs>.05',(abs(d)>.05).mean())
for lo,hi in bins:
    m=(n>=lo)&(n<hi); print(lo,hi,'std',round(d[m].std(),4))
print('corr F2,F3',np.corrcoef(fs['F2'],fs['F3'])[0,1])
avg=(fs['F2']+fs['F3'])/2; print('p10 of mean-of-2-seed F1',np.percentile(avg,10),'mean',avg.mean())
print('== bottom decile F3')
th=np.percentile(fs['F3'],10); b=fs['F3']<=th; sup=cm['F3'].sum(2)
print('n',b.sum(),'median train',np.median(n[b]),'all',np.median(n),'median test',np.median(sup[b].sum(1)),np.median(sup.sum(1)),'n test classes',np.median((sup[b]>0).sum(1)),np.median((sup>0).sum(1)))
print('share with n<256',(n[b]<256).mean(),'share n>=4096',(n[b]>=4096).mean())
Bc=cm['F3'][b].sum(0); E2=Bc.copy(); np.fill_diagonal(E2,0); T2=E2.sum();print('bottom err rows',T2,'acc',np.trace(Bc)/Bc.sum())
for i in np.argsort(E2.ravel())[::-1][:10]:
    a_,b_=divmod(i,16); print(f'  {names[a_]}->{names[b_]} {E2[a_,b_]/T2:.3f}')
pcl=[]
for k in np.where(b)[0]:
    c=cm['F3'][k]; s=c.sum(1); pr=c.sum(0); tp=np.diag(c); pres=s>0
    f=np.where(pres,2*tp/np.maximum(s+pr,1),np.nan); pcl.append(f)
pcl=np.array(pcl); print('bottom: mean class F1 when present & presence',[(names[c][:10],round(float(np.nanmean(pcl[:,c])),3),round(float(np.mean(~np.isnan(pcl[:,c]))),2)) for c in range(16)])
# errors predicted into class with no train support at client
c3=cm['F3']; absent_pred=(c3*(counts[:,None,:]==0)).sum(); off=c3.sum()-np.trace(c3,axis1=1,axis2=2).sum()
print('errors predicted into locally-absent class share',absent_pred/off)
absent_true=(c3*(counts[:,:,None]==0)).sum(); print('test rows whose true class absent locally',absent_true/c3.sum())
# prior shift: test class share vs train share per client
ts=sup/np.maximum(sup.sum(1,keepdims=True),1); tr=counts/np.maximum(n[:,None],1)
tv=.5*np.abs(ts-tr).sum(1); print('TV(train,test prior) overall mean',tv.mean(),'median',np.median(tv),'bottom mean',tv[b].mean(),'corr with F1',np.corrcoef(tv,fs['F3'])[0,1])
