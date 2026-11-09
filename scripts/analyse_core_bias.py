"""Core r50 prediction bias from audited CMs: precision/recall/pred-true per class, F1 without timeDelay, cohort timeDelay. No training."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path('/home/odixe/project/mobihet'); sys.path.insert(0,str(ROOT))
from core.metrics import ten
from scripts.research_metrics import client_f1
B=Path('/home/odixe/project/fl_dagsnet_veremi_v2/runs/pulls')
counts=np.load(ROOT/'evidence/research-data/train_counts.npy'); n=counts.sum(1)
d=ROOT/'artifacts/f50/mh-f3-core-s43-r50/mh-f3-core-s43-r50'
core=np.load(d/'cms_r050.npz')['private'].astype(np.int64)
apf=np.load(B/'fl2-apfed-r2/fl2-apfed-r2/cms_r050.npz')['client'].astype(np.int64)
fav=np.load(B/'fl2-fedavg/fl2-fedavg/cms_r050.npz')['global'].astype(np.int64)
cen=np.load(B/'fl2-central-w1/fl2-central-w1/cm_e050.npy').astype(np.int64)
def pr(c):
    tp=np.diag(c); return tp/np.maximum(c.sum(0),1), tp/np.maximum(c.sum(1),1), c.sum(0)/np.maximum(c.sum(1),1)
P={k:pr(v) for k,v in {'core':core.sum(0),'fedavg':fav.sum(0),'apfed':apf.sum(0),'central':cen}.items()}
print('class  prec[core fav apf cen] rec[...] pred/true[...]')
for c in range(16):
    print(c,*[f'{P[k][0][c]:.3f}' for k in P],'|',*[f'{P[k][1][c]:.3f}' for k in P],'|',*[f'{P[k][2][c]:.2f}' for k in P])
# macro F1 by client excluding timeDelay
def cf(cm,excl=()):
    s=cm.sum(2); p=cm.sum(1); t=np.diagonal(cm,axis1=1,axis2=2)
    f=2*t/np.maximum(s+p,1); pres=s>0
    for e in excl: pres=pres.copy(); pres[:,e]=False
    return (f*pres).sum(1)/np.maximum(pres.sum(1),1)
for name,cm in (('core',core),('fedavg',fav),('apfed',apf)):
    a=cf(cm);b=cf(cm,(13,));print(name,'mean',a.mean(),'p10',np.percentile(a,10),'excl13 mean',b.mean(),'p10',np.percentile(b,10))
# timeDelay by cohort
bins=[(0,256),(256,1024),(1024,4096),(4096,16384),(16384,10**9)]
for lo,hi in bins:
    m=(n>=lo)&(n<hi)
    for name,cm in (('core',core),('fedavg',fav)):
        c=cm[m].sum(0); print(lo,name,'tD P %.3f R %.3f F1 %.3f | benign P %.3f R %.3f'%(c[13,13]/max(c[:,13].sum(),1),c[13,13]/c[13].sum(),2*c[13,13]/(c[13].sum()+c[:,13].sum()),c[0,0]/c[:,0].sum(),c[0,0]/c[0].sum()))
# train vs test r20 F3, class recall
tr=np.load(d/'train_cms_r020.npz'); print(tr.files); T=tr[tr.files[0]].astype(np.int64); te=np.load(d/'cms_r020.npz')['private'].astype(np.int64)
if T.ndim==3:
    for lo,hi in [(0,4096),(4096,10**9)]:
        m=(n>=lo)&(n<hi); a=T[m].sum(0); b=te[m].sum(0)
        print('r20',lo,'train rec',[round(float(a[c,c]/a[c].sum()),3) for c in (0,1,7,11,13)],'test rec',[round(float(b[c,c]/b[c].sum()),3) for c in (0,1,7,11,13)])
    print('train acc',np.trace(T.sum(0))/T.sum(),'test acc',np.trace(te.sum(0))/te.sum())
# external (other vehicles test rows)
for r in (20,):
    for rk in (0,1):
        e=np.load(d/f'external_cms_rank{rk}_r{r:03d}.npz'); print({k:e[k].shape for k in e.files})
