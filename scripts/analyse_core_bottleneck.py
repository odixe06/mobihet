"""Core r50 bottleneck vs oracle/FedAvg/APFed/centralized from audited own-test CMs (seed42); read-only, no training."""
import json,sys,numpy as np
from pathlib import Path
R=Path('/home/odixe/project/mobihet'); B=Path('/home/odixe/project/fl_dagsnet_veremi_v2/runs/pulls')
sys.path.insert(0,str(R))
from scripts.research_metrics import client_f1
n=np.load(R/'evidence/research-data/train_counts.npy'); cnt=n; n=n.sum(1)
def L(p,key=None):
    d=np.load(p); k=key or [k for k in ('private','client','global') if k in d.files][0]; return d[k].astype(np.int64)
cm={'core':L(R/'artifacts/f50/mh-f2-core-s42-r50/mh-f2-core-s42-r50/cms_r050.npz'),
    'oracle':L(R/'artifacts/y50/mh-y0-oracle-s42-r50/mh-y0-oracle-s42-r50/cms_r050.npz'),
    'nofl43':L(R/'artifacts/y50/mh-y6-nofloor-s43-r50/mh-y6-nofloor-s43-r50/cms_r050.npz'),
    'core43':L(R/'artifacts/f50/mh-f3-core-s43-r50/mh-f3-core-s43-r50/cms_r050.npz'),
    'fedavg':L(B/'fl2-fedavg/fl2-fedavg/cms_r050.npz'),
    'apfed':L(B/'fl2-apfed-r2/fl2-apfed-r2/cms_r050.npz')}
cc=sorted((B/'fl2-central-w1/fl2-central-w1').glob('cm_e*.npy'))[-1]; central=np.load(cc).astype(np.int64)
print({k:v.shape for k,v in cm.items()}, central.shape)
def prf(M):
    tp=np.diag(M); p=tp/np.maximum(M.sum(0),1); r=tp/np.maximum(M.sum(1),1); f=2*p*r/np.maximum(p+r,1e-12); return p,r,f
P={k:(v.sum(0)) for k,v in cm.items()}; P['central']=central
ks=list(P)
print('acc', {k:round(np.trace(M)/M.sum(),4) for k,M in P.items()})
print('macroF1', {k:round(prf(M)[2].mean(),4) for k,M in P.items()})
print('macroP', {k:round(prf(M)[0].mean(),4) for k,M in P.items()})
print('macroR', {k:round(prf(M)[1].mean(),4) for k,M in P.items()})
print('cls  '+' '.join(f'{k:>17s}' for k in ks))
for c in range(16):
    print(f'{c:3d} '+' '.join('%.3f/%.3f/%.3f'%tuple(x[c] for x in prf(P[k])) for k in ks))
# predicted share per class vs true share
print('pred share vs true (core, oracle, apfed)')
for c in range(16):
    t=P['core'].sum(1)[c]/P['core'].sum()
    print(c, round(t,4), *[round(P[k].sum(0)[c]/P[k].sum(),4) for k in ('core','oracle','apfed','fedavg')])
# FP sources for rare classes
for k in ('core','oracle','apfed'):
    M=P[k]
    for c in (1,7,11):
        col=M[:,c].copy(); col[c]=0; top=np.argsort(col)[::-1][:3]
        print(k,'class',c,'FP',col.sum(),'from',[(int(t),int(col[t])) for t in top])
# benign/td
for k in ks:
    M=P[k]; E=M.copy(); np.fill_diagonal(E,0)
    print(k,'errors',E.sum(),'b->td',M[0,13],'td->b',M[13,0],'share',round((M[0,13]+M[13,0])/E.sum(),3), 'non b/td errors', E.sum()-M[0,13]-M[13,0])
bins=[(0,256),(256,1024),(1024,4096),(4096,16384),(16384,10**9)]
print('cohort: clients, test share, pooled acc / macroF1 per method')
for lo,hi in bins:
    m=(n>=lo)&(n<hi)
    row=[]
    for k in cm:
        M=cm[k][m].sum(0); row.append('%s %.4f/%.4f'%(k,np.trace(M)/M.sum(),prf(M)[2][M.sum(1)>0].mean()))
    print(lo,hi,m.sum(),round(cm['core'][m].sum()/cm['core'].sum(),3),' | '.join(row))
# per-client: local-only vs absent
fs={k:client_f1(cm[k]) for k in cm}
for lo,hi in bins:
    m=(n>=lo)&(n<hi); print(lo,hi,{k:round(fs[k][m].mean(),4) for k in fs})
# class absent in train but present in test
sup=cm['core'].sum(2)
for k in cm:
    M=cm[k]; ab=(cnt==0)&(sup>0)
    tot=(M.sum(2)*ab).sum(); cor=(np.einsum('kcc->kc',M)*ab).sum(); print(k,'absent acc',round(cor/tot,4),'rows',tot)
tr=cnt>0; te=sup>0
for c in range(1,16):
    row=[]
    for k,M in cm.items():
        fp=M[:,:,c].sum(1)-M[:,c,c]   # per vehicle FP into c
        cells=[fp[tr[:,c]&te[:,c]].sum(),fp[tr[:,c]&~te[:,c]].sum(),fp[~tr[:,c]&te[:,c]].sum(),fp[~tr[:,c]&~te[:,c]].sum()]
        row.append('%s %6d/%6d/%5d/%6d'%(k[:4],*cells))
    print(c,' | '.join(row))
# total FP into attack classes at vehicles that have NO c in train
for k,M in cm.items():
    a=0;b=0
    for c in range(1,16):
        fp=M[:,:,c].sum(1)-M[:,c,c]; a+=fp[~tr[:,c]].sum(); b+=fp[tr[:,c]].sum()
    print(k,'FP into attack c at vehicles w/o c in train',a,'with c',b)
# Errors among large vehicles (>=16384) non b/td
m=n>=16384
for k,M in cm.items():
    S=M[m].sum(0); E=S.copy(); np.fill_diagonal(E,0); print(k,'large: err',E.sum(),'b/td',S[0,13]+S[13,0],'benign->attack(not td)',S[0,1:13].sum()+S[0,14:].sum(),'attack->benign(not td)',S[1:13,0].sum()+S[14:,0].sum())
