"""Train fold -> val fold only. Never reads official test. How much class information is beyond 2nd-order copula, and sender fusion."""
import glob,json,time,sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from scipy.special import ndtri, logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import f1_score, precision_recall_fscore_support as prf
t0=time.time()
order=json.load(open('/home/odixe/project/mobihet/knowledge/input_order.json'))
cols=order if isinstance(order,list) else order.get('order') or order.get('inputs')
assert len(cols)==77
fs=sorted(glob.glob('/home/odixe/project/mobihet-features/data/v4-scaled/train/scenario=*/world=*/part-*.parquet'))
rng=np.random.default_rng(7); pick=rng.choice(len(fs),150,replace=False)
Xt,Yt,Xv,Yv,meta=[],[],[],[],[]
for i in pick:
    t=pq.ParquetFile(fs[i]).read(columns=cols+['client_id','sender_id','label','rcv_time_ns','fold']).to_pandas()
    fold=t['fold'].astype(str).values
    a=t[fold=='train'].sample(frac=.16,random_state=int(i)); b=t[fold=='val']
    Xt.append(a[cols].to_numpy(np.float32)); Yt.append(a['label'].to_numpy(np.int16))
    Xv.append(b[cols].to_numpy(np.float32)); Yv.append(b['label'].to_numpy(np.int16)); meta.append(b[['client_id','sender_id','rcv_time_ns']].assign(f=int(i)))
Xt=np.concatenate(Xt);Yt=np.concatenate(Yt);Xv=np.concatenate(Xv);Yv=np.concatenate(Yv);meta=pd.concat(meta,ignore_index=True)
print('train',Xt.shape,'val',Xv.shape,'classes',np.unique(Yt),time.time()-t0,flush=True)
sub=rng.choice(len(Xv),min(600000,len(Xv)),replace=False); 
C=16;D=77;NB=256
def rep(name,y,p):
    f=f1_score(y,p,average=None,labels=range(16),zero_division=0); pr,rc,_,_=prf(y,p,labels=[13],average=None,zero_division=0)
    print(f'{name:24s} acc {np.mean(y==p):.4f} macroF1 {f.mean():.4f} benign {f[0]:.4f} tD F1 {f[13]:.4f} P {pr[0]:.3f} R {rc[0]:.3f} other15 {np.delete(f,13).mean():.4f}',flush=True)
# class-conditional Gaussian copula, exact method fit (256 bins, normal scores), scored as a Bayes classifier
bins=lambda x:np.clip(((x+4)*(NB/8)).astype(np.int32),0,NB-1)
logp=np.zeros((len(Xv),C),np.float32); pri=np.bincount(Yt,minlength=C)/len(Yt)
def copula_logp(X,c,tab):
    cdf,mass,mu,Li,ld=tab; b=bins(X); r=np.arange(D)
    u=cdf[r,b]-mass[r,b]/2; z=ndtri(np.clip(u,1e-4,1-1e-4))
    w=(z-mu)@Li.T
    logc=-.5*(w*w).sum(1)-ld+.5*(z*z).sum(1)
    return logc+np.log(np.maximum(mass[r,b],1e-7)*(NB/8)).sum(1)
gauss=np.zeros_like(logp)
for c in range(C):
    x=Xt[Yt==c]
    if len(x)<200: logp[:,c]=-1e9; gauss[:,c]=-1e9; continue
    b=bins(x); hist=np.stack([np.bincount(b[:,j],minlength=NB) for j in range(D)]).astype(np.float64)+.01
    mass=hist/hist.sum(1,keepdims=True); cdf=mass.cumsum(1)
    z=ndtri(np.clip(cdf[np.arange(D),b]-mass[np.arange(D),b]/2,1e-4,1-1e-4))
    mu=z.mean(0); cov=np.cov(z.T)+1e-4*np.eye(D); L=np.linalg.cholesky(cov); Li=np.linalg.inv(L); ld=np.log(np.diag(L)).sum()
    tab=(cdf,mass,mu,Li,ld)
    for i in range(0,len(Xv),200000): logp[i:i+200000,c]=copula_logp(Xv[i:i+200000],c,tab)
    m2=x.mean(0); c2=np.cov(x.T.astype(np.float64))+1e-3*np.eye(D); L2=np.linalg.cholesky(c2); w=(Xv-m2)@np.linalg.inv(L2).T
    gauss[:,c]=-.5*(w*w).sum(1)-np.log(np.diag(L2)).sum()
print('copula fit',time.time()-t0,flush=True)
lp=np.log(np.maximum(pri,1e-9)); rep('QDA gauss, prior',Yv,(gauss+lp).argmax(1)); rep('Copula-Bayes, prior',Yv,(logp+lp).argmax(1)); rep('Copula-Bayes, uniform',Yv,logp.argmax(1))
# higher-order pooled model on same train rows
idx=rng.choice(len(Xt),min(1500000,len(Xt)),replace=False)
m=H(max_iter=60,max_leaf_nodes=63,learning_rate=.12,max_bins=128,early_stopping=False,random_state=0).fit(Xt[idx],Yt[idx])
print('hgb fit',time.time()-t0,flush=True)
P=np.concatenate([m.predict_proba(Xv[i:i+300000]) for i in range(0,len(Xv),300000)]).astype(np.float32)
cl=m.classes_; full=np.full((len(Xv),C),1e-7,np.float32); full[:,cl]=np.maximum(P,1e-7)
rep('HGB pooled',Yv,full.argmax(1))
# only benign vs timeDelay
k=(Yv==0)|(Yv==13)
from sklearn.metrics import roc_auc_score as auc
def A(nm,S):
    d=np.nan_to_num(S[k,13]-S[k,0],posinf=1e6,neginf=-1e6); print(nm,'0-vs-13 AUC',round(auc(Yv[k]==13,d),4),flush=True)
A('HGB',np.log(full)); A('Copula',np.maximum(logp+lp,-1e4))
# causal sender fusion: mean log-prob over previous messages of same (file,client,sender), no future rows
def fuse(S):
    o=meta.assign(i=np.arange(len(meta))).sort_values(['f','client_id','sender_id','rcv_time_ns'])
    g=o.groupby(['f','client_id','sender_id'],sort=False).ngroup().to_numpy(); ii=o['i'].to_numpy(); s=S[ii].astype(np.float64)
    cs=s.cumsum(0); start=np.r_[0,np.flatnonzero(np.diff(g))+1]; base=np.zeros_like(cs); 
    off=np.repeat(np.r_[np.zeros((1,C)),cs[start[1:]-1]],np.diff(np.r_[start,len(g)]),axis=0); cnt=np.arange(len(g))-np.repeat(start,np.diff(np.r_[start,len(g)]))+1
    out=np.zeros_like(s); out[ii]=(cs-off)/cnt[:,None]; return out
fh=fuse(np.log(full)); fc=fuse(np.maximum(logp+lp,-1e4))
rep('HGB + sender fusion',Yv,fh.argmax(1)); rep('Copula + sender fusion',Yv,fc.argmax(1)); A('HGB fused',fh); A('Copula fused',fc)
print('rows per pair in val: median',meta.groupby(['f','client_id','sender_id']).size().median(),'done',time.time()-t0)
