"""Read-only full train-label scan and own-test CM diagnostics; no test-based HP selection."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.metrics import ten

ap=argparse.ArgumentParser()
ap.add_argument('--data',type=Path,required=True)
ap.add_argument('--run',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args()
n=np.zeros((5343,16),dtype=np.int64)
files=sorted((a.data/'train').rglob('*.parquet'))
for f in files:
 for b in pq.ParquetFile(f).iter_batches(columns=['client_id','label'],batch_size=262144):
  ids=b.column(0).to_numpy().astype(np.int64); y=b.column(1).to_numpy().astype(np.int64)
  n+=np.bincount(ids*16+y,minlength=5343*16).reshape(5343,16)
assert n.sum()==37361273
z=np.load(a.run/'cms_r010.npz'); assert np.array_equal(z['ids'],np.arange(5343))
cm=z['private']; t=cm.sum(2)
f1=np.array([ten(c)['f1_macro'] for c in cm])
cohorts={}
for lo,hi in [(0,256),(256,1024),(1024,4096),(4096,16384),(16384,100000)]:
 sel=(n.sum(1)>=lo)&(n.sum(1)<hi)
 cohorts[f'n_train[{lo},{hi})']={'clients':int(sel.sum()),'mean_f1':float(f1[sel].mean()),'p10':float(np.percentile(f1[sel],10)),'pooled_f1':ten(cm[sel].sum(0))['f1_macro']}
for name,sel in [('test_class_seen_in_train',n>0),('test_class_absent_in_train',n==0),
                 ('test_class_train_count_1_16',(n>0)&(n<=16)),
                 ('test_class_train_count_17_64',(n>16)&(n<=64)),
                 ('test_class_train_count_65_256',(n>64)&(n<=256)),
                 ('test_class_train_count_gt256',n>256)]:
 support=int(t[sel].sum()); tp=np.diagonal(cm,axis1=1,axis2=2)
 cohorts[name]={'rows':support,'accuracy':float(tp[sel].sum()/max(support,1))}
pool=cm.sum(0); err=pool.copy(); np.fill_diagonal(err,0)
pairs=sorted([(int(err[i,j]),i,j) for i in range(16) for j in range(16) if i!=j],reverse=True)[:12]
snapshot=torch.load(a.run/'models/r010.pt',map_location='cpu',weights_only=True)
q=snapshot['Q'].numpy(); q=np.maximum(q,1e-12); q/=q.sum(-1,keepdims=True)
out={'train_files':len(files),'train_rows':int(n.sum()),'test_rows':int(t.sum()),'classes_per_client':{'min':int((n>0).sum(1).min()),'median':float(np.median((n>0).sum(1)))},'cohorts':cohorts,'top_errors_true_pred':pairs,'pooled':ten(pool),'Q_true_class_prob':np.diag(q).tolist(),'Q_entropy':(-(q*np.log(q)).sum(1)).tolist(),'Q_argmax':q.argmax(1).tolist(),'scope':'Full train label/count scan; own-test descriptive diagnosis, not causal proof; Q collected during train, not post-epoch eval'}
a.output.write_text(json.dumps(out,indent=2))
print(json.dumps({k:out[k] for k in ['cohorts','top_errors_true_pred','Q_true_class_prob','Q_argmax']},indent=2))
