"""Inspect actual audited cell-target packets; bounded group diagnostic, no SGD or tuning."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE/'source/proj'))
from cells import C,PAIRS,EDGES,decode,lookup
from mh.metrics import metrics_from_confusion


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    fx=ROOT/'.compat/papers/pilot/fixture';fixture=json.loads((fx/'manifest.json').read_text())
    for name in ('group_X.npy','group_y.npy'):assert sha(fx/name)==fixture['files'][name]
    x=np.load(fx/'group_X.npy',mmap_mode='r');y=np.load(fx/'group_y.npy')
    rng=np.random.default_rng(1729)
    index=np.concatenate([rng.choice(np.flatnonzero(y==c),128,replace=False) for c in range(C)])
    x,y=x[index].astype('float32'),y[index].astype('int64');rows=[]
    for seed in (42,43):
        parent=BASE/'artifacts/kaggle'/f'bc-u-s{seed}-20261014-stage01-v1'
        audited=json.loads((parent/'AUDITED.json').read_text())
        assert sha(parent/'source/cells.py')==sha(BASE/'source/proj/cells.py')
        for p in (2,4):
            name=f'bc_direct{p}_fl_s{seed}';run=parent/'runs'/name
            n=next(a['rounds'] for a in audited['audits'] if a['run']==name)
            cfg=json.loads((run/'config.json').read_text());assert cfg['fixture']==sha(fx/'manifest.json')
            record=json.loads((run/'complete'/f'round_{n:03d}.json').read_text())
            path=run/'knowledge'/f'round_{n:03d}.bin';assert sha(path)==record['payload_sha']
            identity=hashlib.sha256(json.dumps(dict(pairs=PAIRS[:p],edges=EDGES.tolist(),fixture=fixture['source_id']),sort_keys=True).encode()).digest()[:16]
            payload=decode(path.read_bytes(),2,p,identity)
            table=np.frombuffer(payload,dtype='<f4',count=p*16*C).reshape(p,16,C)
            mask=np.unpackbits(np.frombuffer(payload[table.size*4:-4*C],dtype='u1'),bitorder='little').reshape(p,16,C)
            assert np.array_equal(mask,np.repeat(mask[:,:,:1],C,axis=-1))
            prob,valid=lookup(x,fixture['features'],table,mask[:,:,0].astype(bool))
            pred=prob.argmax(1);cm=np.bincount(y*C+pred,minlength=C*C).reshape(C,C)
            rows.append(dict(seed=seed,pairs=p,audited_packet_round=n,payload_sha=sha(path),
                coverage=float(valid.mean()),eligible_cells=int(mask[:,:,0].sum()),
                entropy_normalized=float((-prob*np.log(np.maximum(prob,1e-12))).sum(1).mean()/np.log(C)),
                mean_true_class_probability=float(prob[np.arange(len(y)),y].mean()),
                mean_max_probability=float(prob.max(1).mean()),metrics=metrics_from_confusion(cm)))
    result=dict(scope='actual full-T count packets; balanced128/class group=2048; table diagnostic is not DAGSNet utility or causal isolation',
                fixture_sha=sha(fx/'manifest.json'),official_test_read=False,model_training=False,records=rows)
    (BASE/'probes/direct-target-forensics-20261015.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
