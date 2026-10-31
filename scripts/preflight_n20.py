"""Bounded real-data preflight of N20's four existing loss combinations; no utility claim."""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from core import data, fl, grouped, local, metrics
from core.dagsnet import width, DAGSNet
from methods.mobihet import Method
from scripts.run_fl import PROTOCOL


def main():
    out=Path('runs/n-study');out.mkdir(exist_ok=True)
    counts=np.load('evidence/research-data/train_counts.npy')
    # Train-only deterministic fixture, cover all classes with a small total row budget.
    candidates=np.flatnonzero(counts.sum(1)<1024);missing=set(range(16));ids=[]
    while missing:
        cid=max(candidates,key=lambda k:(sum(counts[k,c]>0 for c in missing)/counts[k].sum(),-int(k)))
        got={c for c in missing if counts[cid,c]>0};assert got
        ids.append(int(cid));missing-=got
    ids=sorted(ids);cache=out/'smoke-cache';cache.mkdir(exist_ok=True)
    np.save(cache/'train_sizes.npy',counts.sum(1))
    hp=dict(PROTOCOL,arm='none',synth='copula',synth_rows=64,kd_weight='fixed',q_codec='mean_fp16',
            group_rows=64*len(ids),seed=42,q_collect='online',q_agg='count',kd_support='off')
    env=fl.Env(width(1),hp,clients=ids,cache=str(cache))
    assert len(env.train.Y)==int(counts[ids].sum())
    if env.dev.type=='cuda':
        grouped.compile_all()
        ok,rep=local.compile_gate(env.lay,env.train,env.cfg,hp,torch.Generator(device=env.dev).manual_seed(0))
        assert ok,rep
    else:rep={'mode':'CPU eager, no T4 equivalence claim'}
    records=[];r1reference=None
    rng=env.gen.get_state()
    for real,synth in [(False,'soft'),(True,'soft'),(False,'fallback'),(True,'fallback')]:
        hp.update(real_kd=real,real_kd_gate=real,synth_target=synth)
        env.gen.set_state(rng)
        m=Method(env);rows=[]
        for r in [1,2]:
            teacher=None if m.Q is None else m.Q.clone()
            info=m.round(r,fl.cosine_lr(r,50,1e-3,1e-5))
            assert info['skipped_frac']==0 and np.isfinite(info['train_loss'])
            assert torch.isfinite(m.priv_p).all() and torch.isfinite(m.priv_b).all()
            assert info['bytes_up']==info['bytes_down']==544
            cm=m.evaluate()['private'];assert cm.sum()==len(env.test.Y)
            assert np.array_equal(cm.sum(2),np.stack([np.bincount(env.test.Y[env.test.off[k]:env.test.off[k+1]].cpu().numpy(),minlength=16) for k in range(env.K)]))
            rows.append({'round':r,'teacher_present':teacher is not None,'metrics':metrics.ten(cm.sum(0)),**info})
            if r==1:
                if r1reference is None:r1reference=(m.priv_p.clone(),m.priv_b.clone(),m.Q.clone())
                else:
                    for got,expected in zip((m.priv_p,m.priv_b,m.Q),r1reference):assert torch.equal(got,expected)
        state=m.state();path=out/f'smoke-{real}-{synth}.pt';torch.save(state,path)
        loaded=torch.load(path,map_location=env.dev,weights_only=True)
        restored=Method(env);restored.load_state(loaded)
        model=DAGSNet(env.cfg)
        model.load_state_dict(env.lay.state_dict(restored.priv_p[0].cpu(),restored.priv_b[0].cpu()),strict=True)
        assert sum(p.numel() for p in model.parameters())==395408
        for v in model.state_dict().values():assert torch.isfinite(v).all()
        torch.testing.assert_close(restored.priv_p,m.priv_p,rtol=0,atol=0)
        torch.testing.assert_close(restored.priv_b,m.priv_b,rtol=0,atol=0)
        assert np.array_equal(restored.evaluate()['private'],cm)
        records.append({'real_kd':real,'synth_target':synth,'rows':rows,'state_roundtrip':'exact','state_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        del restored,m,loaded,state
    (out/'smoke-audit.json').write_text(json.dumps({'ids':ids,'train_rows':len(env.train.Y),'test_rows':len(env.test.Y),
        'compile_gate':rep,'device':str(env.dev),'group_rows_fixture':hp['group_rows'],'group_rows_production':4096,'r1_four_cells_bitexact':True,
        'scope':'Real-data loss/codec/CM/state integration; local sm86 not T4 proof, no utility/generalization/exact-resume claim','cells':records},indent=2)+'\n')
    print('N20_REAL_PREFLIGHT_PASS',ids,len(env.train.Y),flush=True)


if __name__=='__main__':main()
