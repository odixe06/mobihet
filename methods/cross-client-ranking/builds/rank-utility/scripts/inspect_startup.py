"""Initial W&B identity/finite-step checks; no acceptance or ongoing stable-run polling."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import wandb

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,choices=(42,43),action='append')
    parser.add_argument('--output',default='startup.json');args=parser.parse_args();seeds=args.seed or [42,43]
    api=wandb.Api(timeout=25);entity=api.default_entity;rows=[]
    names={p.name for p in (ROOT/'mh').glob('*.py')}
    fixture=hashlib.sha256((ROOT/'papers/pilot/fixture/manifest.json').read_bytes()).hexdigest()
    for run in api.runs(f'{entity}/mobihet-cross-client-rank-utility',per_page=50):
        cfg=dict(run.config);arm=cfg.get('arm');seed=cfg.get('seed')
        if seed not in seeds or not str(arm).startswith('cr_'):continue
        folder=BASE/'notebooks'/f'cr-u-s{seed}-20261015-v1'
        manifest=json.loads((folder/'source_manifest.json').read_text());inputs=json.loads((folder/'input_manifest.json').read_text())
        warm=json.loads((folder/'warm-proof.json').read_text());source={('mh/' if n in names else 'proj/')+n:h for n,h in manifest.items() if n!='meta.json'}
        fingerprint=cfg.pop('fingerprint');cfg={k:v for k,v in cfg.items() if not k.startswith('_')}
        # W&B JSON canonicalizes integral floats (1.0 -> 1); restore argparse types before hashing.
        for key in ('lr','lr_min','clip','aux_weight','temperature','tau','weight_cap'):cfg[key]=float(cfg[key])
        identity=dict(config=cfg,source=source,fixture=fixture,fork=dict(round=2,sha256=inputs['warm_weights.pt'],fingerprint=warm['fingerprint']),
            inherited_rounds=0 if arm in ('cr_gaussian','cr_dml') else 2,primary='fixed true-class Laplace prior; head diagnostic')
        assert hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()==fingerprint,('W&B identity mismatch',run.id)
        keys=('round','loss','seconds','applied','skipped','progress/round','progress/client','progress/applied','progress/skipped','progress/real_ce','_timestamp')
        summary={k:run.summary.get(k) for k in keys}
        nonfinite=[k for k,v in summary.items() if isinstance(v,(int,float)) and not math.isfinite(v)]
        progress=max(summary.get('round') or 0,summary.get('progress/round') or 0)
        rows.append(dict(seed=seed,arm=arm,run=run.id,url=run.url,state=run.state,fingerprint=fingerprint,
            summary=summary,nonfinite=nonfinite,utility_after_cost_observed=bool(progress>=4 or arm not in ('cr_local','cr_fl'))))
    proof=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=rows,
        evidence='W&B fingerprint reconstructed from actual config/frozen source/fixture/fork; fresh project; initial progress, not terminal artifact acceptance',
        paired_after_cost=all(any(r['seed']==s and r['utility_after_cost_observed'] for r in rows) for s in (42,43)))
    (BASE/'probes'/args.output).write_text(json.dumps(proof,indent=2)+'\n')
    print('Rank startup',len(rows),'identities; paired after-cost',proof['paired_after_cost'],'nonfinite',sum(bool(r['nonfinite']) for r in rows))
    for r in rows:print(r['seed'],r['arm'],r['state'],r['summary']['round'],r['summary']['progress/round'])


if __name__=='__main__':main()
