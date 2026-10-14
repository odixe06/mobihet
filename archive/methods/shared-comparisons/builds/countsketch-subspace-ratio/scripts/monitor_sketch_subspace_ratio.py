"""Read-only live W&B status, matched to exact expected immutable V2 fingerprints."""
import argparse
import datetime
import json
import netrc
from types import SimpleNamespace
from pathlib import Path
import wandb
import requests


def http_runs(build='v2'):
    """Read-only fallback when SDK local service cannot initialize."""
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    query='''query($cursor:String){project(name:"mobihet-BUILD",entityName:"21522798-uit"){
      runs(first:100,after:$cursor){edges{node{name state config summaryMetrics}}
      pageInfo{hasNextPage endCursor}}}}'''.replace('BUILD',build)
    cursor=None
    while True:
        response=requests.post('https://api.wandb.ai/graphql',auth=('api',key),
            json=dict(query=query,variables=dict(cursor=cursor)),timeout=30)
        response.raise_for_status(); payload=response.json(); assert not payload.get('errors'), 'W&B GraphQL read failed'
        project=payload['data']['project']
        if project is None: return
        result=project['runs']
        for edge in result['edges']:
            node=edge['node']; cfg=json.loads(node['config'])
            config={k:v.get('value') if isinstance(v,dict) and 'value' in v else v for k,v in cfg.items()}
            yield SimpleNamespace(id=node['name'],state=node['state'],config=config,
                summary=json.loads(node['summaryMetrics']),url=f'https://wandb.ai/21522798-uit/mobihet-{build}/runs/{node["name"]}')
        if not result['pageInfo']['hasNextPage']: return
        cursor=result['pageInfo']['endCursor']


def main():
    p=argparse.ArgumentParser(); p.add_argument('--family')
    p.add_argument('--build',choices=['v2','v3','v4','v5','v6','v7','v8','v9','v10'],default='v2')
    p.add_argument('--seed',type=int,choices=[42,43],help='only an unresolved seed; stable kernels no longer polled')
    p.add_argument('--http',action='store_true',help='read directly without local SDK service'); a=p.parse_args()
    base=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])/f'papers/{a.build}'
    registry=json.loads((base/'wandb-runs.json').read_text())
    wanted={v['run_id']:v for v in registry if (a.family is None or v['family']==a.family) and (a.seed is None or v['seed']==a.seed)}
    rows=[]
    try:
        if a.http:
            found=list(http_runs(a.build)); transport='authenticated read-only GraphQL'
        else:
            api=wandb.Api(timeout=30)
            found=list(api.runs(f'21522798-uit/mobihet-{a.build}',filters={'config.smoke':False}))
            transport='W&B SDK'
    except wandb.errors.AuthenticationError:
        print('SDK authentication service unavailable; authenticated HTTP read fallback')
        found=list(http_runs(a.build)); transport='authenticated read-only GraphQL'
    try:
        for run in found:
            if run.id not in wanted: continue
            expected=wanted[run.id]
            assert run.config.get('fingerprint')==expected['fingerprint'], 'wrong-source W&B run'
            row={k:expected[k] for k in ('family','seed','arm','run_id','locator')}
            row.update(state=run.state,url=run.url,round=run.summary.get('round'),timestamp=run.summary.get('_timestamp'))
            for key in ('loss','seconds','version','applied','skipped','private/group/gp/f1_macro',
                        'private/group/head/f1_macro','proxy/group/head/f1_macro','private/real_ce',
                        'private/auxiliary','proxy/positive_ce','proxy/negative_ce','public_teacher_applied','algorithm_seconds',
                        'algorithm/teacher_fit_mse','algorithm/residual_norm','algorithm/sampler_acceptance',
                        'algorithm/sampler_max_dwell','sampler_probe/proposed_gate','sampler_probe/movement'):
                if key in run.summary: row[key]=run.summary[key]
            rows.append(row)
    except (wandb.errors.CommError,ValueError) as error:
        print('W&B project not available yet:',type(error).__name__); return
    snapshot=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  evidence='live W&B, not pulled artifact verification',transport=transport,
                  runs=sorted(rows,key=lambda v:(v['seed'],v['arm'])))
    (base/('live-status.json' if a.seed is None else f'live-status-s{a.seed}.json')).write_text(json.dumps(snapshot,indent=2))
    print(json.dumps(snapshot,indent=2))


if __name__=='__main__': main()
