"""Read only the W&B identities implied by each frozen pilot config/source/fixture."""
import datetime
import argparse
import hashlib
import json
import math
from pathlib import Path
import wandb

ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--kernel',default='');parser.add_argument('--output')
args=parser.parse_args()
api=wandb.Api(timeout=30);entity=api.default_entity
fixture=hashlib.sha256((ROOT/'.compat/papers/pilot/fixture/manifest.json').read_bytes()).hexdigest()
rows=[]
for family,build,project in [('class-tail-risk','class-tail-pilot','mobihet-class-tail'),
                            ('proxy-countsketch-repair','contractive-ef-pilot','mobihet-contract-ef')]:
    base=ROOT/'methods'/family/'builds'/build
    for marker in sorted((base/'notebooks').glob('*/PUSHED')):
        folder=marker.parent;meta=json.loads((folder/'kernel-metadata.json').read_text())
        if args.kernel not in meta['id']:continue
        manifest=json.loads((folder/'source_manifest.json').read_text())
        seed=42 if 's42' in folder.name else 43
        if family=='class-tail-risk':
            source={('proj/' if (base/'source/proj'/name).exists() else 'mh/')+name:sha
                    for name,sha in manifest.items() if name!='meta.json'}
            arms=['ct_local','ct_fixed','ct_own','ct_fl','ct_shuffle','ct_classmass','ct_gaussian','ct_dml']
        else:
            source={k:v for k,v in manifest.items() if k!='meta.json'}
            arms=['local','a_own','a_dense','a_small_noef','a_small_contract_ef','a_medium_noef','a_medium_contract_ef']
        for arm in arms:
            config=dict(arm=arm,seed=seed,rounds=20,batch=512,private='dagsnet',lr=.001,lr_min=.00001,
                        clip=1.,eval_every=1,aux_batch=256,aux_weight=1.,temperature=2.,tau=.05,weight_cap=10.,smoke=False)
            fp=hashlib.sha256(json.dumps(dict(config=config,source=source,fixture=fixture),sort_keys=True).encode()).hexdigest()
            locator=f'{entity}/{project}/{arm}_s{seed}-{fp[:8]}'
            parent_round=2 if folder.name.startswith('ct-u-') else 0
            item=dict(kernel=meta['id'],arm=arm,seed=seed,fingerprint=fp,locator=locator,parent_round=parent_round)
            try:run=api.run(locator)
            except wandb.errors.CommError:
                rows.append(dict(item,state='NOT_VISIBLE'));continue
            assert run.config['fingerprint']==fp
            summary={key:run.summary.get(key) for key in ('round','loss','applied','skipped','seconds','_timestamp')}
            nonfinite=[key for key,v in summary.items() if isinstance(v,(int,float)) and not math.isfinite(v)]
            rows.append(dict(item,state=run.state,summary=summary,nonfinite=nonfinite,url=run.url,
                             fresh_training=(summary.get('round') or 0)>parent_round))
output=ROOT/(args.output or 'docs/operations/pilot-startup-2026-10-15.json')
previous=json.loads(output.read_text()) if output.exists() else {}
record=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=rows,
            evidence='Exact config/source/fixture identity; read-only startup; not artifact/utility evidence')
if 'owner_console_seed42' in previous:record['owner_console_seed42']=previous['owner_console_seed42']
output.write_text(json.dumps(record,indent=2)+'\n')
for r in rows:
    if r['state']!='NOT_VISIBLE':print(r['kernel'],r['arm'],r['state'],r['summary'],flush=True)
print('New pilots:',len(rows),'identities;',sum(r['state']!='NOT_VISIBLE' for r in rows),'visible;',
      sum(bool(r.get('nonfinite')) for r in rows),'nonfinite')
