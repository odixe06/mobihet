"""Read exact BoundaryCells run IDs derived from frozen source/config; no run mutation."""
import hashlib
import json
from pathlib import Path
import sys
import datetime
import wandb

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
BASE=ROOT/'papers/boundary-cells'


def main():
    api=wandb.Api(timeout=25);entity=api.default_entity
    observations=[]
    for seed in (42,43):
        folder=BASE/f'notebook/bc-cal-s{seed}-20261013-initial2-v1'
        manifest=json.loads((folder/'source_manifest.json').read_text())
        proj=set(json.loads((BASE/'local-proof-initial2.json').read_text())['source_sha'])
        source={('proj/' if name in proj else 'mh/')+name:sha for name,sha in manifest.items() if name.endswith('.py')}
        fixture=hashlib.sha256((BASE/'input'/('minhtran0601' if seed==42 else 'dinhan25')/'manifest.json').read_bytes()).hexdigest()
        for arm in ('bc_single','bc_direct2_fl','bc_direct4_fl','bc_teacher2_fl','bc_teacher4_fl'):
            config=dict(arm=arm,seed=seed,rounds=20,batch=512,private='dagsnet',lr=.001,lr_min=.00001,
                        clip=1.,eval_every=1,aux_batch=256,aux_weight=.5,temperature=1.,tau=.05,
                        weight_cap=10.,smoke=False)
            fp=hashlib.sha256(json.dumps(dict(config=config,source=source,fixture=fixture),sort_keys=True).encode()).hexdigest()
            locator=f'{entity}/mobihet-boundary-cells/{arm}_s{seed}-{fp[:8]}'
            try:
                run=api.run(locator)
            except wandb.errors.CommError:
                observations.append(dict(seed=seed,arm=arm,locator=locator,state='NOT_VISIBLE_YET'))
                continue
            assert run.config['fingerprint']==fp
            keys=('round','loss','lr','seconds','applied','skipped','preflight/online',
                  'progress/round','progress/client','progress/applied','progress/skipped','progress/real_ce',
                  'private/group/gp/f1_macro','private/group/gp/accuracy')
            summary={key:run.summary.get(key) for key in keys}
            observations.append(dict(seed=seed,arm=arm,locator=locator,url=run.url,state=run.state,
                heartbeat=summary,config_fingerprint=fp))
    record=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=observations,
                evidence='W&B read-only live snapshot, not terminal artifact/cost/utility audit')
    (BASE/'startup-live-latest.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':main()
