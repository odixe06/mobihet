"""Read exact new-batch W&B identities and distinguish old history from fresh training."""
import ast
import argparse
import concurrent.futures
import datetime
from fnmatch import fnmatch
import json
import math
from pathlib import Path

import nbformat
import wandb

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pending-datasets-only',action='store_true')
    parser.add_argument('--tag',default='20261014')
    parser.add_argument('--since-utc',default='2026-10-14T10:39:00+00:00')
    parser.add_argument('--output')
    parser.add_argument('--only',action='append',default=[],help='notebook glob; skip runs whose startup is already stable')
    args=parser.parse_args()
    api=wandb.Api(timeout=25);entity=api.default_entity;jobs={}
    for build in ('full-validation','hybrid-full','boundary-cells'):
        for path in (ROOT/'papers'/build/'notebook').glob(f'*{args.tag}*/PUSHED'):
            folder=path.parent
            if args.only and not any(fnmatch(folder.name,pattern) for pattern in args.only):continue
            if args.pending_datasets_only and 'dsresume' not in folder.name:continue
            if args.tag=='20261014' and 'dsresume' not in folder.name and folder.name.startswith(('hybf1','hybf3')):continue
            metadata=json.loads((folder/'kernel-metadata.json').read_text())
            nb=nbformat.read(folder/metadata['code_file'],as_version=4)
            plan=json.loads((folder/'resume-plan.json').read_text())
            project='mobihet-'+build
            if build!='boundary-cells':
                assignment=next(n for n in ast.parse(nb.cells[3].source).body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='WANDB_PROJECT' for t in n.targets))
                project=ast.literal_eval(assignment.value)
            for name,value in plan.items():
                name=name if name.endswith(('_s42','_s43')) else name+'_s42'
                locator=f'{entity}/{project}/{name}-{value["fingerprint"][:8]}'
                jobs[locator]=dict(kernel=metadata['id'],run=name,project=project,
                    fingerprint=value['fingerprint'],parent_round=value['last_round'],locator=locator)
    def read(item):
        try:run=api.run(item['locator'])
        except wandb.errors.CommError:return dict(item,state='NOT_VISIBLE',fresh_training=False)
        assert run.config['fingerprint']==item['fingerprint']
        keys=('round','loss','lr','applied','skipped','seconds','_timestamp','progress/round_in_flight',
            'progress/applied','progress/skipped','progress/loss','progress/grad_norm','progress/max_logit',
            'progress/softmax_entropy','preflight/online')
        summary={key:run.summary.get(key) for key in keys}
        r=summary.get('round') or 0;flight=summary.get('progress/round_in_flight') or 0
        progressing=r>item['parent_round'] or flight>item['parent_round'] and (summary.get('progress/applied') or 0)>0
        stamp=summary.get('_timestamp') or 0
        fresh=stamp>=datetime.datetime.fromisoformat(args.since_utc).timestamp()
        nonfinite=[k for k,v in summary.items() if isinstance(v,(int,float)) and not math.isfinite(v)]
        return dict(item,state=run.state,url=run.url,summary=summary,fresh_training=bool(progressing and fresh),nonfinite=nonfinite)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(read,jobs.values()))
    record=dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),runs=rows,
        evidence='Read-only initial snapshot; exact run identity; old resumed history not counted as fresh training; not terminal audit')
    filename='startup-dataset-resume-2026-10-14.json' if args.pending_datasets_only else 'startup-batch-2026-10-14.json'
    (ROOT/'docs/operations'/(args.output or filename)).write_text(json.dumps(record,indent=2)+'\n')
    print('Initial W&B:',len(rows),'identities;',sum(x['fresh_training'] for x in rows),'fresh training;',
        sum(bool(x.get('nonfinite')) for x in rows),'nonfinite')
    for row in rows:
        print(row['kernel'],row['run'],row['state'],'fresh',row['fresh_training'],
              'round',row.get('summary',{}).get('round'),'flight',row.get('summary',{}).get('progress/round_in_flight'))


if __name__=='__main__':main()
