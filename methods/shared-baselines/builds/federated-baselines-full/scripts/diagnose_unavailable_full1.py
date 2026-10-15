import json,netrc,os,sys
from pathlib import Path
import wandb
root=Path.cwd();base=root/'methods/shared-baselines/builds/federated-baselines-full'
rid='full1-s42-20261015-q1-v1';plan=json.loads((base/'notebooks'/rid/'resume-plan.json').read_text())
api=wandb.Api(timeout=25);entity=api.default_entity;key=netrc.netrc().authenticators('api.wandb.ai')[2]
records=[]
for arm,item in plan.items():
 locator=f'{entity}/mobihet-full-validation/{arm}_s42-{item["fingerprint"][:8]}'
 run=api.run(locator);assert run.config['fingerprint']==item['fingerprint']
 record=dict(run=locator,state=run.state,summary={k:run.summary.get(k) for k in ('round','progress/round_in_flight','progress/applied','_timestamp','_runtime')})
 names=[f.name for f in run.files()];record['files']=names
 if 'output.log' in names:
  private=base/'private_push'/rid/'wandb-forensics'/arm;private.mkdir(parents=True,exist_ok=True,mode=0o700)
  previous=os.umask(0o077)
  try:run.file('output.log').download(root=str(private),replace=True)
  finally:os.umask(previous)
  log=(private/'output.log');log.chmod(0o600)
  clean=log.read_text().replace(key,'[REDACTED]');record['tail']='\n'.join(clean.splitlines()[-35:])
 records.append(record)
proof=root/'docs/operations/full1-unavailable-2026-10-15-afternoon.json'
proof.write_text(json.dumps(dict(kernel='accanniez/mobihet-'+rid,output_files=0,executed_notebook_http=404,kaggle_failure_message='',wandb=records,evidence='W&B operational logs, not model/checkpoint evidence'),indent=2)+'\n')
for r in records:print(json.dumps(r,indent=2))
