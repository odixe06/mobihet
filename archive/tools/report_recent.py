"""V7–V10 reports and locked gates, only from pulled AUDITED.json evidence."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file())
sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',choices=['v7','v8','v9','v10'],required=True);a=p.parse_args()
    base=ROOT/'papers'/a.build;runs={};metric_rows=[];classes=[]
    for file in sorted((base/'kaggle').glob('*/AUDITED.json')):
        audited=json.loads(file.read_text())
        for item in audited['audits']:
            arm=item['arm'];seed=audited['seed'];folder=file.parent/'runs'/f'{arm}_s{seed}'
            history=[json.loads(f.read_text()) for f in sorted((folder/'complete').glob('round_*.json'))]
            result=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else {}
            assert len(history)==item['rounds']
            runs[(arm,seed)]=dict(audit=item,summary=audited['summary'][arm],history=history,
                result=result,path=str(folder.relative_to(ROOT)))
            for row in history:
                for role,holds in row.get('eval',{}).items():
                    for hold,rules in holds.items():
                        for rule,ev in rules.items():
                            assert set(ev['mean_metrics'])==set(METRIC_KEYS)
                            common=dict(arm=arm,seed=seed,round=row['round'],role=role,holdout=hold,rule=rule)
                            metric_rows.append(dict(common,**ev['mean_metrics']))
                            classes.extend(dict(common,class_id=c,f1=value) for c,value in enumerate(ev['f1_per_class']))
    def ev(arm,seed,role='private',hold='group',rule='gp'):
        run=runs.get((arm,seed))
        if run and run['audit']['complete'] and len(run['history'])==20:
            return run['history'][-1]['eval'][role][hold][rule]
    def score(*args,**kwargs):
        value=ev(*args,**kwargs);return value['f1_macro'] if value else None
    def local(seed,hold='group'):
        audit=ROOT/f'papers/v2/kaggle/v2a-s{seed}-20261011-v1/AUDITED.json'
        if not audit.exists():return None
        record=json.loads(audit.read_text());item=next(v for v in record['audits'] if v['arm']=='local')
        assert item['complete']
        return json.loads((audit.parent/f'runs/local_s{seed}/result.json').read_text())['history'][-1]['eval']['private'][hold]['gp']
    gates={}
    for seed in (42,43):
        if a.build=='v7':
            for rank in (4,8):
                arms=[f'n_nys{rank}_fl',f'n_nys{rank}_own',f'n_nys{rank}_shuffle',f'n_lr{rank}']
                values=[score(x,seed) for x in arms];ref=local(seed)
                if all(v is not None for v in values) and ref:
                    f,own,shuffle,exact=values;arm=arms[0]
                    quality=f-exact>=-.01 and f-own>=.01 and f-shuffle>=.01 and f-ref['f1_macro']>=.03
                    quality=quality and ev(arm,seed)['benign_f1']>=ref['benign_f1']-.03 and score(arm,seed,hold='old')>local(seed,'old')['f1_macro']
                    wire=lambda name:sum(v[-1] for v in runs[(name,seed)]['result']['bytes'])
                    ratio=wire(arm)/wire('n_full')
                    gates[f'rank{rank}_s{seed}']=dict(quality=bool(quality),compression=ratio<=.65,byte_ratio=ratio,
                        vs_exact=f-exact,vs_own=f-own,vs_shuffle=f-shuffle)
        elif a.build=='v8':
            names=['h_hybrid','h_gaussian','h_dml','h_hybrid_own','h_local']
            v=[score(n,seed) for n in names]
            if all(x is not None for x in v):
                hybrid,gauss,dml,own,lo=v
                quality=hybrid-max(gauss,dml)>=.01 and hybrid-own>=.01 and hybrid-lo>=.03
                quality=quality and ev('h_hybrid',seed)['benign_f1']>=ev('h_local',seed)['benign_f1']-.03
                quality=quality and score('h_hybrid',seed,hold='old')>score('h_local',seed,hold='old')
                gates[f'hybrid_s{seed}']=dict(quality=bool(quality),vs_stronger=hybrid-max(gauss,dml),vs_own=hybrid-own,
                    interaction=(hybrid-dml)-(gauss-lo))
        elif a.build=='v9':
            for m in (64,512):
                modes=['fl','own','shuffle','classmean','hard','frozen'];v=[score(f'r{m}_{s}',seed,'proxy',rule='head') for s in modes]
                ref=local(seed)
                if all(x is not None for x in v) and ref:
                    f,own,shuffle,mean,hard,frozen=v
                    cost=runs[(f'r{m}_fl',seed)]['summary'].get('cost_pass')
                    quality=f-ref['f1_macro']>=.05 and all(f-other>=.01 for other in (own,shuffle,mean,hard))
                    gates[f'M{m}_s{seed}']=dict(quality=bool(quality),cost=cost,vs_local=f-ref['f1_macro'],
                        vs_own=f-own,vs_shuffle=f-shuffle,vs_classmean=f-mean,vs_hard=f-hard,
                        refresh_contribution=f-frozen>=.005,vs_frozen=f-frozen)
        else:
            item=runs.get(('e_fl',seed))
            if item:
                probe=item['history'][0]['diagnostics'].get('sampler_probe') if item['history'] else None
                gates[f'energy_s{seed}']=dict(cost=item['summary'].get('cost_pass'),quality=None,
                    reason='cost stop before r20; quality unmeasured',cost_ratio=item['summary'].get('cost',{}).get('ratio'),
                    sampler_r1=probe)
                v=[score(n,seed,'proxy',rule='head') for n in ('e_fl','e_ce_raw','e_ce_aug','e_own','e_shuffle')]
                if all(x is not None for x in v):
                    f,ce,aug,own,shuffle=v
                    gates[f'energy_s{seed}']['quality']=f-max(ce,aug)>=-.01 and f-own>=.01 and f-shuffle>=.01
    (base/'gates.json').write_text(json.dumps(gates,indent=2)+'\n')
    for name,rows,extra in [('metrics-all.csv',metric_rows,METRIC_KEYS),('per-class-f1.csv',classes,['class_id','f1'])]:
        with (base/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['arm','seed','round','role','holdout','rule',*extra]);writer.writeheader();writer.writerows(rows)
    lines=[f'# {a.build.upper()} — audited artifacts','',f'{len(runs)} arm đãaudit; {sum(v["audit"]["complete"] for v in runs.values())} complete20round.',
        'Source/config/weights/BN/payload/pred→clientCM/10metric đãaudit; liveW&B/status khôngthayproof. StageU, chưaR1/R2/DP.','',
        '| arm | seed | rounds | complete | cost ratio | final group score |','|---|---:|---:|---|---:|---:|']
    for (arm,seed),run in sorted(runs.items()):
        role=run['result'].get('role','private');rule='head' if role=='proxy' else 'gp'
        value=score(arm,seed,role,rule=rule)
        lines.append(f'| {arm} | {seed} | {len(run["history"])} | {run["audit"]["complete"]} | {run["summary"].get("cost",{}).get("ratio","")} | {value if value is not None else "unmeasured r20"} |')
    lines+=['','## Locked gates','', '[Full-precision gates](gates.json). Chưađủpairedcontrols thìpending; costFAILkhônglàqualityFAIL.','',
        '```json',json.dumps(gates,indent=2),'```','','## Đủ10metric từng evaluation','',
        '[CSV](metrics-all.csv), [F1 theo lớp](per-class-f1.csv). Meanclientmetrics, khôngpooledconfusion.','',
        '| arm | seed | round | role | holdout | rule | '+' | '.join(METRIC_KEYS)+' |',
        '|---|---:|---:|---|---|---|'+'---:|'*10]
    for row in metric_rows:
        lines.append('| '+' | '.join([str(row[k]) for k in ('arm','seed','round','role','holdout','rule')]+[f'{row[k]:.6f}' for k in METRIC_KEYS])+' |')
    (base/'report.md').write_text('\n'.join(lines)+'\n')
    print(a.build,len(runs),'audited arms;',len(metric_rows),'ten-metric rows;',gates)


if __name__=='__main__':main()
