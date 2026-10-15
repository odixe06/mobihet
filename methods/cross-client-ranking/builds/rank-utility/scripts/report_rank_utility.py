"""Fixed-r20 paired utility, all ten metrics and inherited-round provenance from terminal audited outputs only."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS


def main():
    out=ROOT/'docs/methods/cross-client-ranking/builds/rank-utility/results';out.mkdir(exist_ok=True)
    metrics=[];perclass=[];sources={};gates={};table=[]
    for seed in (42,43):
        candidates=sorted((BASE/'artifacts/kaggle').glob(f'cr-u-s{seed}-20261015-v*/AUDITED.json'))
        if not candidates:continue
        marker=candidates[-1];parent=marker.parent
        audit=json.loads(marker.read_text());sources[str(seed)]=audit['kernel'];runs={}
        for arm in audit['audits']:
            name=arm['run'].rsplit('_s',1)[0]
            rows=[json.loads(p.read_text()) for p in sorted((parent/'runs'/arm['run']/'complete').glob('round_*.json'))];runs[name]=rows
            for row in rows:
                for role,holds in row['eval'].items():
                    for hold,rules in holds.items():
                        for rule,ev in rules.items():
                            key=dict(seed=seed,arm=name,round=row['round'],role=role,holdout=hold,rule=rule)
                            metrics.append(dict(key,**ev['mean_metrics']))
                            perclass.extend(dict(key,class_id=c,f1=f) for c,f in enumerate(ev['f1_per_class']))
            last=rows[-1];table.append((seed,name,last['round'],last['eval']['private']['group']['gp']['f1_macro']))
        if not audit['utility_r20']:continue
        def ev(arm):return runs[arm][-1]['eval']['private']['group']['gp']
        value=ev('cr_fl');strongest=max(('cr_gaussian','cr_dml'),key=lambda a:ev(a)['f1_macro'])
        contrast={a:value['f1_macro']-ev('cr_'+a)['f1_macro'] for a in ('local','own','shuffle','fixed','classweight')}
        benign=value['benign_f1']>=max(ev('cr_local')['benign_f1'],ev(strongest)['benign_f1'])-.005
        cost={k:float(np.mean([r[k] for r in runs['cr_fl']])/np.mean([r[k] for r in runs['cr_local']])) for k in ('algorithm_seconds','sec')}
        passed=contrast['local']>=.015 and contrast['own']>=.01 and contrast['shuffle']>=.01 and benign
        strong=value['f1_macro']-ev(strongest)['f1_macro']
        gates[str(seed)]=dict(utility_pass=bool(passed),contrasts=contrast,benign_pass=bool(benign),strongest=strongest,
            vs_strongest=strong,cost_ratios_r3_20=cost,cost_pass=all(v<=2 for v in cost.values()),
            scale_eligible=bool(passed and strong>=.015 and all(v<=2 for v in cost.values())))
    for name,rows,extra in [('metrics-all.csv',metrics,METRIC_KEYS),('per-class-f1.csv',perclass,['class_id','f1'])]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['seed','arm','round','role','holdout','rule',*extra]);w.writeheader();w.writerows(rows)
    paired=len(gates)==2
    result=dict(sources=sources,gates=gates,paired_r20=paired,paired_pass=all(v['utility_pass'] for v in gates.values()) if paired else None,
        scale_eligible=all(v['scale_eligible'] for v in gates.values()) if paired else None,
        classweight_mechanism_unsupported=all(abs(v['contrasts']['classweight'])<=.005 for v in gates.values()) if paired else None,
        scope='Fixedr20 true-class-prior gp; inherited CE-r1/r2 retain original lineage, no new gp metrics invented; no test-driven selection')
    (out/'gates.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# CrossClientRank — terminal audited utility','',f'Pairedr20: {paired}; paired utility PASS: {result["paired_pass"]}; scale eligible: {result["scale_eligible"]}.',
        '', '| seed | arm | logical round | group gp |','|---|---|---:|---:|']
    lines.extend(f'| {s} | {a} | {r} | {v:.6f} |' for s,a,r,v in table)
    lines+=['','[10metrics](metrics-all.csv) · [Per-class](per-class-f1.csv) · [Gates/provenance](gates.json).',
        'Sáu forks chỉ báo newly measuredr3–20; r1/r2 gốc không được gọi gp mới. References báo actualr1–20.',
        'R3 cost preflight không thay pairedr20 utility hoặc post-warm cost. Không đổi headline/HP/round sau FAIL.',
        'NNLS control clip[0,4], không exact-gradient-matched; probe không là calibration-equivalence proof.',
        'Audit nguồn/state/BN/wire/predictions→10metrics; không independent saved-weight inference hoặc NNLS refit.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print('Rank utility report',len(metrics),'10metric rows; pairedr20',paired)


if __name__=='__main__':main()
