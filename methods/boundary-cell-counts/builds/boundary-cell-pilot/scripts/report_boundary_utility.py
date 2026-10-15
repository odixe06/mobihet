"""Fixed-r20 gates from audited stages; missing controls stay unknown, necessary failures stay failures."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS


def main():
    out=ROOT/'docs/methods/boundary-cell-counts/builds/boundary-cell-pilot/results'
    runs={};sources={};metrics=[];perclass=[]
    for folder in sorted((BASE/'artifacts/kaggle').glob('bc-u-s4?-*/')):
        marker=folder/'AUDITED.json'
        if not marker.exists():continue
        audit=json.loads(marker.read_text())
        for item in audit['audits']:
            run=folder/'runs'/item['run']
            rows=[json.loads(p.read_text()) for p in sorted((run/'complete').glob('round_*.json'))]
            assert len(rows)==item['rounds']
            if len(rows)<=len(runs.get(item['run'],[])):continue
            previous=runs.get(item['run'],[])
            assert [r['weights_sha'] for r in rows[:len(previous)]]==[r['weights_sha'] for r in previous]
            runs[item['run']]=rows;sources[item['run']]=audit['kernel']
    for name,rows in sorted(runs.items()):
        arm,seed=name.rsplit('_s',1)
        for row in rows:
            for role,holds in row['eval'].items():
                for hold,rules in holds.items():
                    for rule,ev in rules.items():
                        key=dict(seed=int(seed),arm=arm,round=row['round'],role=role,holdout=hold,rule=rule)
                        metrics.append(dict(key,**ev['mean_metrics']))
                        perclass.extend(dict(key,class_id=c,f1=f) for c,f in enumerate(ev['f1_per_class']))
    gates={};forensics={};lines=['# BoundaryCells — audited fixed-r20 utility','',
        'Primary private/group/gp target-mass đã khóa. Không dùng official test, best round hoặc đổi headline.',
        '', '| seed | candidate | rounds | group gp | delta strongest reference | gate |', '|---|---|---:|---:|---:|---|']
    for seed in (42,43):
        refs=[f'bc_{a}_s{seed}' for a in ('local','mean','gaussian')]
        if not all(len(runs.get(n,[]))==20 for n in refs):continue
        def ev(name):return runs[name][-1]['eval']['private']['group']['gp']
        strongest=max(refs,key=lambda n:ev(n)['f1_macro'])
        for mode in ('direct','teacher'):
            for p in (2,4):
                stem=f'bc_{mode}{p}';name=f'{stem}_fl_s{seed}';rows=runs.get(name,[])
                if not rows:continue
                value=ev(name);delta=value['f1_macro']-ev(strongest)['f1_macro']
                controls={scope:f'{stem}_{scope}_s{seed}' for scope in ('own','shuffle','marginal')}
                complete=len(rows)==20
                contrasts={s:value['f1_macro']-ev(n)['f1_macro'] if complete and len(runs.get(n,[]))==20 else None
                           for s,n in controls.items()}
                benign=bool(value['benign_f1']>=max(ev(refs[0])['benign_f1'],ev(strongest)['benign_f1'])-.005) if complete else None
                necessary=bool(delta>=.015 and benign) if complete else None
                full=complete and all(len(runs.get(n,[]))==20 for n in controls.values())
                passed=bool(necessary and contrasts['own']>=.01 and contrasts['shuffle']>=.01) if full else None
                decision=('FAIL necessary reference/benign gate; no extension needed' if complete and not necessary else
                          'PASS' if passed else 'FAIL' if passed is False else 'PENDING; controls/horizon incomplete')
                ratio={key:float(np.mean([r[key] for r in rows[2:]])/np.mean([r[key] for r in runs[refs[0]][2:]]))
                       for key in ('algorithm_seconds','sec')} if complete else None
                gates[f'{stem}_s{seed}']=dict(rounds=len(rows),strongest=strongest,vs_strongest=delta,
                    contrasts=contrasts,control_rounds={s:len(runs.get(n,[])) for s,n in controls.items()},
                    benign_pass=benign,necessary_gate_pass=necessary,full_r20_controls=full,utility_pass=passed,
                    cost_ratios_r3_20=ratio,decision=decision)
                forensics[name]=dict(rounds=len(rows),diagnostics=rows[-1]['diagnostics'],
                    last5_private_gp=[r['eval']['private']['group']['gp']['f1_macro'] for r in rows[-5:]],
                    private_head=value['f1_macro'] if 'head' not in rows[-1]['eval']['private']['group'] else rows[-1]['eval']['private']['group']['head']['f1_macro'],
                    proxy_head=rows[-1]['eval'].get('proxy',{}).get('group',{}).get('head',{}).get('f1_macro'),
                    last5_proxy_head=[r['eval'].get('proxy',{}).get('group',{}).get('head',{}).get('f1_macro') for r in rows[-5:]],
                    mean_private_real_ce=float(np.mean([v['real_ce'] for v in rows[-1]['clients']])),
                    mean_teacher_loss=float(np.mean([v['loss'] for v in rows[-1]['proxy_clients']])) if rows[-1]['proxy_clients'] else None,
                    limits='Saved learned teacher/private behavior; no isolation of static-table softness versus teacher fit/BN/extrapolation')
                lines.append(f'| {seed} | {stem} | {len(rows)}/20 | {value["f1_macro"]:.6f} | {delta:+.6f} | {decision} |')
    result=dict(sources=sources,gates=gates,scope='fixed r20; marginal diagnostic; own/shuffle>=.010; strongest>=.015; benign>=max(local,strongest)-.005',
        all_candidates_necessary_fail=len(gates)==8 and all(v['necessary_gate_pass'] is False for v in gates.values()))
    (out/'utility-gates.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'utility-failure-analysis.json').write_text(json.dumps(forensics,indent=2)+'\n')
    for name,rows,cols in [('utility-metrics-all.csv',metrics,METRIC_KEYS),('utility-per-class-f1.csv',perclass,['class_id','f1'])]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['seed','arm','round','role','holdout','rule',*cols]);w.writeheader();w.writerows(rows)
    lines+=['', '[10metrics mọi round](utility-metrics-all.csv) · [Per-class F1](utility-per-class-f1.csv) · [Gates/provenance](utility-gates.json).',
        '', 'Thiếu controls không được gọi full paired-r20 completion. Candidate đã r20 và trượt necessary strong-reference gate',
        'thì không cần huấn luyện nốt controls để xác nhận không đủ điều kiện tăng quy mô.',
        'Cost algorithm và full round so plain CE r3–20; không dummy-inflate denominator.',
        'Audit gồm source/state/payload/predictions→10metrics; strict reconstruction client0 mỗi round, không sampled saved-weight inference.',
        '[Teacher/private diagnostics](utility-failure-analysis.json) và direct-table forensic mô tả target/teacher bottleneck;',
        'teacher failure chưa được quy cho một nguyên nhân duy nhất; không mặc định thêm round sửa target tĩnh.']
    (out/'utility-report.md').write_text('\n'.join(lines)+'\n')
    print('Boundary utility report',len(metrics),'10metric rows;',len(gates),'candidate/seed gates')


if __name__=='__main__':main()
