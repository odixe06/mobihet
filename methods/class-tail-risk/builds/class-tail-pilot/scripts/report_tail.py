"""Report audited ClassTail rounds only; evaluate fixed-r20 gates when both seeds exist."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS


def main():
    out=ROOT/'docs/methods/class-tail-risk/builds/class-tail-pilot/results';out.mkdir(exist_ok=True)
    runs={};metrics=[];perclass=[];gates={};sources={};forensics={};table=[]
    for seed in (42,43):
        candidates=[BASE/'artifacts/kaggle'/f'ct-{stage}-s{seed}-20261015-v1' for stage in ('u','cal')]
        parent=next((p for p in candidates if (p/'AUDITED.json').exists()),None)
        if parent is None:continue
        proof=json.loads((parent/'AUDITED.json').read_text());sources[str(seed)]=proof['kernel']
        for audit in proof['audits']:
            arm=audit['run'].rsplit('_s',1)[0]
            rows=[json.loads(p.read_text()) for p in sorted((parent/'runs'/audit['run']/'complete').glob('round_*.json'))]
            runs[(arm,seed)]=rows
            end=rows[-1]['eval']['private']['group']
            table.append((seed,arm,len(rows),end['gp']['f1_macro'],end['head']['f1_macro']))
            for row in rows:
                for role,holds in row['eval'].items():
                    for hold,rules in holds.items():
                        for rule,ev in rules.items():
                            keys=dict(seed=seed,arm=arm,round=row['round'],role=role,holdout=hold,rule=rule)
                            metrics.append(dict(keys,**ev['mean_metrics']))
                            perclass.extend(dict(keys,class_id=c,f1=f) for c,f in enumerate(ev['f1_per_class']))
        if not proof.get('utility_r20'):continue
        def ev(arm):return runs[(arm,seed)][-1]['eval']['private']['group']['gp']
        value=ev('ct_fl');f=value['f1_macro'];local=ev('ct_local')
        strongest=max(('ct_gaussian','ct_dml'),key=lambda arm:ev(arm)['f1_macro'])
        contrasts={name:f-ev('ct_'+name)['f1_macro'] for name in ('local','own','shuffle','classmass')}
        benign=value['benign_f1']>=max(local['benign_f1'],ev(strongest)['benign_f1'])-.005
        ratio={key:float(np.mean([r[key] for r in runs[('ct_fl',seed)][2:]])/
                         np.mean([r[key] for r in runs[('ct_local',seed)][2:]]))
               for key in ('algorithm_seconds','sec')}
        utility=contrasts['local']>=.015 and contrasts['own']>=.01 and contrasts['shuffle']>=.01 and benign
        strong=f-ev(strongest)['f1_macro'];cost=all(v<=2 for v in ratio.values())
        gates[str(seed)]=dict(utility_pass=bool(utility),contrasts=contrasts,benign_pass=bool(benign),
            strongest=strongest,vs_strongest=strong,cost_ratios_postwarmup=ratio,cost_pass=bool(cost),
            scale_eligible=bool(utility and strong>=.015 and cost))
        last=runs[('ct_fl',seed)][-1];diag=last['diagnostics']
        forensics[str(seed)]=dict(last5_group_gp=[r['eval']['private']['group']['gp']['f1_macro'] for r in runs[('ct_fl',seed)][-5:]],
            head_vs_local=last['eval']['private']['group']['head']['f1_macro']-runs[('ct_local',seed)][-1]['eval']['private']['group']['head']['f1_macro'],
            terminal_tail_fraction=diag['tail_fraction'],terminal_membership_difference=diag['macro_membership_difference'],
            terminal_classes_in_tail_range=diag['classes_in_tail_range'],threshold_at_bin_floor=int(np.sum(np.asarray(diag['threshold'])==2**-12)),
            weighted_tail_fraction=float(np.dot(diag['tail_fraction'],diag['counts'])/sum(diag['counts'])),
            perclass_fl_minus_classmass=(np.asarray(value['f1_per_class'])-ev('ct_classmass')['f1_per_class']).tolist(),
            limits='Saved terminal scan diagnostics and aligned r20 controls; no causal isolation of quantization/calibration/hard-example selection')
    for name,rows,columns in [('metrics-all.csv',metrics,[*METRIC_KEYS]),('per-class-f1.csv',perclass,['class_id','f1'])]:
        with (out/name).open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['seed','arm','round','role','holdout','rule',*columns])
            writer.writeheader();writer.writerows(rows)
    paired=len(gates)==2
    classmass_equivalent=paired and all(abs(v['contrasts']['classmass'])<=.005 for v in gates.values())
    result=dict(sources=sources,gates=gates,paired_r20=paired,
        paired_utility_pass=all(v['utility_pass'] for v in gates.values()) if paired else None,
        paired_scale_eligible=all(v['scale_eligible'] for v in gates.values()) if paired else None,
        classmass_mechanism_unsupported=bool(classmass_equivalent) if paired else None,
        scope='fixed r20; primary group/private/gp; postwarmup cost r3-20; no automatic launch or official test')
    (out/'gates.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'failure-analysis.json').write_text(json.dumps(forensics,indent=2)+'\n')
    probe=json.loads((BASE/'probes/paired-mechanism-gate.json').read_text())
    lines=['# ClassTail — audited results','',
        'Paired r2 mechanism PASS; calibration không chứng minh utility. R20 chỉ chốt từ đủ hai seed đã audit.',
        '', '| seed | classes tail10–40% | pooled/own membership diff |', '|---|---:|---:|']
    for seed,item in probe['seeds'].items():
        lines.append(f'| {seed} | {item["classes_in_tail_range"]}/16 | {item["macro_membership_difference"]:.6f} |')
    lines+=['',f'Audited evaluation rows đủ10metrics: {len(metrics)}; paired r20: {paired}.',
        '[10metrics mọi round](metrics-all.csv) · [Per-class F1](per-class-f1.csv) · [Locked gates](gates.json).',
        '', '| seed | arm | rounds | group gp primary | group head diagnostic |', '|---|---|---:|---:|---:|']
    lines.extend(f'| {s} | {a} | {n} | {gp:.6f} | {head:.6f} |' for s,a,n,gp,head in table)
    lines+=['',f'Paired utility PASS: {result["paired_utility_pass"]}; scale eligible: {result["paired_scale_eligible"]}.',
        f'Pooled/class-mass difference≤0,005 cả hai seed: {result["classmass_mechanism_unsupported"]}.',
        '[Failure diagnostics](failure-analysis.json): late thresholds ở bin floor và tail nhỏ mô tả lượng tử hóa;',
        'không cô lập nguyên nhân duy nhất. Pooled không hơn own/shuffle/class-mass thì chưa chứng minh tri thức hard-example peer.',
        'Không kéo dài hoặc đổi alpha/lambda/bin sau FAIL; không suy cả họ tail-risk không thể hữu ích.',
        '', 'Benign guard so local và reference Gaussian/DML mạnh nhất; không chọn best round.',
        'Cost báo cả algorithm và wall round r3–20; baseline CE không scan ở các round đó.',
        'Class-mass CE ngang pooled trong≤0,005 cả hai seed thì chưa quy lợi cho hard-example mechanism.',
        'R1/contact/DP chưa kiểm. Source/fixture/state/predictions đã audit; không sampled saved-weight inference.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('ClassTail report',len(metrics),'10metric rows; paired r20',paired)


if __name__=='__main__':main()
