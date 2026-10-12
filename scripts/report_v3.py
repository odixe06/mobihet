"""V3 locked paired gates and ten metrics, exclusively from terminal audited artifacts."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS
from mh.v3_math import FAMILIES


def main():
    base=ROOT/'papers/v3'; runs={}; failures={}; rows=[]; classes=[]
    registry={v['run_id']:v for v in json.loads((base/'wandb-runs.json').read_text())}
    for path in sorted((base/'kaggle').glob('*/AUDITED.json')):
        audit=json.loads(path.read_text())
        for item in audit['audits']:
            arm=item['arm']; seed=audit['seed']; run=path.parent/'runs'/f'{arm}_s{seed}'
            cfg=json.loads((run/'config.json').read_text()); fp=cfg['fingerprint']
            expected=registry[f'{arm}_s{seed}-{fp[:8]}']; assert fp==expected['fingerprint']
            if not item['complete']: failures[(arm,seed)]=item
            if not item['rounds']: continue
            history=[json.loads(p.read_text()) for p in sorted((run/'complete').glob('round_*.json'))]
            assert len(history)==item['rounds']
            result=json.loads((run/'result.json').read_text()) if (run/'result.json').exists() else {}
            runs[(arm,seed)]=dict(history=history,complete=item['complete'],
                bytes=result.get('bytes',[]),minutes=result.get('minutes',sum(h['sec'] for h in history)/60),
                fingerprint=fp,source=str(run.relative_to(ROOT)))
    references={}
    for seed in (42,43):
        audit=json.loads((ROOT/f'papers/v2/kaggle/v2a-s{seed}-20261011-v1/AUDITED.json').read_text())
        for arm in ('local','a_own'):
            verified=next(v for v in audit['audits'] if v['arm']==arm); assert verified['complete']
            run=ROOT/f'papers/v2/runs/{arm}_s{seed}'; result=json.loads((run/'result.json').read_text())
            cfg=json.loads((run/'config.json').read_text()); assert result['fingerprint']==cfg['fingerprint']
            assert result['complete'] and len(result['history'])==20
            references[(arm,seed)]=result
    def ev(arm,seed,role='private',hold='group',rule='gp'):
        run=runs.get((arm,seed)) or references.get((arm,seed))
        return run['history'][-1]['eval'][role][hold][rule] if run and run['complete'] else None
    candidates=[(f'a_{f}_{size}_{suffix}',f'a_{f}_{size}_{own}',f'{f.upper()}_{size}')
        for f,suffix,own in [('r','ef','own_ef'),('t','train','own_train'),('o','ef','own_ef')]
        for size in ('small','medium')]+[('a_p_medium_ef','a_p_medium_own_ef','P_medium')]
    gates={}
    for seed in (42,43):
        local=ev('local',seed); own=ev('a_own',seed); dense=ev('a_dense32',seed)
        def common(arm):
            v=ev(arm,seed); old=ev(arm,seed,hold='old'); lo=ev('local',seed,hold='old')
            return v['f1_macro']-local['f1_macro']>=.03 and old['f1_macro']>lo['f1_macro'] and v['benign_f1']>=local['benign_f1']-.03
        for arm,control,label in candidates:
            failure=failures.get((arm,seed))
            if failure:
                scientific=failure.get('failure_category')=='numerical' or failure['rounds']>0
                gates[f'{label}_s{seed}']=dict(pass_quality=False if scientific else None,
                    reason='failed/partial under locked configuration' if scientific else 'startup failure needs diagnosis',
                    rounds=failure['rounds']); continue
            value=ev(arm,seed); codec=ev(control,seed)
            if value is None or codec is None or dense is None: continue
            dg=dense['f1_macro']-own['f1_macro']; gain=value['f1_macro']-own['f1_macro']
            teacher=ev('a_dense32',seed,'proxy',rule='head')['f1_macro']-local['f1_macro']>=.05
            retain=gain/dg if dg>0 else None
            passed=teacher and common('a_dense32') and common(arm) and dg>0 and gain>=.01 and retain>=.8 and value['f1_macro']-codec['f1_macro']>=.01
            component='a_p_medium_noef' if label.startswith('P') else arm.replace('_ef','_noef').replace('_train','_comm')
            ablation=ev(component,seed); contrast=value['f1_macro']-ablation['f1_macro'] if ablation else None
            item=dict(pass_quality=bool(passed),dense_teacher_pass=bool(teacher),gain_vs_own=gain,
                retention=retain,gain_vs_codec=value['f1_macro']-codec['f1_macro'],
                component_contrast=contrast,component_contribution=contrast>=.005 if contrast is not None else None)
            if label.startswith('P'):
                random=ev('a_o_rank8_ef',seed); adaptive=value['f1_macro']-random['f1_macro'] if random else None
                item.update(adaptive_basis_contrast=adaptive,adaptive_contribution=adaptive>=.005 if adaptive is not None else None)
            gates[f'{label}_s{seed}']=item
    paired={label:(all(gates[f'{label}_s{s}']['pass_quality'] for s in (42,43))
        if all(gates.get(f'{label}_s{s}',{}).get('pass_quality') is not None for s in (42,43)) else None)
        for _,_,label in candidates}
    (base/'gates.json').write_text(json.dumps(dict(per_seed=gates,paired=paired),indent=2))
    for (arm,seed),run in sorted(runs.items()):
        for h in run['history']:
            for role,holds in h.get('eval',{}).items():
                for hold,rules in holds.items():
                    for rule,evv in rules.items():
                        rows.append(dict(arm=arm,seed=seed,round=h['round'],role=role,holdout=hold,rule=rule,**evv['mean_metrics']))
                        classes.extend(dict(arm=arm,seed=seed,round=h['round'],role=role,holdout=hold,rule=rule,
                            class_id=c,f1=v) for c,v in enumerate(evv['f1_per_class']))
    for name,data,fields in [('metrics-all.csv',rows,[*METRIC_KEYS]),('per-class-f1.csv',classes,['class_id','f1'])]:
        with (base/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['arm','seed','round','role','holdout','rule',*fields]); writer.writeheader(); writer.writerows(data)
    lines=['# V3 — artifact report (đang thu nếu chưa đủ48run)','',
        '[Manifest khóa](../../docs/research/pilot-v3.md), [ledger](rebuild.md), [gates](gates.json).',
        'Chỉ AUDITED.json mới vào report; W&B/status không thay bằng chứng weights/prediction.',
        f'{len(runs)} run có checkpoint đãaudit; {len(failures)} arm failed/partial. StageU, chưa R1/R2 thực nghiệm.','',
        '| hướng | paired quality |','|---|---|']
    lines.extend(f'| {label} | {"PENDING" if v is None else "PASS" if v else "FAIL"} |' for label,v in paired.items())
    lines+=['','## Reference provenance','',
        'Local/a_own: V2A cùng recipe/fixture/runtime đãaudit. Dense32: mới ởV3R, cùngfp32 vớicarrier; '
        'densefp16A không làretention denominator. Hai seed không CI; không chọnhead/bestround để cứu cổng.','',
        '| arm | seed | fingerprint | source |','|---|---:|---|---|']
    for (arm,seed),run in sorted(references.items()):
        lines.append(f'| {arm} | {seed} | {run["fingerprint"]} | ../v2/runs/{arm}_s{seed} |')
    lines+=['','## Headline đủr20','', '| arm | seed | private group/gp F1 | benign | minutes | up B | down B |','|---|---:|---:|---:|---:|---:|---:|']
    for (arm,seed),run in sorted(runs.items()):
        if not run['complete']: continue
        value=ev(arm,seed); up=sum(v[-1] for v in run['bytes'] if v[2]=='up'); down=sum(v[-1] for v in run['bytes'] if v[2]=='down')
        lines.append(f'| {arm} | {seed} | {value["f1_macro"]:.6f} | {value["benign_f1"]:.6f} | {run["minutes"]:.2f} | {up} | {down} |')
    lines+=['','## Failed/partial','', '| arm | seed | committed rounds | category |','|---|---:|---:|---|']
    lines.extend(f'| {arm} | {seed} | {v["rounds"]} | {v.get("failure_category","unknown")} |' for (arm,seed),v in sorted(failures.items()))
    lines+=['','## Đủ10metric mọi evaluation','',
        '[CSV full precision](metrics-all.csv), [per-class](per-class-f1.csv). Mean metric từ client confusion riêng; không pooledCM.', '',
        '| arm | seed | round | role | holdout | rule | '+' | '.join(METRIC_KEYS)+' |','|---|---:|---:|---|---|---|'+'---:|'*10]
    lines.extend('| '+' | '.join([str(row[k]) for k in ('arm','seed','round','role','holdout','rule')]+[f'{row[k]:.6f}' for k in METRIC_KEYS])+' |' for row in rows)
    lines+=['','Internal holdout từtrain80%, scaler có giới hạn leakage kế thừa; trusted sum simulation chưacrypto/DP. '
        'Byte application chưa ACK/frame/deadline/bootstrap. Sampled decisive inference không re-infer mọiclient. '
        'Không tune/kéo dài cổngtrượt; chỉ quaU mớicontactstage.']
    (base/'report.md').write_text('\n'.join(lines)+'\n')
    print('V3 artifact report:',len(runs),'runs;',len(rows),'ten-metric rows;',paired)


if __name__=='__main__': main()
