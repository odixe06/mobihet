"""Locked V4 MALA/support paired gates and all ten metrics from AUDITED artifacts."""
import csv
import json
import sys
import torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS


def main():
    base=ROOT/'papers/v4'; runs={}; failed={}; rows=[]; classes=[]
    expected={v['run_id']:v for v in json.loads((base/'wandb-runs.json').read_text())}
    for marker in sorted((base/'kaggle').glob('*/AUDITED.json')):
        audit=json.loads(marker.read_text())
        for item in audit['audits']:
            arm=item['arm']; seed=audit['seed']; path=marker.parent/'runs'/f'{arm}_s{seed}'
            cfg=json.loads((path/'config.json').read_text()); fp=cfg['fingerprint']
            assert expected[f'{arm}_s{seed}-{fp[:8]}']['fingerprint']==fp
            if not item['complete']: failed[(arm,seed)]=item
            if (path/'result.json').exists():
                result=json.loads((path/'result.json').read_text()); assert result['fingerprint']==fp
                runs[(arm,seed)]=result
            elif item['rounds']:
                history=[json.loads(p.read_text()) for p in sorted((path/'complete').glob('round_*.json'))]
                state=torch.load(path/f'resume/round_{item["rounds"]:03d}.pt',map_location='cpu',weights_only=True)
                runs[(arm,seed)]=dict(history=history,bytes=state['bytes'],complete=False,
                    minutes=sum(h['sec'] for h in history)/60,fingerprint=fp)
    local={}
    for seed in (42,43):
        audit=json.loads((ROOT/f'papers/v2/kaggle/v2a-s{seed}-20261011-v1/AUDITED.json').read_text())
        assert next(v for v in audit['audits'] if v['arm']=='local')['complete']
        local[seed]=json.loads((ROOT/f'papers/v2/runs/local_s{seed}/result.json').read_text())
    def ev(arm,seed,hold='group'):
        value=local[seed] if arm=='local' else runs.get((arm,seed))
        return value['history'][-1]['eval']['private'][hold]['gp'] if value and value['complete'] else None
    gates={}
    definitions={'M':('c_mala_fl',['c_fl','c_mala_own','c_mala_shuffle','diag'],[.015,.01,.01,.015]),
                 'S':('s_structural',['s_diag','s_own','s_shuffle'],[.015,.01,.01])}
    for seed in (42,43):
        for label,(arm,controls,thresholds) in definitions.items():
            failure=failed.get((arm,seed))
            if failure:
                scientific=failure.get('failure_category')=='numerical' or failure['rounds']>0
                gates[f'{label}_s{seed}']=dict(pass_quality=False if scientific else None,
                    reason='failed/partial locked configuration' if scientific else 'startup failure needs diagnosis',
                    rounds=failure['rounds']); continue
            if any(ev(a,seed) is None for a in [arm,*controls]): continue
            value=ev(arm,seed); baseline=ev('local',seed)
            contrasts={c:value['f1_macro']-ev(c,seed)['f1_macro'] for c in controls}
            common=value['f1_macro']-baseline['f1_macro']>=.03 and ev(arm,seed,'old')['f1_macro']>ev('local',seed,'old')['f1_macro'] and value['benign_f1']>=baseline['benign_f1']-.03
            passed=common and all(contrasts[c]>=threshold for c,threshold in zip(controls,thresholds))
            gates[f'{label}_s{seed}']=dict(pass_quality=bool(passed),common_pass=bool(common),contrasts=contrasts)
            if label=='S':
                marginal=ev('s_marginal',seed)
                contrast=value['f1_macro']-marginal['f1_macro'] if marginal else None
                gates[f'{label}_s{seed}'].update(first_session_contrast=contrast,
                    first_session_contribution=contrast>=.005 if contrast is not None else None)
    paired={label:(all(gates[f'{label}_s{s}']['pass_quality'] for s in (42,43))
        if all(gates.get(f'{label}_s{s}',{}).get('pass_quality') is not None for s in (42,43)) else None) for label in definitions}
    (base/'gates.json').write_text(json.dumps(dict(per_seed=gates,paired=paired),indent=2))
    for (arm,seed),result in sorted(runs.items()):
        for h in result['history']:
            for role,holds in h.get('eval',{}).items():
                for hold,rules in holds.items():
                    for rule,value in rules.items():
                        rows.append(dict(arm=arm,seed=seed,round=h['round'],role=role,holdout=hold,rule=rule,**value['mean_metrics']))
                        classes.extend(dict(arm=arm,seed=seed,round=h['round'],role=role,holdout=hold,rule=rule,class_id=c,f1=v)
                                       for c,v in enumerate(value['f1_per_class']))
    fields=['arm','seed','round','role','holdout','rule']
    for name,data,extra in [('metrics-all.csv',rows,list(METRIC_KEYS)),('per-class-f1.csv',classes,['class_id','f1'])]:
        with (base/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields+extra); w.writeheader(); w.writerows(data)
    lines=['# V4 — artifact report (đang thu nếu chưa đủ20run)','',
        '[MALA contract](../../docs/research/pilot-v4-mala.md), [support contract](../../docs/research/pilot-v4-support.md), [ledger](rebuild.md).',
        'Chỉ terminal AUDITED artifacts; W&B/queue/status không là F1 đãverify. U stage, chưa R1/R2 thực nghiệm.','',
        '| hướng | paired quality |','|---|---|']
    lines.extend(f'| {k} | {"PENDING" if v is None else "PASS" if v else "FAIL"} |' for k,v in paired.items())
    lines+=['','## Headline r20','', '| arm | seed | group/private/gp macro-F1 | benign | minutes | up B | down B |','|---|---:|---:|---:|---:|---:|---:|']
    for (arm,seed),run in sorted(runs.items()):
        if not run['complete']: continue
        value=ev(arm,seed); up=sum(v[-1] for v in run['bytes'] if v[2]=='up'); down=sum(v[-1] for v in run['bytes'] if v[2]=='down')
        lines.append(f'| {arm} | {seed} | {value["f1_macro"]:.6f} | {value["benign_f1"]:.6f} | {run["minutes"]:.2f} | {up} | {down} |')
    lines+=['','## Failed/partial','', '| arm | seed | committed rounds | category |','|---|---:|---:|---|']
    lines.extend(f'| {arm} | {seed} | {v["rounds"]} | {v.get("failure_category","unknown")} |' for (arm,seed),v in sorted(failed.items()))
    lines+=['','## Đủ10metric mọi evaluation','',
        '[Full precision CSV](metrics-all.csv), [per-class F1](per-class-f1.csv), [gates](gates.json). Mean metric từ từngclientCM.','',
        '| '+' | '.join(fields+list(METRIC_KEYS))+' |','|---|---:|---:|---|---|---|'+'---:|'*10]
    lines.extend('| '+' | '.join([str(row[k]) for k in fields]+[f'{row[k]:.6f}' for k in METRIC_KEYS])+' |' for row in rows)
    lines+=['','MALA acceptance/ESJD và supportpacket/table invariants audit riêng, không dùng fidelity làm F1proof. '
        'OwnMALA giữcommonGaussian bootstrap, ownsupport chỉcommonprior; không đồng nhất hai owncontrol. '
        'Localreference V2A đãverify cùngprivate recipe/fixture/runtime; freshdiag/weightedGaussian ởV4. '
        'Internal holdouts/scaler/simulator cógiới hạn kế thừa; chưacrypto/DP, hai seedkhôngCI. '
        'No contact/fullscale khiqualityFAIL. Git/DVC chưabackup.']
    (base/'report.md').write_text('\n'.join(lines)+'\n')
    print('V4 report:',len(runs),'runs;',len(rows),'ten-metric rows;',paired)


if __name__=='__main__': main()
