"""V5 terminal audited artifacts, locked teacher/private/cost gates and ten metrics."""
import csv
import json
import sys
from pathlib import Path
import torch
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1]); sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS


def main():
    base=ROOT/'papers/v5'; runs={}; failures={}; audited={}; rows=[]; classes=[]
    expected={v['run_id']:v for v in json.loads((base/'wandb-runs.json').read_text())}
    for marker in sorted((base/'kaggle').glob('*/AUDITED.json')):
        audit=json.loads(marker.read_text())
        for item in audit['audits']:
            arm=item['arm']; seed=audit['seed']; path=marker.parent/'runs'/f'{arm}_s{seed}'
            identity=json.loads((path/'config.json').read_text()); fp=identity['fingerprint']
            assert expected[f'{arm}_s{seed}-{fp[:8]}']['fingerprint']==fp
            audited[(arm,seed)]=item
            if not item['complete']: failures[(arm,seed)]=item
            if (path/'result.json').exists(): runs[(arm,seed)]=json.loads((path/'result.json').read_text())
            elif item['rounds']:
                history=[json.loads(p.read_text()) for p in sorted((path/'complete').glob('round_*.json'))]
                state=torch.load(path/f'resume/round_{item["rounds"]:03d}.pt',map_location='cpu',weights_only=True)
                runs[(arm,seed)]=dict(history=history,bytes=state['bytes'],complete=False,
                    role='proxy' if arm.startswith('b_') else 'private',minutes=sum(h['sec'] for h in history)/60)
    local={}
    for seed in (42,43):
        proof=json.loads((ROOT/f'papers/v2/kaggle/v2a-s{seed}-20261011-v1/AUDITED.json').read_text())
        assert next(v for v in proof['audits'] if v['arm']=='local')['complete']
        local[seed]=json.loads((ROOT/f'papers/v2/runs/local_s{seed}/result.json').read_text())
    def ev(arm,seed,hold='group'):
        result=local[seed] if arm=='local' else runs.get((arm,seed))
        if not result or not result['complete']: return None
        role=result['role']; rule='head' if role=='proxy' else 'gp'
        return result['history'][-1]['eval'][role][hold][rule]
    gates={}
    definitions={f'H_{size}':(f'b_anchor_{size}_fl',[f'b_anchor_{size}_{m}' for m in ('gauss','own','shuffle')]) for size in ('small','medium')}
    definitions.update({f'G_{ipc}':(f'g{ipc}_grad_trained',[f'g{ipc}_{m}' for m in ('diag','noise','own')]) for ipc in (1,4)})
    for seed in (42,43):
        for label,(arm,controls) in definitions.items():
            failure=failures.get((arm,seed))
            if failure:
                scientific=failure.get('failure_category') in ('numerical','cost','privacy','coverage') or failure['rounds']>0
                gates[f'{label}_s{seed}']=dict(pass_quality=False if scientific else None,reason=failure.get('failure_category','unknown'),rounds=failure['rounds']); continue
            if any(ev(a,seed) is None for a in [arm,*controls]): continue
            value=ev(arm,seed); baseline=ev('local',seed)
            contrast={a:value['f1_macro']-ev(a,seed)['f1_macro'] for a in controls}
            if label.startswith('H'):
                passed=value['f1_macro']-baseline['f1_macro']>=.05 and all(v>=.01 for v in contrast.values())
                item=dict(pass_quality=bool(passed),gain_vs_local=value['f1_macro']-baseline['f1_macro'],contrasts=contrast,role='teacher/head')
            else:
                common=value['f1_macro']-baseline['f1_macro']>=.03 and ev(arm,seed,'old')['f1_macro']>ev('local',seed,'old')['f1_macro'] and value['benign_f1']>=baseline['benign_f1']-.03
                cost=audited[(arm,seed)].get('cost_pass')
                passed=common and cost is True and contrast[controls[0]]>=.015 and contrast[controls[1]]>=.015 and contrast[controls[2]]>=.01
                ipc=label[-1]; mean=ev(f'g{ipc}_mean_trained',seed); frozen=ev(f'g{ipc}_grad_frozen',seed); mf=ev(f'g{ipc}_mean_frozen',seed)
                objective=value['f1_macro']-mean['f1_macro'] if mean else None
                encoder=value['f1_macro']-frozen['f1_macro'] if frozen else None
                interaction=value['f1_macro']-mean['f1_macro']-frozen['f1_macro']+mf['f1_macro'] if all(v is not None for v in (mean,frozen,mf)) else None
                item=dict(pass_quality=bool(passed),common_pass=bool(common),cost_pass=cost,contrasts=contrast,
                    objective_contrast=objective,objective_contribution=objective>=.005 if objective is not None else None,
                    encoder_contrast=encoder,encoder_contribution=encoder>=.005 if encoder is not None else None,interaction=interaction,role='private/gp')
            gates[f'{label}_s{seed}']=item
    paired={label:(all(gates[f'{label}_s{s}']['pass_quality'] for s in (42,43))
        if all(gates.get(f'{label}_s{s}',{}).get('pass_quality') is not None for s in (42,43)) else None) for label in definitions}
    (base/'gates.json').write_text(json.dumps(dict(per_seed=gates,paired=paired),indent=2))
    for (arm,seed),run in sorted(runs.items()):
        for h in run['history']:
            for role,holds in h.get('eval',{}).items():
                for hold,rules in holds.items():
                    for rule,value in rules.items():
                        rows.append(dict(arm=arm,seed=seed,round=h['round'],role=role,holdout=hold,rule=rule,**value['mean_metrics']))
                        classes.extend(dict(arm=arm,seed=seed,round=h['round'],role=role,holdout=hold,rule=rule,class_id=c,f1=v) for c,v in enumerate(value['f1_per_class']))
    fields=['arm','seed','round','role','holdout','rule']
    for name,data,extra in [('metrics-all.csv',rows,list(METRIC_KEYS)),('per-class-f1.csv',classes,['class_id','f1'])]:
        with (base/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields+extra); writer.writeheader(); writer.writerows(data)
    lines=['# V5 — artifact report (đangthu nếu chưa đủ46run)','',
        '[Hmanifest](../../docs/research/pilot-v5-anchors.md), [G/Jmanifest](../../docs/research/pilot-v5-books.md), [ledger](rebuild.md).',
        'ChỉAUDITED; Hteacher/head khôngprivateIDS. G/Jprivate/gp cầncost2×baseline. NoR1/R2/crypto/DPproof.','',
        '| hướng | paired quality |','|---|---|']
    lines.extend(f'| {label} | {"PENDING" if v is None else "PASS" if v else "FAIL"} |' for label,v in paired.items())
    lines+=['','## Headline đủr20','', '| arm | seed | role/rule | group macro-F1 | benignF1 | active minutes | up B | down B | costpass |','|---|---:|---|---:|---:|---:|---:|---:|---|']
    for (arm,seed),run in sorted(runs.items()):
        if not run['complete']: continue
        value=ev(arm,seed); role=run['role']; rule='head' if role=='proxy' else 'gp'
        up=sum(v[-1] for v in run['bytes'] if v[2]=='up'); down=sum(v[-1] for v in run['bytes'] if v[2]=='down')
        lines.append(f'| {arm} | {seed} | {role}/{rule} | {value["f1_macro"]:.6f} | {value["benign_f1"]:.6f} | {run["minutes"]:.2f} | {up} | {down} | {audited[(arm,seed)].get("cost_pass")} |')
    lines+=['','## Failed/partial/unstarted','', '| arm | seed | committed rounds | category |','|---|---:|---:|---|']
    lines.extend(f'| {arm} | {seed} | {v["rounds"]} | {v.get("failure_category","unknown")} |' for (arm,seed),v in sorted(failures.items()))
    lines+=['','## Đủ10metric mọi evaluation','', '[Full precision](metrics-all.csv), [per-class](per-class-f1.csv), [gates](gates.json). Mean metric từtừngclientconfusion.','',
        '| '+' | '.join(fields+list(METRIC_KEYS))+' |','|---|---:|---:|---|---|---|'+'---:|'*10]
    lines.extend('| '+' | '.join([str(v[k]) for k in fields]+[f'{v[k]:.6f}' for k in METRIC_KEYS])+' |' for v in rows)
    lines+=['','Source/weights/payload/peer-map/encoderancestry/optimizerrows/pred→10metric đãaudit, sampleddecisive inference khôngmọiclient/fullholdout. '
        'H Gaussian/sharedanchor/coveragebootstrap cũng ởowncontrol; G/Jown giữcommonGaussian/prior. '
        'LocalV2reference đãverify, freshG/Jdiag khácarmIDs dù cùngrecipe. Internalholdout/scaler/simulator cógiới hạn kếthừa. '
        'Two-seed khôngCI, khôngchọnhead/bestround/hp đểcứucổng. Costpreflight/model0round khôngmodelqualityproof; '
        'coststop khôngOOM vàkhôngrerunđểvượtcấp2×.']
    (base/'report.md').write_text('\n'.join(lines)+'\n')
    print('V5 report',len(runs),'runs',len(rows),'ten-metric rows',paired)


if __name__=='__main__': main()
