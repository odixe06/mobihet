"""Report V2 only from collected/verified run links; preserve every ten-metric row."""
import csv
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1]); sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS
from scripts.kaggle.gen_v2 import FAMILIES


def main():
    base=ROOT/'papers/v2'; runs={}; rows=[]; failures={}; classes=[]
    for path in sorted((base/'kaggle').glob('*/AUDITED.json')):
        audit=json.loads(path.read_text())
        for arm,state in audit['summary'].items():
            if not state['complete'] or state['rc']!=0:
                failures[(arm,audit['seed'])]=next(v for v in audit['audits'] if v['arm']==arm)
    for folder in sorted((base/'runs').glob('*')):
        if (folder/'result.json').exists():
            r=json.loads((folder/'result.json').read_text())
        else:
            cfg=json.loads((folder/'config.json').read_text())['config']; key=(cfg['arm'],cfg['seed'])
            if key not in failures: continue
            history=[json.loads(p.read_text()) for p in sorted((folder/'complete').glob('round_*.json'))]
            if not history: continue
            state=torch.load(folder/'resume'/f'round_{history[-1]["round"]:03d}.pt',map_location='cpu',weights_only=True)
            r=dict(config=cfg,history=history,bytes=state['bytes'],complete=False,
                   role='proxy' if cfg['arm'].startswith('b_') else 'private',minutes=sum(h['sec'] for h in history)/60)
        key=(r['config']['arm'],r['config']['seed']); runs[key]=r
        for h in r['history']:
            for role,holds in h.get('eval',{}).items():
                for hold,rules in holds.items():
                    for rule,ev in rules.items():
                        rows.append(dict(arm=key[0],seed=key[1],round=h['round'],role=role,
                                         holdout=hold,rule=rule,**ev['mean_metrics']))
                        classes.extend(dict(arm=key[0],seed=key[1],round=h['round'],role=role,
                            holdout=hold,rule=rule,class_id=c,f1=value) for c,value in enumerate(ev['f1_per_class']))
    with (base/'metrics-all.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['arm','seed','round','role','holdout','rule',*METRIC_KEYS]); w.writeheader(); w.writerows(rows)
    with (base/'per-class-f1.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['arm','seed','round','role','holdout','rule','class_id','f1']); w.writeheader(); w.writerows(classes)
    def ev(arm,seed,role='private',hold='group',rule='gp'):
        r=runs.get((arm,seed))
        return r['history'][-1]['eval'][role][hold][rule] if r and r['complete'] else None
    gates={}
    for seed in (42,43):
        local=ev('local',seed)
        if local is None: continue
        f=lambda arm,role='private',rule='gp':ev(arm,seed,role,rule=rule)['f1_macro']
        def common(arm):
            v=ev(arm,seed); old=ev(arm,seed,hold='old'); lo=ev('local',seed,hold='old')
            return bool(v['f1_macro']-local['f1_macro']>=.03 and old['f1_macro']>lo['f1_macro']
                        and v['benign_f1']>=local['benign_f1']-.03)
        if all(ev(a,seed) is not None for a in FAMILIES['a']):
            dense=f('a_dense'); own=f('a_own'); dg=dense-own
            teacher=f('a_dense','proxy','head')-local['f1_macro']>=.05
            for size in ('small','medium'):
                arm=f'a_{size}_ef'; contrast=f(arm)-own; ef=f(arm)-f(f'a_{size}_noef')
                gates[f'A_{size}_s{seed}']=dict(pass_quality=bool(teacher and common('a_dense') and common(arm)
                    and dg>0 and contrast>=.01 and contrast/dg>=.8),
                    teacher_pass=bool(teacher),gain_vs_own=contrast,retention=contrast/dg if dg>0 else None,
                    ef_contrast=ef,ef_contribution=bool(ef>=.005))
        for size in ('small','medium'):
            if (f'a_{size}_ef',seed) in failures:
                gates[f'A_{size}_s{seed}']=dict(pass_quality=False,reason='numerical failure under locked configuration',
                    committed_rounds=failures[(f'a_{size}_ef',seed)]['rounds'])
        for k in (512,1024):
            failed=failures.get((f'b_seed{k}',seed))
            if failed and failed.get('failure_category')=='numerical':
                gates[f'B_{k}_s{seed}']=dict(pass_teacher=False,reason='nonfinite proxy loss under locked configuration',
                    committed_rounds=failed['rounds'])
            elif all(ev(a,seed,'proxy',rule='head') is not None for a in ('b_dense',f'b_own{k}',f'b_seed{k}')):
                dense=f('b_dense','proxy','head')
                value=f(f'b_seed{k}','proxy','head'); own=f(f'b_own{k}','proxy','head'); dg=dense-own
                gates[f'B_{k}_s{seed}']=dict(pass_teacher=bool(value-local['f1_macro']>=.05 and dg>0 and (value-own)/dg>=.8),
                    gain_vs_local=value-local['f1_macro'],gain_vs_own=value-own,
                    retention=(value-own)/dg if dg>0 else None)
        if all(ev(a,seed) is not None for a in FAMILIES['c']):
            value=f('c_fl'); diag=f('diag'); own=f('c_own'); shuffle=f('c_shuffle')
            ess=runs[('c_fl',seed)]['history'][-1]['ratio']['ess_fraction']
            guard=bool(np.median(ess)>=.25 and min(ess)>=.10)
            gates[f'C_s{seed}']=dict(pass_quality=bool(common('c_fl') and value-diag>=.015 and value-own>=.01
                 and value-shuffle>=.01 and guard),gain_vs_diag=value-diag,gain_vs_own=value-own,
                 gain_vs_shuffle=value-shuffle,ess_guard=guard,ess_fraction=ess)
    paired={}
    for prefix,field in [('A_small','pass_quality'),('A_medium','pass_quality'),('B_512','pass_teacher'),
                         ('B_1024','pass_teacher'),('C','pass_quality')]:
        items=[gates.get(f'{prefix}_s{s}') for s in (42,43)]
        paired[prefix]=all(v[field] for v in items) if all(v is not None for v in items) else None
    (base/'gates.json').write_text(json.dumps(dict(per_seed=gates,paired=paired),indent=2))
    missing=[(arm,seed) for arms in FAMILIES.values() for arm in arms for seed in (42,43)
             if (arm,seed) not in runs and (arm,seed) not in failures]
    lines=['# V2 — báo cáo khả thi'+(' (đang thu, chưa đóng đợt)' if missing else ''),'',
        'Nguồn: các run đã pull/verify trong `papers/v2/runs/`; [ledger](rebuild.md), '
        '[cổng khóa](../../docs/research/pilot-v2.md). Stage U không giới hạn contact; chưa chứng minh R1/R2 thực nghiệm.', '',
        f'Đã có {len(runs)} run được thu. Giai đoạn B là teacher-only; chưa gọi teacher PASS là private IDS PASS.', '',
        '## Cổng hai seed','', '| hướng | trạng thái |','|---|---|']
    for key,value in paired.items(): lines.append(f'| {key} | {"PENDING" if value is None else "PASS" if value else "FAIL"} |')
    lines+=['','Chi tiết contrast/retention/ESS: [gates.json](gates.json). Không chuyển headline gp sang head để cứu cổng.',
        '', '## Headline r20 — từng seed','', '| arm | seed | role/rule | macro-F1 | benign F1 | phút job |',
        '|---|---:|---|---:|---:|---:|']
    for (arm,seed),audit in sorted(failures.items()):
        lines.append(f'| {arm} | {seed} | FAILED sau r{audit["rounds"]} | — | — | — |')
    for (arm,seed),r in sorted(runs.items()):
        if not r['complete']: continue
        role=r['role']; rule='head' if role=='proxy' else 'gp'; value=ev(arm,seed,role,rule=rule)
        lines.append(f'| {arm} | {seed} | {role}/{rule} | {value["f1_macro"]:.6f} | {value["benign_f1"]:.6f} | {r["minutes"]:.2f} |')
    lines+=['','## Loss, compute và payload','',
        'Loss real, auxiliary/KD và positive/negative NCE được lưu riêng trong marker JSON. Không suy chống học thuộc từ loss tổng.',
        'Byte dưới là application codec đã lượng tử hóa/giải mã, chưa frame/deadline hoặc cryptographic overhead.', '',
        '| arm | seed | up tổng B | down tổng B | applied | skipped |','|---|---:|---:|---:|---:|---:|']
    for (arm,seed),r in sorted(runs.items()):
        up=sum(v[-1] for v in r['bytes'] if v[2]=='up'); down=sum(v[-1] for v in r['bytes'] if v[2]=='down')
        logs=[v for row in r['history'] for name in ('clients','proxy_clients') for v in row[name]]
        lines.append(f'| {arm} | {seed} | {up} | {down} | {sum(v["applied"] for v in logs)} | {sum(v["skipped"] for v in logs)} |')
    lines+=['','## Đủ10metric mọi round đánh giá','',
        '[CSV đủ precision](metrics-all.csv). Mean metric từ confusion riêng từng client, không từ confusion gộp.', '',
        '| arm | seed | round | role | holdout | rule | '+' | '.join(METRIC_KEYS)+' |',
        '|---|---:|---:|---|---|---|'+'---:|'*10]
    for row in rows:
        lines.append('| '+' | '.join([str(row[k]) for k in ('arm','seed','round','role','holdout','rule')]+
                                     [f'{row[k]:.6f}' for k in METRIC_KEYS])+' |')
    lines+=['','## Giới hạn bằng chứng','',
        'Verifier strict-load mọi checkpoint, state/payload/hash chain, predictions→confusion từng client→10metric; '
        'sample inference client0 group ở các checkpoint định trước, decisive margin>0,02; không re-infer toàn bộ holdout.',
        'Hai seed không CI. Holdout nội bộ có scaler leakage từ train80%; split/contact/client là simulator, '
        'không đo đường thật. BN policy khác H1 cũ; không dùng số P2 có codec/prior khác làm đối chứng trực tiếp.',
        'Không tune/kéo dài hướng trượt. Chỉ PASS U mới xem xét contact với codec/manifest riêng; '
        'B cần private-KD gate sau teacher. Git/DVC chưa backup trong phiên này.']
    (base/'report.md').write_text('\n'.join(lines)+'\n')
    if rows:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True)
        for ax,seed in zip(axs,(42,43)):
            for (arm,s),r in sorted(runs.items()):
                if s!=seed: continue
                role=r['role']; rule='head' if role=='proxy' else 'gp'
                pts=[(h['round'],h['eval'][role]['group'][rule]['f1_macro']) for h in r['history'] if 'eval' in h]
                if pts: ax.plot(*zip(*pts),marker='.',label=arm)
            ax.set_title(f'seed{seed}'); ax.set_xlabel('round'); ax.grid(alpha=.2)
        axs[0].set_ylabel('group macro-F1 (private gp / teacher head)')
        handles,labels=axs[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=8)
        fig.tight_layout(rect=(0,.22,1,1)); fig.savefig(base/'convergence.png',dpi=180); fig.savefig(base/'convergence.svg'); plt.close(fig)
        figures=['','## Hình từ artifact đã kiểm','',
            '![Convergence](convergence.png)',
            'Đường dừng sớm giữ đúng checkpoint đã có; không kéo tới r20. Private gp và teacher head là hai role khác nhau.',
            '', '[F1 từng lớp](per-class-f1.csv). Class ID theo `knowledge/meta.json`.', '']
        for family,arms in FAMILIES.items():
            for seed in (42,43):
                panels=[]
                for arm in arms:
                    r=runs.get((arm,seed))
                    if r is None: continue
                    evaluated=[h for h in r['history'] if 'eval' in h]
                    if not evaluated: continue
                    h=evaluated[-1]; role=r['role']; rule='head' if role=='proxy' else 'gp'
                    value=h['eval'][role]['group'][rule]
                    panels.append((arm,h['round'],value))
                if not panels: continue
                fig,axes=plt.subplots(2,len(panels),figsize=(3.2*len(panels),6),squeeze=False)
                for col,(arm,round_no,value) in enumerate(panels):
                    cm=np.asarray(value['confusion_clients']).sum(axis=0)
                    normalized=cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
                    axes[0,col].imshow(normalized,vmin=0,vmax=1,cmap='Blues')
                    axes[0,col].set_xticks([0,5,10,15]); axes[0,col].set_yticks([0,5,10,15])
                    axes[0,col].set_title(f'{arm} r{round_no}',fontsize=9)
                    axes[0,col].set_xlabel('predicted class'); axes[0,col].set_ylabel('true class')
                    axes[1,col].bar(range(16),value['f1_per_class']); axes[1,col].set_ylim(0,1)
                    axes[1,col].set_xticks([0,5,10,15])
                    axes[1,col].set_xlabel('class ID'); axes[1,col].set_ylabel('mean client F1')
                fig.tight_layout(); name=f'classes-{family}-s{seed}.png'
                fig.savefig(base/name,dpi=150); plt.close(fig)
                figures.extend([f'![{family} seed{seed}]({name})',
                    'Trên: confusion gộp để minh họa, chuẩn hóa theo nhãn thật. Dưới: F1 trung bình từng client; '
                    'checkpoint cuối có đánh giá ghi trong tiêu đề, không phải best round.', ''])
        with (base/'report.md').open('a') as f: f.write('\n'.join(figures)+'\n')
    print('report',len(runs),'runs',len(rows),'ten-metric rows; paired gates',paired)


if __name__=='__main__': main()
