"""Audited repair quality, partial feasibility and actual sparse-packet forensics."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS
from mh.v2_math import unpack,CountSketch,Layout
from mh.dagsnet import build_model,CFG


def carrier(run,rows):
    small='small' in run.name;depth,width,k=(3,256,128) if small else (5,2048,2048)
    sketch=CountSketch(395024,depth,width,1701,'cpu');union=torch.zeros(395024,dtype=torch.bool)
    previous=torch.zeros(depth*width);ties=[];zero=0
    for r in rows:
        number=r['round'];state=torch.load(run/f'resume/round_{number:03d}.pt',map_location='cpu',weights_only=True)['algorithm']
        update=unpack((run/f'knowledge/round_{number:03d}.bin').read_bytes(),expected_base=number-1)
        indices=update['indices'][update['values']!=0];union[indices]=True;zero+=len(indices)==0
        estimate=sketch.estimate(state['last_mean']+previous).abs()
        threshold=estimate.kthvalue(len(estimate)-k+1).values
        ties.append(int((estimate==threshold).sum()));previous=state['residual']
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True);at=0;layers=[]
    for name,shape,n in Layout(model).items:
        layers.append(dict(name=name,parameters=n,touched=int(union[at:at+n].sum())));at+=n
    return dict(rounds=len(rows),unique_coordinates=int(union.sum()),parameter_fraction=float(union.float().mean()),
        zero_updates=zero,median_ties_at_topk_cutoff=float(np.median(ties)),layers=layers,
        note='Actual nonzero wire patches; coverage does not measure raw-update reconstruction error')


def main():
    torch.set_num_threads(1)
    out=ROOT/'docs/methods/proxy-countsketch-repair/builds/contractive-ef-pilot/results';out.mkdir(exist_ok=True)
    metrics=[];perclass=[];gates={};forensics={};table=[]
    for seed in (42,43):
        parent=BASE/'artifacts/kaggle'/f'cef-s{seed}-20261015-v1'
        if not (parent/'AUDITED.json').exists():continue
        audited=json.loads((parent/'AUDITED.json').read_text());runs={}
        for audit in audited['audits']:
            name=audit['run'];arm=name.rsplit('_s',1)[0];run=parent/'runs'/name
            rows=[json.loads(p.read_text()) for p in sorted((run/'complete').glob('*.json'))];runs[arm]=rows
            for row in rows:
                for role,holds in row['eval'].items():
                    for hold,rules in holds.items():
                        for rule,ev in rules.items():
                            base=dict(seed=seed,arm=arm,round=row['round'],role=role,holdout=hold,rule=rule)
                            metrics.append(dict(base,**ev['mean_metrics']))
                            perclass.extend(dict(base,class_id=c,f1=v) for c,v in enumerate(ev['f1_per_class']))
            last=rows[-1];ev=last['eval']['private']['group']
            table.append((seed,arm,len(rows),ev['gp']['f1_macro'],ev['head']['f1_macro']))
            if '_contract_' in arm:forensics[name]=carrier(run,rows)
        def ev(arm,r=20,role='private',hold='group',rule='gp'):
            return runs[arm][r-1]['eval'][role][hold][rule]
        def common(arm):
            return ev(arm)['f1_macro']-ev('local')['f1_macro']>=.03 and ev(arm,hold='old')['f1_macro']>ev('local',hold='old')['f1_macro'] and ev(arm)['benign_f1']>=ev('local')['benign_f1']-.03
        for size in ('small','medium'):
            arm=f'a_{size}_contract_ef';n=len(runs[arm]);end=ev(arm,r=n)
            contrast=end['f1_macro']-ev('a_own',r=n)['f1_macro']
            recent=[r['eval']['private']['group']['gp']['f1_macro'] for r in runs[arm][-5:]]
            if n==20:
                dense_gain=ev('a_dense')['f1_macro']-ev('a_own')['f1_macro']
                teacher=ev('a_dense',role='proxy',rule='head')['f1_macro']-ev('local')['f1_macro']>=.05
                ef=ev(arm)['f1_macro']-ev(f'a_{size}_noef')['f1_macro']
                quality=teacher and common('a_dense') and common(arm) and dense_gain>0 and contrast>=.01 and contrast/dense_gain>=.8
                gates[f'{size}_s{seed}']=dict(rounds=n,quality_pass=bool(quality),teacher_pass=bool(teacher),
                    gain_vs_own=contrast,retention=contrast/dense_gain if dense_gain>0 else None,
                    ef_contrast=ef,ef_contribution=ef>=.005,last5=recent)
            else:
                gates[f'{size}_s{seed}']=dict(rounds=n,quality_pass=None,aligned_gain_vs_own=contrast,
                    last5=recent,feasibility='no support for extension if far below own and flat/declining; not fixed-r20 FAIL')
    for name,rows,extra in [('metrics-all.csv',metrics,list(METRIC_KEYS)),('per-class-f1.csv',perclass,['class_id','f1'])]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['seed','arm','round','role','holdout','rule',*extra]);w.writeheader();w.writerows(rows)
    (out/'gates.json').write_text(json.dumps(gates,indent=2)+'\n')
    (BASE/'probes/terminal-carrier-forensics.json').write_text(json.dumps(forensics,indent=2)+'\n')
    lines=['# CountSketch repair — audited utility and partial feasibility','',
        'Decoded residual contraction/numerical stability không chứng minh proxy quality. Primary group/private/gp giữ nguyên.',
        '', '| seed | arm | rounds/20 | private gp | private head diagnostic |','|---|---|---:|---:|---:|']
    lines += [f'| {s} | {a} | {n}/20 | {gp:.6f} | {head:.6f} |' for s,a,n,gp,head in table]
    lines+=['','[Toàn10metrics](metrics-all.csv) · [Per-class](per-class-f1.csv) · [Fixed-r20 gates/partial curves](gates.json).',
        '', 'Các arm chưa20 không được gọi strict-r20 FAIL. Không đổi gp sang head để cứu gate.',
        'Sparse patch coverage/ties chỉ mô tả carrier; không cô lập raw-update error hoặc bác cả họ CountSketch/EF.',
        'Không có official test hoặc inference từ saved weights trong audit này.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('Repair report',len(metrics),'10metric rows;',len(gates),'arm/seed gates')


if __name__=='__main__':main()
