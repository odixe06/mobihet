"""All ten metrics for every audited full round; label calibration and continuations honestly."""
import argparse
import csv
import hashlib
import io
import json
import sys
from pathlib import Path
import numpy as np

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
sys.path.insert(0,str(ROOT))
from mh.metrics import METRIC_KEYS


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--build',choices=('full-validation','hybrid-full'),default='full-validation')
    build=parser.parse_args().build
    base=ROOT/'papers'/build
    metrics=[];classes=[];runs={};budget=[]
    for marker in sorted((base/'kaggle').glob('*/AUDITED.json')):
        record=json.loads(marker.read_text())
        manifest_path=marker.parent/'source/data_manifest.json'
        if not manifest_path.exists():continue
        data=json.loads(manifest_path.read_text())
        assert data['source_rows']==43045415 and data['test_rows']==10761343
        for audit in record.get('audits',[]):
            arm=audit['arm'];run=marker.parent/'runs'/f'{arm}_s42'
            identity=json.loads((run/'config.json').read_text())
            assert hashlib.sha256((json.dumps(data,indent=2)+'\n').encode()).hexdigest()==identity['fixture']
            rows=[json.loads(p.read_text()) for p in sorted((run/'complete').glob('round_*.json'))]
            assert len(rows)==audit['rounds']
            for row in rows:
                with np.load(run/f'predictions/round_{row["round"]:03d}.npz') as predictions:
                    labels=predictions['test_y']
                    serialized=io.BytesIO();np.save(serialized,labels)
                    assert hashlib.sha256(serialized.getbuffer()).hexdigest()==data['files']['test_y.npy'], 'wrong test labels/order'
                    assert np.array_equal(np.bincount(labels,minlength=16),data['test_counts'])
                for role,holds in row['eval'].items():
                    for rule,value in holds['test'].items():
                        common=dict(arm=arm,seed=42,round=row['round'],role=role,holdout='test',rule=rule)
                        assert set(value['mean_metrics'])==set(METRIC_KEYS)
                        metrics.append(dict(common,**value['mean_metrics']))
                        classes.extend(dict(common,class_id=j,class_name=data['classes'][j],f1=f) for j,f in enumerate(value['f1_per_class']))
            previous=runs.get(arm)
            if previous:
                for a,b in zip(previous['rows'],rows):assert a==b,'continuation changed overlapping history'
            if not previous or len(rows)>len(previous['rows']):
                runs[arm]=dict(rows=rows,path=str(run.relative_to(ROOT)),complete=audit['complete'])
            budget.append(dict(kernel=record['kernel'],arm=arm,**record['summary'][arm]))
    # Imported history may occur in several sessions; identical duplicates are one round.
    unique={}
    for row in metrics:
        key=tuple(row[k] for k in ('arm','seed','round','role','holdout','rule'))
        assert key not in unique or unique[key]==row
        unique[key]=row
    metrics=sorted(unique.values(),key=lambda r:(r['arm'],r['round'],r['role'],r['rule']))
    for filename,rows,fields in (
        ('metrics-all.csv',metrics,['arm','seed','round','role','holdout','rule',*METRIC_KEYS]),
        ('per-class-f1.csv',classes,['arm','seed','round','role','holdout','rule','class_id','class_name','f1'])):
        with (base/filename).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    (base/'budget-measured.json').write_text(json.dumps(budget,indent=2))
    title='Hybrid full' if build=='hybrid-full' else 'Full H1/H2'
    lines=[f'# {title} — kết quả đã audit, chưa đủ50round','',
        '20client,43.045.415trainrow/10.761.343rawtestrow, scalerđúngmộtlần, seed42; learnedmodelsDAGSNet.',
        'Protocol train80%/test20% cốđịnh; testdescriptive, khôngHP/roundselection. StageU, chưaR1/R2/DP.',
        'Meanclient10metric; pooledconfusion khônglàheadline. Accuracy=precision_micro=recall_micro=recall_weighted=f1_micro.',
        'Sybil/session vàreceiverfingerprint shortcuts, benign-source bias; timeDelay khôngcóđủsignal trongcontent.',
        'Nhữnggiớihạn này khôngđượcgiải quyết bằngsource/weightsaudit. [Dataset](../../knowledge/dataset.md).','',
        '| arm | completed/planned | source/weights/prediction |','|---|---:|---|']
    for arm,run in sorted(runs.items()):lines.append(f'| {arm} | {len(run["rows"])}/50 | `{run["path"]}` |')
    lines+=['','## Mọi evaluation đã commit — đủ10metrics','',
        '[CSV sốđầyđủ](metrics-all.csv), [F1 từnglớp](per-class-f1.csv), [budget T4](budget-measured.json), [ledger](rebuild.md).',
        'R3calibration khônglàrun50round đãhoànthành; khôngchọnbestround hoặcđổihead đểcứu kếtquả.','',
        '| arm | round | role | rule | '+' | '.join(METRIC_KEYS)+' |',
        '|---|---:|---|---|'+'---:|'*10]
    for row in metrics:
        lines.append('| '+' | '.join([str(row[k]) for k in ('arm','round','role','rule')]+[f'{row[k]:.6f}' for k in METRIC_KEYS])+' |')
    (base/'report.md').write_text('\n'.join(lines)+'\n')
    latest=[f'# {title} — 10 metrics các round cuối đã audit','',
        '100% train80%, seed42; mean20client, private/test/gp. Round cuối hiện có, chưa phải r50.',
        'Official test chỉ mô tả, không chọn HP/best round hoặc kết luận paired utility từ một seed.',
        '[Toàn bộ round/source/audit](report.md) · [CSV](metrics-all.csv)','',
        '| arm | round | '+' | '.join(METRIC_KEYS)+' |','|---|---:|'+'---:|'*10]
    for arm,run in sorted(runs.items()):
        last=len(run['rows'])
        for row in metrics:
            if row['arm']==arm and row['role']=='private' and row['rule']=='gp' and row['round']>max(0,last-3):
                latest.append('| '+' | '.join([arm,str(row['round'])]+[f'{row[k]:.6f}' for k in METRIC_KEYS])+' |')
    (base/'latest-rounds.md').write_text('\n'.join(latest)+'\n')
    print('Full report:',len(runs),'audited arms,',len(metrics),'ten-metric rows; actual maxrounds',
        {k:len(v['rows']) for k,v in runs.items()})


if __name__=='__main__':main()
