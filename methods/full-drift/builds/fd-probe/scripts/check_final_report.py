"""Check final exports against selected audited evaluation JSONs and complete train lineages."""
import csv
import json
import math
from pathlib import Path

from collect_fd import ROOT,sha
from mh.metrics import METRIC_KEYS


def main():
    directory=ROOT/'docs/methods/full-drift/results/final'
    completion=json.loads((directory/'completion.json').read_text())
    assert len(completion)==6 and {r['clients'] for r in completion}=={20,50,100}
    for row in completion:
        assert row['seed']==42 and row['training_complete'] and row['evaluation_complete']
        assert row['training_rounds']==row['evaluated_rounds']==list(range(1,51))
    rows=list(csv.DictReader((directory/'metrics.csv').open()))
    def key(r):return int(r['clients']),r['arm'],int(r['seed']),int(r['round']),r['role'],r['rule']
    index={key(r):r for r in rows};assert len(index)==len(rows)==1800
    costs=list(csv.DictReader((directory/'costs.csv').open()))
    assert len(costs)==len({(r['clients'],r['arm'],r['seed'],r['round']) for r in costs})==300
    for row in rows:
        for metric in METRIC_KEYS:assert math.isfinite(float(row[metric])) and 0<=float(row[metric])<=1
    provenance=json.loads((directory/'provenance.json').read_text());matched=set()
    assert provenance['generator_sha']==sha(Path(__file__).with_name('report_final.py'))
    for record in provenance['training_audits']+provenance['evaluation_audits']:
        assert sha(ROOT/record['audit_path'])==record['audit_sha']
    for record in provenance['evaluation_audits']:
        proofpath=ROOT/record['audit_path'];proof=json.loads(proofpath.read_text())
        cfg=json.loads((proofpath.parent.parent/proof['session']/'runs'/proof['run']/'config.json').read_text())['config']
        for rnd in record['used_rounds']:
            actual=json.loads((proofpath.parent/'runs'/proof['run']/'eval'/f'round_{rnd:03d}.json').read_text())
            clients=len(actual['eval']['private']['test']['head']['confusion_clients'])
            for role,holds in actual['eval'].items():
                for rule,values in holds['test'].items():
                    k=(clients,cfg['arm'],cfg['seed'],rnd,role,rule);export=index[k]
                    assert k not in matched;matched.add(k)
                    for metric in METRIC_KEYS:assert abs(float(export[metric])-values['mean_metrics'][metric])<1e-12
                    assert abs(float(export['benign_f1'])-values['benign_f1'])<1e-12
    assert matched==set(index)
    classes=list(csv.DictReader((directory/'per-class-r50.csv').open()));assert len(classes)==576
    groups={}
    for row in classes:groups.setdefault(key(row),[]).append(row)
    for k,group in groups.items():
        assert {int(r['class_id']) for r in group}==set(range(16))
        for field,metric in (('f1','f1_macro'),('precision','precision_macro'),('recall','recall_macro')):
            assert abs(sum(float(r[field]) for r in group)/16-float(index[k][metric]))<1e-12
        benign=next(float(r['f1']) for r in group if int(r['class_id'])==0)
        assert abs(benign-float(index[k]['benign_f1']))<1e-12
    text=(directory/'report.md').read_text()
    assert 'đủ 6 chuỗi train + chấm r1–50 đã audit' in text and 'CHƯA ĐỦ' not in text
    for suffix in ('svg','png'):assert (directory/f'learning-curves.{suffix}').stat().st_size>1000
    result=dict(complete=True,scenarios=6,training_rounds=300,metric_rows_checked=1800,
                metric_values_checked=1800*len(METRIC_KEYS),per_class_rows_checked=576,
                report_sha=sha(directory/'report.md'),provenance_sha=sha(directory/'provenance.json'))
    (directory/'VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
