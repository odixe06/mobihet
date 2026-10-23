"""Audit P29/P30 artifacts, provenance and locked P30 local mechanism gates."""
import argparse
import json
from pathlib import Path
from statistics import mean
import sys

import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'source/proj'))
from fd_probe23 import sha
from mh.metrics import metrics_from_confusion
from check_ccvr_probes import audit as audit_heads


def audit_directions(path,value,fixture):
    arrays=np.load(path.with_suffix('.npz'))
    assert sha(path.with_suffix('.npz'))==value['predictions_sha']
    checks=0
    for split in ('old','group'):assert np.array_equal(arrays[split+'_y'],np.load(fixture/f'{split}_y.npy'))
    for client,entry in value['clients'].items():
        assert entry['autograd_max_abs_error']<1e-9 and entry['kl_autograd_max_abs_error']<1e-9
        reports={'original':entry['original']}
        for name,v in entry['directions'].items():reports[name]={s:r['report'] for s,r in v['splits'].items()}
        for name,splits in reports.items():
            for split,expected in splits.items():
                y=arrays[split+'_y'].astype('int64');pred=arrays[f'{client}_{name}_{split}'].astype('int64')
                assert pred.shape==y.shape and pred.min()>=0 and pred.max()<16
                cm=np.bincount(y*16+pred,minlength=256).reshape(16,16)
                assert np.array_equal(cm,expected['confusion'])
                for key,v in metrics_from_confusion(cm).items():assert abs(v-expected['metrics'][key])<1e-12
                f1=2*np.diag(cm)/(cm.sum(0)+cm.sum(1)).clip(1)
                assert np.allclose(f1,expected['per_class'],atol=1e-12,rtol=0)
                checks+=1
    return dict(prediction_cm_metric_checks=checks,scope='local derivative and fixed-step diagnostic; no method utility acceptance')


def gate_heads(value):
    arms=next(iter(value['clients'].values())).keys()
    summary={arm:dict(mean=mean(c[arm]['group']['metrics']['f1_macro'] for c in value['clients'].values()),
                      benign=mean(c[arm]['group']['per_class'][0] for c in value['clients'].values()),
                      worst=min(c[arm]['group']['metrics']['f1_macro'] for c in value['clients'].values())) for arm in arms}
    gates={}
    for control in ('mean_only','zero_cross','gaussian_original','gaussian_weighted'):
        delta={k:summary['joint'][k]-summary[control][k] for k in ('mean','benign','worst')}
        gates[control]=dict(delta=delta,pass_gate=delta['mean']>=.005 and delta['benign']>=-.005 and delta['worst']>=-.010)
    return dict(summary=summary,gates=gates,mechanism_ready=all(v['pass_gate'] for v in gates.values()),
                prediction_cm_metric_checks=len(value['clients'])*len(summary)*2,
                scope='local s42 frozen-head screen; not Kaggle/online/contact/R2 proof')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('paths',nargs='+',type=Path)
    parser.add_argument('--weights',required=True,type=Path);parser.add_argument('--fixture',required=True,type=Path)
    a=parser.parse_args()
    for path in a.paths:
        value=json.loads(path.read_text());is_direction='directions' in next(iter(value['clients'].values()))
        name='fd_probe29.py' if is_direction else 'fd_probe30.py'
        assert value['source_sha']==sha(BASE/'source/proj'/name)
        assert value['weights_sha']==sha(a.weights) and value['fixture_sha']==sha(a.fixture/'manifest.json')
        assert set(map(int,value['clients']))==set(range(20))
        for dependency,digest in value['dependencies'].items():assert sha(BASE/'source/proj'/dependency)==digest
        if is_direction:result=audit_directions(path,value,a.fixture)
        else:
            assert value['population_sha']==sha(a.weights.parents[3]/'source/data_manifest.json')
            audit_heads(path);result=gate_heads(value)
        result.update(complete=True,result_sha=sha(path))
        target=path.with_suffix('.audit.json');target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print('AUDITED',path,result['prediction_cm_metric_checks'],'prediction/CM/10-metric checks',flush=True)
        if 'mechanism_ready' in result:print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
