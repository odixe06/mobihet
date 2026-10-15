"""Actual recovery controller: imported r3, wrapper/resume arguments, cost and budget refusal."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import nbformat

BASE=Path(__file__).resolve().parents[1]
cases=[]
for folder in sorted((BASE/'notebooks').glob('cr-u-s*-20261015-v2')):
    seed=json.loads((folder/'recovery.json').read_text())['seed'];meta=json.loads((folder/'kernel-metadata.json').read_text())
    nb=nbformat.read(folder/meta['code_file'],as_version=4)
    for case in ('normal','cost-fail','worker-fail','budget'):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'manifest.json').write_text('{}');calls=[];verified=[]
            for arm in ('cr_local','cr_fl'):
                p=out/f'{arm}_s{seed}';(p/'complete').mkdir(parents=True)
                (p/'result.json').write_text('{}')
                (p/'complete/round_003.json').write_text(json.dumps(dict(algorithm_seconds=30 if case=='cost-fail' and arm=='cr_fl' else 10,sec=10)))
            def verify(run,*_):
                verified.append(run.name);return dict(logical_rounds=3,committed_rounds=1)
            original=sys.modules.get('rank_verify');sys.modules['rank_verify']=SimpleNamespace(verify=verify)
            class Process:
                def __init__(self,cmd,**_):
                    arm=cmd[cmd.index('--arm')+1];calls.append(arm)
                    assert 'rank_bytes_compat.py' in cmd[1] and 'boundary_logging.py' in cmd[2]
                    assert '--stop_after' not in cmd and ('--resume' in cmd)==(arm in ('cr_local','cr_fl'))
                    (out/f'{arm}_s{seed}/result.json').write_text(json.dumps(dict(logical_rounds=20,inherited_rounds=0 if arm in ('cr_gaussian','cr_dml') else 2,complete=True)))
                    self.returncode=7 if case=='worker-fail' else 0
                def poll(self):return self.returncode
            scope=dict(Path=Path,json=json,os=__import__('os'),sys=sys,OUT=out,WORK=out,SRC=out,SESSION_T0=0,MANIFEST={},hashlib=hashlib,
                subprocess=SimpleNamespace(Popen=Process,STDOUT=-2),time=SimpleNamespace(monotonic=lambda:2.24*3600 if case=='budget' and len(verified)==2 else 0,sleep=lambda _:None))
            try:
                try:exec(compile(nb.cells[3].source,'emitted recovery controller','exec'),scope)
                except AssertionError:assert case in ('cost-fail','worker-fail')
                else:
                    assert case in ('normal','budget');summary=json.loads((out/'utility_summary.json').read_text())
                    assert len(summary)==8 and all(v['complete'] for v in summary.values())==(case=='normal')
                assert len(verified)==2
                if case in ('cost-fail','budget'):assert not calls
                if case=='normal':assert len(calls)==8
            finally:
                if original is None:sys.modules.pop('rank_verify',None)
                else:sys.modules['rank_verify']=original
        cases.append(dict(kernel=meta['id'],case=case,pass_check=True))
    (folder/'controller-preflight.json').write_text(json.dumps(cases[-4:],indent=2)+'\n')
(BASE/'probes/recovery-controller-preflight.json').write_text(json.dumps(dict(pass_check=bool(cases),cases=cases),indent=2)+'\n')
print('Recovery emitted controllers PASS',len(cases))
