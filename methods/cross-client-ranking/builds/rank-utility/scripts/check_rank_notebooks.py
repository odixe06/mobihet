"""Exercise actual emitted cost/utility controllers, including cost failure and exhausted budget."""
import json
import netrc
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import nbformat

BASE=Path(__file__).resolve().parents[1]


def main():
    cases=[]
    for seed in (42,43):
        folder=BASE/'notebooks'/f'cr-u-s{seed}-20261015-v1';meta=json.loads((folder/'kernel-metadata.json').read_text())
        nb=nbformat.read(folder/meta['code_file'],as_version=4);nbformat.validate(nb)
        assert meta['is_private'] and meta['enable_gpu'] and meta['machine_shape']=='NvidiaTeslaT4' and '@sha256:' in meta['docker_image']
        assert nb.metadata['kaggle']['accelerator']=='nvidiaTeslaT4'
        key=netrc.netrc().authenticators('api.wandb.ai')[2];assert key not in (folder/meta['code_file']).read_text()
        private=BASE/'private_push'/folder.name/meta['code_file'];assert key in private.read_text() and private.stat().st_mode&0o777==0o600
        for case in ('normal','cost-fail','worker-fail','budget'):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);launches=[];verified=[]
                def verify(run,*_):
                    verified.append(run.name)
                    return dict(logical_rounds=3,committed_rounds=1)
                prior=sys.modules.get('rank_verify');sys.modules['rank_verify']=SimpleNamespace(verify=verify)
                (out/'manifest.json').write_text('{}')
                class Process:
                    def __init__(self,command,**kwargs):
                        arm=command[command.index('--arm')+1];gate='--stop_after' in command;launches.append((arm,gate))
                        assert '--warm' in command and '--wandb_online' in command and command[command.index('--rounds')+1]=='20'
                        assert command[command.index('--temperature')+1]=='1' and command[command.index('--weight_cap')+1]=='4'
                        inherited=0 if arm in ('cr_gaussian','cr_dml') else 2;folder_=out/f'{arm}_s{seed}'
                        assert ('--resume' in command)==(not gate and arm in ('cr_local','cr_fl'))
                        if gate:
                            assert command[command.index('--stop_after')+1]=='3'
                            (folder_/'complete').mkdir()
                            row=dict(algorithm_seconds=25 if case=='cost-fail' and arm=='cr_fl' else 15 if arm=='cr_fl' else 10,
                                sec=25 if arm=='cr_fl' else 20)
                            (folder_/'complete/round_003.json').write_text(json.dumps(row))
                        (folder_/'result.json').write_text(json.dumps(dict(logical_rounds=3 if gate else 20,inherited_rounds=inherited,complete=not gate)))
                        self.returncode=7 if case=='worker-fail' and not gate else 0
                    def poll(self):return self.returncode
                scope=dict(Path=Path,json=json,os=os,sys=sys,OUT=out,WORK=out,SRC=out,SESSION_T0=0,MANIFEST={},
                    hashlib=__import__('hashlib'),subprocess=SimpleNamespace(Popen=Process,STDOUT=-2),
                    time=SimpleNamespace(monotonic=lambda:2.4*3600 if case=='budget' and len(verified)==2 else 0,sleep=lambda _:None))
                try:
                    try:exec(compile(nb.cells[3].source,'emitted rank utility controller','exec'),scope)
                    except AssertionError:assert case in ('cost-fail','worker-fail')
                    else:
                        assert case in ('normal','budget')
                        summary=json.loads((out/'utility_summary.json').read_text());assert len(summary)==8
                        assert all(v['complete'] for v in summary.values())==(case=='normal')
                    if case=='cost-fail':assert len(launches)==2
                    if case=='normal':assert len(launches)==10 and len(verified)==2
                    if case=='budget':assert len(launches)==2
                finally:
                    if prior is None:sys.modules.pop('rank_verify',None)
                    else:sys.modules['rank_verify']=prior
            cases.append(dict(kernel=meta['id'],case=case,pass_check=True))
        (folder/'controller-preflight.json').write_text(json.dumps(cases[-4:],indent=2)+'\n')
    (BASE/'probes/notebook-preflight.json').write_text(json.dumps(dict(controllers=cases,pass_check=True),indent=2)+'\n')
    print('Rank notebooks PASS:8 emitted controller cases; private0600/metadata/CE-r3 cost gate')


if __name__=='__main__':main()
