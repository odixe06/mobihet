"""Validate actual utility controllers, import refusals and W&B resume step behavior."""
import contextlib
import io
import json
import netrc
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import nbformat

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
sys.path.insert(0,str(ROOT))
from scripts.kaggle.boundary_logging import install
from scripts.kaggle.boundary_resume import import_run


def helpers():
    calls=[]
    class Run:
        def log(self,data,*args,**kwargs):calls.append((data,args,kwargs))
    install(SimpleNamespace(sdk=SimpleNamespace(wandb_run=SimpleNamespace(Run=Run))))
    run=Run();run.log({'preflight/online':1});run.log({'round':3},step=3)
    assert calls[0][2]=={'commit':False} and calls[1][2]=={'step':3}
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);source=root/'mount/bc_single_s42';source.mkdir(parents=True)
        identity=dict(fingerprint='fp',source={'mh/a.py':'sha'},fixture='fixture')
        (source/'config.json').write_text(json.dumps(identity))
        (source/'complete').mkdir();(source/'complete/round_001.json').write_text('{}')
        (source/'weights').mkdir();(source/'weights/round_001.pt').write_bytes(b'bounded-checkpoint')
        expected=dict(identity,last_round=1)
        def verify(folder,*args,**kwargs):
            assert (folder/'weights/round_001.pt').read_bytes()==b'bounded-checkpoint'
            return dict(rounds=1,cost={'pass_cost':True})
        record=import_run(root/'mount',root/'out',source.name,expected,{},verify)
        assert record['markers_last'] and record['copied_files']==3
        for label,path,value,checker in (
            ('empty',root/'missing',expected,verify),
            ('source',root/'mount',dict(expected,source={}),verify),
            ('fixture',root/'mount',dict(expected,fixture='wrong'),verify),
            ('cost',root/'mount',expected,lambda *a,**k:dict(rounds=1,cost={'pass_cost':False}))):
            try:import_run(path,root/label,source.name,value,{},checker)
            except AssertionError:pass
            else:raise AssertionError(('import refusal missing',label))
    return dict(import_copy_and_refusals=True,wandb_preflight_no_step_advance=True,
                limit='bounded import fixture; actual production parent reverified on T4 before data decode')


def controller(folder,case):
    meta=json.loads((folder/'kernel-metadata.json').read_text());nb=nbformat.read(folder/meta['code_file'],as_version=4)
    key=netrc.netrc().authenticators('api.wandb.ai')[2]
    private=folder.parents[1]/'private_push'/folder.name/meta['code_file']
    assert key not in (folder/meta['code_file']).read_text() and key in private.read_text()
    assert private.stat().st_mode & 0o777==0o600
    assert meta['is_private'] and meta['enable_gpu'] and meta['enable_internet']
    assert meta['machine_shape']=='NvidiaTeslaT4' and '@sha256:' in meta['docker_image']
    assert nb.metadata['kaggle']['isGpuEnabled'] and nb.metadata['kaggle']['isInternetEnabled']
    assert nb.cells[2].source.index('imports=[import_run(')<nb.cells[2].source.index('x=torch.from_numpy(np.load(')
    expected=json.loads((folder/'resume-plan.json').read_text())
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type=='code':compile(cell.source,str(folder),'exec')
    with tempfile.TemporaryDirectory() as directory:
        out=Path(directory);launches=[]
        for name,value in expected.items():
            target=out/name;target.mkdir()
            (target/'result.json').write_text(json.dumps(dict(history=[dict(round=r) for r in range(1,value['last_round']+1)])))
        class Clock:
            @staticmethod
            def monotonic():return 12*3600 if case=='budget' else 0
            @staticmethod
            def sleep(_):pass
        class Process:
            def __init__(self,command,**kwargs):
                arm=command[command.index('--arm')+1];seed=command[command.index('--seed')+1]
                name=f'{arm}_s{seed}';launches.append(name)
                assert '--smoke' not in command and '--stop_after' not in command
                assert command[command.index('--rounds')+1]=='20'
                assert command[command.index('--aux_weight')+1]=='.5' and command[command.index('--temperature')+1]=='1'
                assert ('--resume' in command)==(name in expected)
                (out/name/'result.json').write_text(json.dumps(dict(history=[dict(round=r) for r in range(1,21)])))
                self.returncode=7 if case=='failure' and len(launches)==3 else 0
            def poll(self):return self.returncode
            def terminate(self):self.returncode=-15
            def kill(self):self.returncode=-9
        scope=dict(Path=Path,json=json,os=os,sys=sys,time=Clock,WORK=ROOT,OUT=out,EXPECTED=expected,
                   SESSION_T0=0,subprocess=SimpleNamespace(Popen=Process,STDOUT=-2))
        try:exec(compile(nb.cells[3].source,'actual BoundaryCells utility','exec'),scope)
        except AssertionError:assert case=='failure'
        else:assert case!='failure'
        summary=json.loads((out/'utility_summary.json').read_text())
        assert len(summary)==20
        if case=='budget':
            assert not launches and not any(x['complete'] for x in summary.values())
            assert sum(x['rounds'] for x in summary.values())==10
        elif case=='normal':assert len(launches)==20 and all(x['complete'] for x in summary.values())
        else:assert any(x.get('rc')==7 for x in summary.values())
    return dict(kernel=meta['id'],case=case,pass_check=True)


def main():
    rows=[]
    with contextlib.redirect_stdout(io.StringIO()):
        support=helpers()
        for folder in sorted((ROOT/'papers/boundary-cells/notebook').glob('*20261014-stage01*')):
            rows.extend(controller(folder,case) for case in ('normal','budget','failure'))
    assert len(rows)==6
    (ROOT/'docs/operations/boundary-utility-preflight-2026-10-14.json').write_text(json.dumps(
        dict(helpers=support,controllers=rows,scientific_source_changed=False),indent=2)+'\n')
    print('Boundary utility PASS: 6 emitted controller cases, import refusals, W&B step preservation')


if __name__=='__main__':main()
