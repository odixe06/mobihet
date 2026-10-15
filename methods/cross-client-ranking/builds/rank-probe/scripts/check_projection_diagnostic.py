"""Exact projection algebra and emitted diagnostic-controller failure/budget fixtures."""
import json
import netrc
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import nbformat
import numpy as np

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
sys.path.insert(0,str(BASE/'source/proj'))
from rank_math import basis
from rank_projection_diagnostic import measure


def main():
    rng=np.random.default_rng(1729);z=rng.normal(size=(512,16));y=np.arange(512)%16
    prob=np.exp(z-z.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
    fit=np.arange(512)//16%2==0;delta=np.einsum('nkd,d->nk',basis(prob,y,z),rng.normal(size=32))
    d=measure(prob,y,z,delta,fit)
    assert d['test']['residual']<1e-20 and d['rank']==32
    rows=[]
    for seed in (42,43):
        folder=BASE/'notebooks'/f'ccr-diag-s{seed}-20261015-v1';meta=json.loads((folder/'kernel-metadata.json').read_text())
        nb=nbformat.read(folder/meta['code_file'],as_version=4);nbformat.validate(nb)
        key=netrc.netrc().authenticators('api.wandb.ai')[2];assert key not in (folder/meta['code_file']).read_text()
        private=BASE/'private_push'/folder.name/meta['code_file'];assert key in private.read_text() and private.stat().st_mode&0o777==0o600
        for case in ('normal','failure','budget'):
            with tempfile.TemporaryDirectory() as tmp:
                out=Path(tmp);launches=[]
                class Process:
                    def __init__(self,command,**kwargs):
                        assert '--phase' not in command and 'rank_projection_diagnostic.py' in command[1]
                        shard=int(command[-1]);launches.append(shard)
                        for k in range(shard,20,2):(out/f'diagnostic_{k:02d}.json').write_text(json.dumps(dict(original_energies_reproduced=True)))
                        self.returncode=7 if case=='failure' else 0
                    def poll(self):return self.returncode
                wb=SimpleNamespace(settings=SimpleNamespace(_offline=False),log=lambda _:None,finish=lambda:None)
                scope=dict(Path=Path,json=json,os=os,sys=sys,OUT=out,WORK=out,SRC=out,SESSION_T0=0,
                    DIAGNOSTIC_SHA='fixture',REPLAY=dict(parent='fixture'),
                    time=SimpleNamespace(monotonic=lambda:10000 if case=='budget' else 0,sleep=lambda _:None),
                    subprocess=SimpleNamespace(Popen=Process,STDOUT=-2),
                    wandb=SimpleNamespace(init=lambda **_:wb,Settings=lambda **_:None))
                try:exec(compile(nb.cells[3].source,'actual diagnostic controller','exec'),scope)
                except AssertionError:assert case!='normal'
                else:
                    assert case=='normal' and launches==[0,1]
                    assert json.loads((out/'diagnostic_result.json').read_text())['models']==20
            rows.append(dict(kernel=meta['id'],case=case,pass_check=True))
        (folder/'controller-preflight.json').write_text(json.dumps(rows[-3:],indent=2)+'\n')
    (BASE/'probes/diagnostic-preflight.json').write_text(json.dumps(dict(exact_basis_residual=d['test']['residual'],controllers=rows),indent=2)+'\n')
    print('Diagnostic PASS: exact calibration-span residual; 6 emitted controller cases')


if __name__=='__main__':main()
