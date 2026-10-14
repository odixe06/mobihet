"""Exercise production resume import using real bounded smoke artifacts and corruption cases."""
import json
import shutil
import sys
from pathlib import Path
import torch

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'papers/full-validation/proj'))
import scripts  # register compatibility imports before loading the frozen verifier
from scripts.kaggle.full_resume import import_run
from full_verify import verify
from scripts.check_v3_driver import equal


def main():
    torch.set_num_threads(1)
    source=ROOT/'papers/full-validation/smoke/initial1-20261012/h_gaussian_s42'
    if not source.exists():
        candidates=list((ROOT/'papers/full-validation/smoke').glob('*/h_gaussian_s42'))
        assert len(candidates)==1,candidates
        source=candidates[0]
    identity=json.loads((source/'config.json').read_text())
    expected=dict(fingerprint=identity['fingerprint'],source=identity['source'],fixture=identity['fixture'],last_round=3)
    work=ROOT/'papers/full-validation/smoke/import-20261013'
    mount=work/'mount';previous=mount/source.name
    assert not work.exists(),'new import check tag required'
    shutil.copytree(source,previous)
    record=import_run(mount,work/'output','h_gaussian',expected,verify)
    restored=work/'output'/source.name
    for r in range(1,4):
        for kind in ('weights','resume'):
            assert equal(torch.load(source/f'{kind}/round_{r:03d}.pt',weights_only=True,map_location='cpu'),
                         torch.load(restored/f'{kind}/round_{r:03d}.pt',weights_only=True,map_location='cpu'))
    failures=0
    for case,location,descriptor in (
        ('empty',work/'empty',expected),
        ('wrong-round',mount,dict(expected,last_round=2)),
        ('wrong-source',mount,dict(expected,source={})),
        ('wrong-data',mount,dict(expected,fixture='invalid'))):
        try:import_run(location,work/case,'h_gaussian',descriptor,verify)
        except AssertionError:failures+=1
        else:raise AssertionError(('missing resume gate',case))
    duplicate=mount/'duplicate'/source.name;duplicate.mkdir(parents=True)
    shutil.copyfile(previous/'config.json',duplicate/'config.json')
    try:import_run(mount,work/'ambiguous','h_gaussian',expected,verify)
    except AssertionError:failures+=1
    else:raise AssertionError('ambiguous source accepted')
    tampered=work/'tampered'/source.name;shutil.copytree(source,tampered)
    with (tampered/'weights/round_003.pt').open('ab') as file:file.write(b'broken')
    try:import_run(work/'tampered',work/'bad-output','h_gaussian',expected,verify)
    except AssertionError:failures+=1
    else:raise AssertionError('tampered checkpoint accepted')
    print('PASS exact full import:',record['last_round'],'rounds; six refusal cases; original source untouched')


if __name__=='__main__':main()
