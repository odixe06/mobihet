"""Exercise the exact generated cross-account import cell on a real fixture checkpoint."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scripts import run_l50 as L, run_w20 as W, f50_common


def main():
    build=json.loads((ROOT/'runs/w-study/builder.json').read_text())
    quota=json.loads((L.OUT/'quota.json').read_text())
    eligible=[q for q in sorted(quota['accounts'],key=lambda q:-q.get('gpu_left',0)) if q.get('gpu_left',0)>=7.475 and not(q.get('inventory_error') or q['active_recent'] or q['unknown_recent'])]
    assert len(eligible)>=2
    owner=eligible[1]['user']; run='mh-l0-core-lossless-s42-r50'; kernel='mh-l50-import-preflight'
    head=__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip()
    with tempfile.TemporaryDirectory(prefix='l50-import-',dir=L.OUT) as temp:
        temp=Path(temp);source=temp/'source'/run;source.mkdir(parents=True)
        for name in ('state.pt','state_rank1.pt'):
            shutil.copyfile(L.OUT/'smoke-core/straight/state.pt',source/name)
        inputs=temp/'input';inputs.mkdir();archive=inputs/'l50-resume.bin'
        with tarfile.open(archive,'w') as tar:tar.add(source,arcname=run)
        with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        ready=dict(ref=f'{owner}/mh-l50-fixture-unpublished',archive_sha256=digest,round=4)
        L.notebook(owner,kernel,run,f50_common.hp_for('core',42),head,build,6.5,resume=ready,until=50)
        doc=json.loads((ROOT/'notebooks'/kernel/(kernel+'.ipynb')).read_text())
        code=next(''.join(c['source']) for c in doc['cells'] if 'checkpoint_files = ' in ''.join(c.get('source',[])))
        working=temp/'working';working.mkdir()
        code=code.replace('/kaggle/input',str(inputs)).replace('/kaggle/working',str(working))
        exec(code,dict(Path=Path,hashlib=hashlib,torch=torch))
        assert (working/run/'state_rank1.pt').is_file()
        with archive.open('r+b') as f:f.seek(100);f.write(b'bad')
        try:exec(code,dict(Path=Path,hashlib=hashlib,torch=torch))
        except AssertionError:pass
        else:raise AssertionError('Corrupted archive not rejected')
    (L.OUT/'transfer-preflight.json').write_text(json.dumps(dict(exact_import_cell_pass=True,tamper_rejected=True,
        source_sha256={n:W.sha(ROOT/n) for n in ('scripts/run_l50.py','scripts/stage_l50.py')}),indent=2)+'\n')
    print('L50_TRANSFER_PREFLIGHT_PASS')


if __name__=='__main__':main()
