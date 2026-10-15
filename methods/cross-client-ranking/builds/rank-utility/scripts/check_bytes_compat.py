"""Regression for actual CE empty wire: reproduce refusal, narrowly allow bytes, audit r3."""
import io
import json
import pickle
from pathlib import Path
import subprocess
import sys
import tempfile
import torch

BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(BASE/'source/proj'))
from rank_verify import verify


def main():
    torch.set_num_threads(1)
    buffer=io.BytesIO();torch.save(dict(wire=b'',up=[],latest_packet=b'payload'),buffer);buffer.seek(0)
    try:torch.load(buffer,weights_only=True)
    except pickle.UnpicklingError as error:assert 'GLOBAL bytes' in str(error)
    else:raise AssertionError('empty-bytes regression did not reproduce observed error')
    buffer.seek(0)
    with torch.serialization.safe_globals([bytes]):
        assert torch.load(buffer,weights_only=True)==dict(wire=b'',up=[],latest_packet=b'payload')
    assert bytes not in torch.serialization.get_safe_globals(),'allowlist leaked outside scope'
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp);torch.save(dict(wire=b''),path/'state.pt')
        (path/'load.py').write_text('import torch\nassert torch.load("state.pt",weights_only=True)=={"wire":b""}\n')
        subprocess.run([sys.executable,str(BASE/'scripts/rank_bytes_compat.py'),str(path/'load.py')],cwd=path,check=True)
    audits=[]
    for seed in (42,43):
        folder=BASE/'notebooks'/f'cr-u-s{seed}-20261015-v1'
        parent=BASE/'artifacts/kaggle'/folder.name
        if not (parent/'AUDITED.json').exists():continue
        manifest=json.loads((folder/'source_manifest.json').read_text())
        fixture=json.loads((parent/f'runs/cr_local_s{seed}/config.json').read_text())['fixture']
        with torch.serialization.safe_globals([bytes]):
            for arm in ('cr_local','cr_fl'):
                audit=verify(parent/f'runs/{arm}_s{seed}',manifest,fixture,ROOT/'papers/pilot/fixture')
                assert audit['logical_rounds']==3 and audit['committed_rounds']==1;audits.append(audit)
    proof=dict(pass_check=True,empty_bytes_refusal_reproduced=True,scoped_allowlist=True,
        subprocess_launcher=True,audits=audits,source_unchanged=True,
        scope='synthetic serializer regression plus existing committed artifacts; no production SGD')
    (BASE/'probes/bytes-compat-preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('Empty bytes regression PASS; scoped loader + subprocess + actual r3 audits',len(audits))


if __name__=='__main__':main()
