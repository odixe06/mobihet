"""Terminal source/weights provenance and independent wire/scalar-gate audit; no utility claim."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE/'source/proj'))
from rank_math import aggregate,decode_up,decode_down,gates


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(seed):
    rid=f'ccr-probe-s{seed}-20261015-v1';folder=BASE/'notebooks'/rid
    if not (folder/'PUSHED').exists():return None
    dest=BASE/'artifacts/kaggle'/rid;marker=dest/'AUDITED.json'
    if marker.exists():return json.loads(marker.read_text())
    meta=json.loads((folder/'kernel-metadata.json').read_text());helper=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
    prefix=[sys.executable,str(helper/'kaggle_as.py'),meta['id'].split('/')[0],'--']
    status=subprocess.run(prefix+['kaggle','kernels','status',meta['id']],check=True,capture_output=True,text=True)
    print(status.stdout.strip(),flush=True)
    if not any(f'KernelWorkerStatus.{s}' in status.stdout for s in ('COMPLETE','ERROR')):return None
    subprocess.run(prefix+[sys.executable,str(helper/'pull_kernel_output.py'),meta['id'],'-p',str(dest),'-j','4',
        '--pattern',r'^(source/|runs/)'],check=True)
    private=BASE/'private_push'/rid;executed=private/'executed.ipynb'
    subprocess.run(prefix+[sys.executable,str(helper/'download_executed_notebook.py'),meta['id'],'--version','1',
        '--output',str(executed)],check=True,umask=0o077);executed.chmod(0o600)
    reference=nbformat.read(folder/meta['code_file'],as_version=4);pushed=nbformat.read(private/meta['code_file'],as_version=4)
    key=next(n.value.value for n in ast.parse(pushed.cells[1].source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='_INLINE_WANDB_KEY' for t in n.targets))
    actual=nbformat.reads(executed.read_text().replace(key,'[REDACTED]'),as_version=4)
    assert len(actual.cells)==len(reference.cells)
    for i,cell in enumerate(actual.cells):
        if cell.cell_type=='code':assert cell.source==(pushed.cells[i].source.replace(key,'[REDACTED]') if i==1 else reference.cells[i].source)
    actual.cells[1].source=reference.cells[1].source;actual.metadata['credential_cell_redacted']=True
    nbformat.write(actual,folder/'executed.redacted.ipynb')
    manifest=json.loads((folder/'source_manifest.json').read_text())
    assert manifest==json.loads((dest/'source/source_manifest.json').read_text())
    for name,digest in manifest.items():assert sha(dest/'source'/name)==digest
    assert sha(BASE/'source/proj/rank_math.py')==manifest['rank_math.py'],'verifier math source drift'
    for name in ('input_manifest.json','protocol.json','warm-proof.json'):
        assert json.loads((folder/name).read_text())==json.loads((dest/'source'/name).read_text())
    protocol=json.loads((folder/'protocol.json').read_text());identity=hashlib.sha256((dest/'source/protocol.json').read_bytes()).digest()[:16]
    runs=dest/'runs';wires=[(runs/f'client_{k:02d}.bin').read_bytes() for k in range(20)]
    wire,cohort=aggregate(wires,identity);assert wire==(runs/'cohort.bin').read_bytes()
    assert cohort==json.loads((runs/'cohort.json').read_text())
    _,_,eligible,_=decode_down(wire,identity);clients=[]
    for k,up in enumerate(wires):
        n,_,_=decode_up(up,identity);support=eligible&(n>=32).all(0)
        client=json.loads((runs/f'client_{k:02d}_probe.json').read_text())
        assert client['client']==k and np.array_equal(client['support'],support)
        assert client['scan_rows']==int(n[0].sum()) and client['valid_scan_rows']==int(n[0,eligible].sum())
        for c,item in enumerate(client['classes']):
            assert (item is not None)==bool(support[c])
            if item is not None:
                assert item['rows']==(min(256,int(n[0,c]))+1)//2
                assert all(np.isfinite(item[v]) and item[v]>=0 for v in ('own_energy','delta_energy','residual_energy'))
        assert client['projection']['columns']==32
        clients.append(client)
    computed=gates(clients,eligible);result=json.loads((runs/'probe_result.json').read_text())
    for name,value in computed.items():assert result[name]==value
    assert result['seed']==seed and result['protocol']==protocol and not result['model_training'] and not result['holdout_read']
    inputs=json.loads((folder/'input_manifest.json').read_text())
    assert result['weights_sha']==inputs['warm_weights.pt'] and result['original_fixture_sha']==inputs['manifest.json']
    assert not any(n.startswith(('group_','old_','test_')) for n in inputs)
    record=dict(kernel=meta['id'],version=1,source=manifest,result=result,
        evidence='exact executed/source/input/warm-weight provenance; independent moment-wire and scalar-gate reconstruction; no independent full model scan/gradient recomputation',utility_measured=False)
    marker.write_text(json.dumps(record,indent=2)+'\n');print('AUDITED CrossClientRank',seed,'probe',result['pass_probe'],flush=True)
    return record


if __name__=='__main__':
    rows=[collect(seed) for seed in (42,43)]
    if all(rows):
        passed=all(r['result']['pass_probe'] for r in rows)
        result=dict(pass_probe=passed,results={str(s):r['result'] for s,r in zip((42,43),rows)},utility_measured=False,
            decision='probe PASS; utility protocol/controls still require owner decision' if passed else 'stop; do not open utility or retune')
        (BASE/'probes/paired-gate.json').write_text(json.dumps(result,indent=2)+'\n')
        print('PAIRED CrossClientRank',passed,'utility not measured')
