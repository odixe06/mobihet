"""Audit frozen-source replay and aggregate projection diagnostics; never change original gate."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import nbformat
import numpy as np

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]


def collect(seed):
    rid=f'ccr-diag-s{seed}-20261015-v1';folder=BASE/'notebooks'/rid;dest=BASE/'artifacts/kaggle'/rid
    if not (folder/'PUSHED').exists():return None
    marker=dest/'AUDITED.json'
    if marker.exists():return json.loads(marker.read_text())
    meta=json.loads((folder/'kernel-metadata.json').read_text());helper=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
    prefix=[sys.executable,str(helper/'kaggle_as.py'),meta['id'].split('/')[0],'--']
    status=subprocess.run(prefix+['kaggle','kernels','status',meta['id']],check=True,capture_output=True,text=True)
    print(status.stdout.strip(),flush=True)
    if 'KernelWorkerStatus.COMPLETE' not in status.stdout:return None
    subprocess.run(prefix+[sys.executable,str(helper/'pull_kernel_output.py'),meta['id'],'-p',str(dest),'-j','4',
        '--pattern',r'^(source/|runs/)'],check=True)
    private=BASE/'private_push'/rid;executed=private/'executed.ipynb'
    subprocess.run(prefix+[sys.executable,str(helper/'download_executed_notebook.py'),meta['id'],'--version','1','--output',str(executed)],check=True,umask=0o077)
    executed.chmod(0o600)
    reference=nbformat.read(folder/meta['code_file'],as_version=4);pushed=nbformat.read(private/meta['code_file'],as_version=4)
    key=next(n.value.value for n in ast.parse(pushed.cells[1].source).body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='_INLINE_WANDB_KEY' for t in n.targets))
    actual=nbformat.reads(executed.read_text().replace(key,'[REDACTED]'),as_version=4)
    assert len(actual.cells)==len(reference.cells)
    for i,c in enumerate(actual.cells):
        if c.cell_type=='code':assert c.source==pushed.cells[i].source.replace(key,'[REDACTED]')
    actual.cells[1].source=reference.cells[1].source;actual.metadata['credential_cell_redacted']=True
    nbformat.write(actual,folder/'executed.redacted.ipynb')
    manifest=json.loads((folder/'source_manifest.json').read_text())
    assert manifest==json.loads((dest/'source/source_manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((dest/'source'/name).read_bytes()).hexdigest()==digest
    spec=json.loads((folder/'diagnostic-manifest.json').read_text())
    assert spec==json.loads((dest/'source/diagnostic_manifest.json').read_text())
    assert hashlib.sha256((dest/'source/rank_projection_diagnostic.py').read_bytes()).hexdigest()==spec['sha256']
    nodes=ast.parse(reference.cells[2].source).body
    replay=next(ast.literal_eval(n.value) for n in nodes if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='REPLAY' for t in n.targets))
    assert replay==json.loads((dest/'runs/replay.json').read_text())
    parent=BASE/'artifacts/kaggle'/f'ccr-probe-s{seed}-20261015-v1'
    assert hashlib.sha256((parent/'AUDITED.json').read_bytes()).hexdigest()==spec['audit_sha']
    assert bytes.fromhex(replay['cohort'])==(parent/'runs/cohort.bin').read_bytes()
    rows=[json.loads((dest/f'runs/diagnostic_{k:02d}.json').read_text()) for k in range(20)]
    for k,row in enumerate(rows):
        assert row['client']==k and row['seed']==seed and row['original_energies_reproduced']
        assert bytes.fromhex(replay['up'][k])==(parent/f'runs/client_{k:02d}.bin').read_bytes()
        singular=np.asarray(row['singular_values']);assert np.isfinite(singular).all() and np.all(singular[:-1]>=singular[1:])
        assert row['rank']==int((singular>singular[0]*1e-10).sum())
        assert len(row['coefficients_original'])==len(row['fit_scale'])==32
        for c,item in enumerate(row['classes']):
            old=replay['clients'][k]['classes'][c];assert (old is None)==(item is None)
            if item is not None:
                assert item['test']['rows']==old['rows']
                for new,oldkey in [('delta','delta_energy'),('residual','residual_energy')]:
                    assert np.isclose(item['test'][new],old[oldkey],rtol=1e-6,atol=1e-18)
        for side in ('fit','test'):
            valid=[c[side] for c in row['classes'] if c is not None];n=sum(c['rows'] for c in valid)
            assert n==row[side]['rows']
            for key_ in ('delta','predicted','residual'):
                assert np.isclose(sum(c[key_]*c['rows'] for c in valid)/n,row[side][key_],rtol=1e-10,atol=1e-18)
    result=json.loads((dest/'runs/diagnostic_result.json').read_text())
    assert result['models']==20 and result['all_original_energies_reproduced']
    assert not result['model_training'] and not result['holdout_read'] and not result['gate_changed']
    record=dict(kernel=meta['id'],version=1,result=result,diagnostic=spec,
        max_condition=max(r['condition'] for r in rows if r['condition'] is not None),
        min_rank=min(r['rank'] for r in rows),max_abs_coefficient=max(abs(x) for r in rows for x in r['coefficients_original']),
        evidence='Exact executed/frozen source; original energies reproduced on T4; scalar/SVD/class accounting audit; no independent local full-gradient inference')
    marker.write_text(json.dumps(record,indent=2)+'\n');print('AUDITED projection diagnostic',seed,record['max_condition'],flush=True)
    return record


if __name__=='__main__':
    rows=[collect(s) for s in (42,43)]
    if all(rows):(BASE/'probes/paired-projection-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
