"""New immutable tags for verified same-owner checkpoint-dataset continuations."""
import json
import netrc
import os
from pathlib import Path
import subprocess

import nbformat as nbf

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2])


def main():
    base=(ROOT/'.compat/papers/hybrid-full').resolve()
    key=netrc.netrc().authenticators('api.wandb.ai')[2];assert key
    for index in (1,3):
        parent=base/'notebook'/f'hybf{index}-s42-20261014-cont01-v1'
        metadata=json.loads((parent/'kernel-metadata.json').read_text())
        bridge=json.loads((ROOT/f'docs/archive/operations/hybrid-checkpoint{index}-2026-10-14.json').read_text())
        assert bridge['uploaded'] and bridge['verified_full_parent']
        rid=f'hybf{index}-s42-20261014-dsresume-v1'
        public=base/'notebook'/rid;private=base/'private_push'/rid
        assert not (public/'PUSHED').exists() and not (private/'PUSHED').exists()
        assert not (private/(rid+'.ipynb')).exists(), 'private source already generated; inspect instead of overwriting'
        nb=nbf.read(parent/metadata['code_file'],as_version=4)
        nb.cells[0].source+='\nCheckpoint attachment is an audited private same-owner dataset; ERROR parent rejected as kernel source.'
        meta=dict(metadata,id=metadata['id'].split('/')[0]+'/mobihet-'+rid,title='mobihet-'+rid,
            code_file=rid+'.ipynb',kernel_sources=[],dataset_sources=metadata['dataset_sources']+[bridge['dataset']])
        assert len(meta['title'])<=50 and meta['is_private']
        nbf.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code':compile(cell.source,rid,'exec')
        public.mkdir(parents=True,exist_ok=True);private.mkdir(parents=True,mode=0o700,exist_ok=True)
        nbf.write(nb,public/meta['code_file'])
        (public/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
        for name in ('source_manifest.json','resume-plan.json'):(public/name).write_bytes((parent/name).read_bytes())
        (public/'checkpoint-dataset.json').write_text(json.dumps(bridge,indent=2)+'\n')
        subprocess.run(['git','check-ignore','-q',str((private/meta['code_file']).resolve())],cwd=ROOT,check=True)
        credential=nbf.reads(nbf.writes(nb),as_version=4)
        credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
        for name,body in [(meta['code_file'],nbf.writes(credential)),('kernel-metadata.json',json.dumps(meta,indent=2))]:
            fd=os.open(private/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            with os.fdopen(fd,'w') as file:file.write(body)
        print(meta['id'],'private checkpoint dataset',bridge['dataset'],'prepared; not pushed',flush=True)


if __name__=='__main__':main()
