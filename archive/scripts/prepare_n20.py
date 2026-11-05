"""Generate eight authorized N20 notebooks at pushed HEAD; preserve M scientific source bytes."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SKILL=ROOT/'.agents/skills/kaggle-training-notebook/scripts'


def main():
    out=ROOT/'runs/n-study';out.mkdir(exist_ok=True)
    assert not (out/'batch.json').exists(),'Frozen batch exists; inspect before regenerate'
    quota=json.loads((out/'quota.json').read_text())
    accounts=[x['user'] for x in quota['accounts']]
    assert len(accounts)==8 and all(x['gpu_left']>=6.9 and x['gpu_reserved']==0 for x in quota['accounts'])
    assert '54 passed' in (out/'tests.log').read_text()
    smoke=json.loads((out/'smoke-audit.json').read_text());assert smoke['r1_four_cells_bitexact'] and len(smoke['cells'])==4
    sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
    assert subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],text=True,cwd=ROOT).split()[0]==sha
    parent=json.loads((ROOT/'evidence/m20-summary/batch.json').read_text())[2]
    overlay=parent['overlay_sha256']
    for name,digest in overlay.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    common=[v for v in parent['hp'] if not v.startswith(('real_kd=','seed='))]
    common+=['q_collect=online','q_agg=count','kd_support=off']
    result=[]
    for cell,(real,target) in enumerate([(False,'soft'),(True,'soft'),(False,'fallback'),(True,'fallback')]):
        for offset,seed in enumerate([42,43]):
            i=cell*2+offset;slug=f'mh-n{i}-'+('gate' if real else 'noreal')+f'-{target}-s{seed}'
            hp=common+[f'real_kd={str(real).lower()}',f'real_kd_gate={str(real).lower()}',f'synth_target={target}',f'seed={seed}']
            account=accounts[i]
            subprocess.run([sys.executable,str(ROOT/'scripts/gen_notebook.py'),'--owner',account,'--slug',slug,
                '--method','mobihet','--width','1','--rounds','50','--until','20','--max-hours','6',
                '--sha',sha,'--overlay',*overlay,'--set',*hp],check=True,cwd=ROOT,capture_output=True)
            folder=ROOT/'notebooks'/slug;nb=folder/(slug+'.ipynb');meta=folder/'kernel-metadata.json'
            subprocess.run([sys.executable,str(SKILL/'embed_wandb_key.py'),str(nb),'--metadata',str(meta)],check=True,capture_output=True)
            document=json.loads(nb.read_text());metadata=json.loads(meta.read_text())
            assert metadata['is_private'] is True and metadata['machine_shape']=='NvidiaTeslaT4'
            assert document['metadata']['kaggle']['isGpuEnabled']
            for block in document['cells']:
                if block['cell_type']=='code':compile(''.join(block['source']),str(nb),'exec')
            result.append({'arm':f'N{i}','cell':'ABCD'[cell],'account':account,'slug':slug,'version':1,'hp':hp,
                'base_sha':sha,'source_sha256':overlay['methods/mobihet.py'],'overlay_sha256':overlay,
                'cap_hours':6,'endpoint':20,'lr_schedule_rounds':50,'notebook_sha256':hashlib.sha256(nb.read_bytes()).hexdigest()})
    target=out/'batch.json'
    assert not target.exists(),'Frozen batch exists; inspect before regenerate'
    target.write_text(json.dumps(result,indent=2)+'\n')
    print('N20_NOTEBOOKS_LOCKED',hashlib.sha256(target.read_bytes()).hexdigest())


if __name__=='__main__':main()
