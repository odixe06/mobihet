"""Immutable V3 source dataset and online private notebooks; no changes to pushed V2."""
import argparse
import hashlib
import json
import netrc
import os
import shutil
import sys
from pathlib import Path
import nbformat as nbf
ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[2]); sys.path.insert(0,str(ROOT))
from scripts.kaggle.gen_v2 import AUTH,SETUP,RUN
from mh.v3_math import FAMILIES

DATASET='mobihet-v3-20261011-src'
PREFLIGHT=r'''
from mh.v3_math import basis,matrix,unpad,orthogonal,coordinates
for gpu in range(2):
    device=torch.device(f'cuda:{gpu}'); torch.cuda.set_device(device)
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(device).train()
    x=torch.from_numpy(np.load(SRC/'train_0_X.npy')[:32]).to(device).float()
    y=torch.from_numpy(np.load(SRC/'train_0_y.npy')[:32]).to(device).long()
    with torch.autocast('cuda',dtype=torch.float16): loss=torch.nn.functional.cross_entropy(model(x)[0].float(),y)
    loss.backward(); layout=Layout(model)
    gradient=torch.cat([p.grad.reshape(-1) for p in model.parameters()]); assert torch.isfinite(gradient).all()
    for rank in (1,8,32):
        q=basis(rank,1,device); a=matrix(gradient); coeff=a@q
        left=orthogonal(coeff); reconstructed=unpad(left@(a.T@left).T)
        assert (gradient-reconstructed).norm()<=gradient.norm()*(1+1e-5)+2e-5
    for k in (512,24576):
        idx=coordinates(layout,k,1,True).to(device); assert len(torch.unique(idx))==k
    print('T4 V3 production-rank projection preflight PASS',gpu,flush=True)
    del model,x,y,gradient,a,q,coeff,left,reconstructed,layout,idx
    torch.cuda.empty_cache()
'''


def main():
    p=argparse.ArgumentParser(); p.add_argument('--family',choices=FAMILIES,required=True)
    p.add_argument('--owner42',required=True); p.add_argument('--owner43',required=True); args=p.parse_args()
    base=ROOT/'papers/v3'; runtime=json.loads((ROOT/'knowledge/runtime.json').read_text())
    key=netrc.netrc().authenticators('api.wandb.ai')[2]; assert key
    assert '**/private_push/' in (ROOT/'.gitignore').read_text().splitlines()
    code=[f.name for f in sorted((ROOT/'mh').glob('*.py'))]
    manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted((ROOT/'mh').glob('*.py'))}
    manifest.update({'meta.json':hashlib.sha256((ROOT/'knowledge/meta.json').read_bytes()).hexdigest(),
                     'pilot-v3.md':hashlib.sha256((ROOT/'docs/research/pilot-v3.md').read_bytes()).hexdigest()})
    for owner,seed in ((args.owner42,42),(args.owner43,43)):
        rid=f'v3{args.family}-s{seed}-20261011-v1'; public=base/'notebook'/rid; private=base/'private_push'/rid
        assert not (public/'PUSHED').exists() and not (private/'PUSHED').exists(),'never edit pushed notebook'
        src=base/'input'/owner
        if (src/'source_manifest.json').exists():
            assert json.loads((src/'source_manifest.json').read_text())==manifest,'new source requires new dataset/attempt'
        else:
            src.mkdir(parents=True,exist_ok=True)
            for name in code: shutil.copyfile(ROOT/'mh'/name,src/name)
            shutil.copyfile(ROOT/'knowledge/meta.json',src/'meta.json')
            shutil.copyfile(ROOT/'docs/research/pilot-v3.md',src/'pilot-v3.md')
            for f in (ROOT/'papers/pilot/fixture').iterdir(): shutil.copyfile(f,src/f.name)
            (src/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
            (src/'dataset-metadata.json').write_text(json.dumps(dict(id=f'{owner}/{DATASET}',title=DATASET,licenses=[{'name':'CC0-1.0'}])))
        public.mkdir(parents=True,exist_ok=True); private.mkdir(parents=True,exist_ok=True)
        for name in ('source_manifest.json','manifest.json'): shutil.copyfile(src/name,public/name)
        setup=SETUP.replace("elif f.name=='pilot-v2.md'","elif f.name=='pilot-v3.md'")
        runner=RUN.replace("'mh.v2_run'","'mh.v3_run'")
        nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
            '# MobiHet V3 '+args.family.upper()+' — private CE+KD\n'
            'Protocol/gates: immutable input pilot-v3.md. All learned models DAGSNet. '
            'No raw/perturbed/latent/per-record gradient export. Trusted cohort sum >=5; not crypto/DP. '
            'Stage U, internal group/old only. R20 headline private/gp; R1/R2 not proven. '
            'Online W&B required before training; private inline owner-authorized fallback.\n\n'
            'Ten metrics from per-client confusion, mean across clients: accuracy; precision_macro/micro/weighted; '
            'recall_macro/micro/weighted; f1_macro/micro/weighted. Single-label accuracy=micro P/R/F1=weighted recall.'),
            nbf.v4.new_code_cell(AUTH),nbf.v4.new_code_cell(f'CODE={code!r}\n'+setup+PREFLIGHT),
            nbf.v4.new_code_cell(f'SEED={seed}\nMETHODS={FAMILIES[args.family]!r}\n'+runner)])
        nb.metadata.update(kernelspec={'name':'python3','display_name':'Python 3','language':'python'},
            language_info={'name':'python','version':'3.12'},kaggle={'accelerator':'nvidiaTeslaT4',
            'isGpuEnabled':True,'isInternetEnabled':True,'language':'python','sourceType':'notebook'})
        filename=rid+'.ipynb'; nbf.validate(nb)
        for cell in nb.cells:
            if cell.cell_type=='code': compile(cell.source,rid,'exec')
        nbf.write(nb,public/filename)
        meta=dict(id=f'{owner}/mobihet-{rid}',title=f'mobihet-{rid}',code_file=filename,language='python',
            kernel_type='notebook',is_private=True,enable_gpu=True,enable_internet=True,
            machine_shape=runtime['machine_shape'],docker_image=runtime['docker_image'],
            dataset_sources=[f'{owner}/{DATASET}'],kernel_sources=[],competition_sources=[])
        assert len(meta['title'])<=50
        (public/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
        credential=nbf.reads(nbf.writes(nb),as_version=4)
        credential.cells[1].source=credential.cells[1].source.replace('_INLINE_WANDB_KEY = ""','_INLINE_WANDB_KEY = '+json.dumps(key))
        nbf.write(credential,private/filename); os.chmod(private/filename,0o600)
        (private/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)); os.chmod(private/'kernel-metadata.json',0o600)
        print(meta['id'],'prepared; not pushed')


if __name__=='__main__': main()
