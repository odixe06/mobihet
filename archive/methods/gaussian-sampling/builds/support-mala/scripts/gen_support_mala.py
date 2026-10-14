"""Frozen V4 MALA/support input and owner-private online notebooks."""
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
from mh.v4_run import FAMILIES
DATASET='mobihet-v4-20261012-src'
DOCS=['pilot-v4-mala.md','pilot-v4-support.md']
PREFLIGHT=r'''
from types import SimpleNamespace
from mh.v4_mala import potential,transition,accept
from mh.v4_support import Method as Support
for gpu in range(2):
    device=torch.device(f'cuda:{gpu}'); torch.cuda.set_device(device)
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(device).eval()
    g=torch.Generator(device=device).manual_seed(4701)
    z=torch.randn(16,66,device=device,generator=g); labels=torch.arange(16,device=device)
    mu=torch.zeros(16,66,device=device); sd=torch.ones_like(mu)
    lp,score=potential(model,z,labels,mu,sd)
    proposed=transition(z,lp,score,g); nlp,ngrad=potential(model,proposed,labels,mu,sd)
    next_z,_,_,taken=accept(z,proposed,lp,score,nlp,ngrad,g)
    assert torch.isfinite(next_z).all() and all(p.requires_grad for p in model.parameters())
    x=torch.from_numpy(np.load(SRC/'train_0_X.npy')[:32]).to(device)
    y=torch.from_numpy(np.load(SRC/'train_0_y.npy')[:32]).to(device).long()
    run=SimpleNamespace(dev=device,K=5,X=[x]*5,Y=[y]*5,n=[32]*5,bytes=[],init=model,
                        a=SimpleNamespace(arm='s_structural'))
    support=Support(run); sx=support.draw_support(labels,g,0)
    assert torch.isfinite(sx).all() and len(support.support_packet)==10592
    print('T4 V4 score/Metropolis/support codec preflight PASS',gpu,flush=True)
    del model,z,labels,mu,sd,lp,score,proposed,nlp,ngrad,next_z,taken,x,y,run,support,sx
    torch.cuda.empty_cache()
torch.cuda.set_device(0)
'''


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--family',choices=FAMILIES,required=True)
    parser.add_argument('--owner42',required=True); parser.add_argument('--owner43',required=True); args=parser.parse_args()
    base=ROOT/'papers/v4'; runtime=json.loads((ROOT/'knowledge/runtime.json').read_text())
    key=netrc.netrc().authenticators('api.wandb.ai')[2]; assert key
    assert '**/private_push/' in (ROOT/'.gitignore').read_text().splitlines()
    code=[p.name for p in sorted((ROOT/'mh').glob('*.py'))]
    manifest={name:hashlib.sha256((ROOT/'mh'/name).read_bytes()).hexdigest() for name in code}
    manifest['meta.json']=hashlib.sha256((ROOT/'knowledge/meta.json').read_bytes()).hexdigest()
    manifest.update({name:hashlib.sha256((ROOT/'docs/research'/name).read_bytes()).hexdigest() for name in DOCS})
    for owner,seed in ((args.owner42,42),(args.owner43,43)):
        rid=f'v4{args.family}-s{seed}-20261012-v1'; public=base/'notebook'/rid; private=base/'private_push'/rid
        assert not (public/'PUSHED').exists() and not (private/'PUSHED').exists(),'never edit pushed notebook'
        src=base/'input'/owner
        if (src/'source_manifest.json').exists():
            assert json.loads((src/'source_manifest.json').read_text())==manifest,'new source requires new dataset/attempt'
        else:
            src.mkdir(parents=True,exist_ok=True)
            for name in code: shutil.copyfile(ROOT/'mh'/name,src/name)
            shutil.copyfile(ROOT/'knowledge/meta.json',src/'meta.json')
            for name in DOCS: shutil.copyfile(ROOT/'docs/research'/name,src/name)
            for file in (ROOT/'papers/pilot/fixture').iterdir(): shutil.copyfile(file,src/file.name)
            (src/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
            (src/'dataset-metadata.json').write_text(json.dumps(dict(id=f'{owner}/{DATASET}',title=DATASET,licenses=[{'name':'CC0-1.0'}])))
        public.mkdir(parents=True,exist_ok=True); private.mkdir(parents=True,exist_ok=True)
        for name in ('source_manifest.json','manifest.json'): shutil.copyfile(src/name,public/name)
        setup=SETUP.replace("elif f.name=='pilot-v2.md'", "elif f.name.startswith('pilot-v4-') and f.suffix=='.md'")
        runner=RUN.replace("'mh.v2_run'","'mh.v4_run'")
        nb=nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
            '# MobiHet V4 '+args.family.upper()+'\n'
            'Frozen pilot-v4-mala/support contracts in input. All learned models DAGSNet. '
            'No raw/perturbed/latent/per-record gradients exported; trusted cohort sum >=5, not crypto/DP. '
            'Stage U only; internal group/old, r20 private/gp; R1/R2 not proven. '
            'W&B authenticated online before train, private inline owner-authorized fallback. '
            'Ten classification metrics and per-class F1 from mean client confusion, not pooled confusion.'),
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
        print(meta['id'],'prepared, not pushed')


if __name__=='__main__': main()
