"""Teacher immutable inputs, private-online notebooks, bounded parent wall time."""
import argparse
import hashlib
import json
import netrc
import os
import shutil
import sys
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.kaggle.gen_v2 import AUTH, SETUP
from scripts.kaggle.teacher_controller import COST_RUN

DATASET = None
DOC = None
PREFLIGHT = r'''
from mh.model_packet import Codec
for gpu in range(2):
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(f'cuda:{gpu}')
    codec=Codec(model,'teacher:T4preflight')
    for half in (False,True):
        packet=codec.encode(model.state_dict(),0,1,half=half)
        decoded,base,version=codec.decode(packet,f'cuda:{gpu}',0)
        model.load_state_dict(decoded,strict=True)
        assert version==1
    print('T4 teacher coherentmodel/BN codec PASS',gpu,flush=True)
    del model; torch.cuda.empty_cache()
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--build',choices=['v9','v10'],required=True)
    parser.add_argument('--family',required=True)
    parser.add_argument('--owner42', required=True)
    parser.add_argument('--owner43', required=True)
    args = parser.parse_args()
    global DATASET,DOC
    if args.build=='v9':
        from mh.v9_run import FAMILIES
        DOC='pilot-v9-residual.md'
    else:
        from mh.v10_run import FAMILIES
        DOC='pilot-v10-energy.md'
    assert args.family in FAMILIES
    DATASET=f'mobihet-{args.build}-teacher-20261012-src'
    base=ROOT/'papers'/args.build
    runtime = json.loads((ROOT / 'knowledge/runtime.json').read_text())
    key = netrc.netrc().authenticators('api.wandb.ai')[2]
    assert key and '**/private_push/' in (ROOT / '.gitignore').read_text().splitlines()
    code = [p.name for p in sorted((ROOT / 'mh').glob('*.py'))]
    manifest = {name: hashlib.sha256((ROOT / 'mh' / name).read_bytes()).hexdigest() for name in code}
    manifest['meta.json'] = hashlib.sha256((ROOT / 'knowledge/meta.json').read_bytes()).hexdigest()
    manifest[DOC] = hashlib.sha256((ROOT / 'docs/research' / DOC).read_bytes()).hexdigest()
    for owner, seed in ((args.owner42, 42), (args.owner43, 43)):
        rid = f'{args.build}{args.family}-s{seed}-20261012-v1'
        public, private = base / 'notebook' / rid, base / 'private_push' / rid
        assert not (public / 'PUSHED').exists() and not (private / 'PUSHED').exists()
        source = base / 'input' / owner
        if (source / 'source_manifest.json').exists():
            assert json.loads((source / 'source_manifest.json').read_text()) == manifest
        else:
            source.mkdir(parents=True, exist_ok=True)
            for name in code:
                shutil.copyfile(ROOT / 'mh' / name, source / name)
            shutil.copyfile(ROOT / 'knowledge/meta.json', source / 'meta.json')
            shutil.copyfile(ROOT / 'docs/research' / DOC, source / DOC)
            for path in (ROOT / 'papers/pilot/fixture').iterdir():
                shutil.copyfile(path, source / path.name)
            (source / 'source_manifest.json').write_text(json.dumps(manifest, indent=2))
            (source / 'dataset-metadata.json').write_text(json.dumps(dict(
                id=f'{owner}/{DATASET}', title=DATASET, licenses=[{'name': 'CC0-1.0'}])))
        public.mkdir(parents=True, exist_ok=True)
        private.mkdir(parents=True, exist_ok=True)
        for name in ('source_manifest.json', 'manifest.json'):
            shutil.copyfile(source / name, public / name)
        setup = SETUP.replace("elif f.name=='pilot-v2.md'", f"elif f.name=='{DOC}'")
        runner=COST_RUN
        nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
            '# MobiHet teacher-first residual / common-base energy\n'
            'Frozen teacher-first contract; measured 2-round cost gate then same-schedule resume. All learned models DAGSNet. '
            'No raw/perturbed/latent/per-record gradients exported. FP32 statistics uplink, coherent paid proxy snapshot, '
            'actual decoded Gaussian and FP16 coherent teacher-state downlink, cohort>=5 trusted emulator, no crypto/DP. '
            'Teacher internal group/head, r20, ten mean-client classification metrics. Stage U; R1/R2 not proven. '
            'W&B online authentication before training; private inline key owner-authorized.'),
            nbf.v4.new_code_cell(AUTH), nbf.v4.new_code_cell(f'CODE={code!r}\n'+setup+PREFLIGHT),
            nbf.v4.new_code_cell(f'SEED={seed}\nMETHODS={FAMILIES[args.family]!r}\nMODULE={str("mh."+args.build+"_run")!r}\n'+runner)])
        nb.metadata.update(kernelspec={'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
            language_info={'name': 'python', 'version': '3.12'},
            kaggle={'accelerator': 'nvidiaTeslaT4', 'isGpuEnabled': True,
                    'isInternetEnabled': True, 'language': 'python', 'sourceType': 'notebook'})
        filename = rid + '.ipynb'
        nbf.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == 'code':
                compile(cell.source, rid, 'exec')
        nbf.write(nb, public / filename)
        metadata = dict(id=f'{owner}/mobihet-{rid}', title=f'mobihet-{rid}', code_file=filename,
            language='python', kernel_type='notebook', is_private=True, enable_gpu=True, enable_internet=True,
            machine_shape=runtime['machine_shape'], docker_image=runtime['docker_image'],
            dataset_sources=[f'{owner}/{DATASET}'], kernel_sources=[], competition_sources=[])
        assert len(metadata['title']) <= 50
        (public / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
        credential = nbf.reads(nbf.writes(nb), as_version=4)
        credential.cells[1].source = credential.cells[1].source.replace(
            '_INLINE_WANDB_KEY = ""', '_INLINE_WANDB_KEY = '+json.dumps(key))
        nbf.write(credential, private / filename)
        (private / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
        for name in (filename, 'kernel-metadata.json'):
            os.chmod(private / name, 0o600)
        print(metadata['id'], 'prepared; not pushed')


if __name__ == '__main__':
    main()
