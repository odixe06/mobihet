"""Hybrid immutable inputs, private-online notebooks, bounded parent wall time."""
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
from scripts.kaggle.gen_v2 import AUTH, SETUP, RUN
from mh.v8_run import FAMILIES

DATASET = 'mobihet-v8-hybrid-20261012-src'
DOC = 'pilot-v8-hybrid.md'
PREFLIGHT = r'''
from mh.model_packet import Codec
for gpu in range(2):
    model=build_model(dict(CFG,n_features=66,num_classes=16),latent=True).to(f'cuda:{gpu}')
    codec=Codec(model,'v8:T4preflight')
    for half in (False,True):
        packet=codec.encode(model.state_dict(),0,1,half=half)
        decoded,base,version=codec.decode(packet,f'cuda:{gpu}',0)
        model.load_state_dict(decoded,strict=True)
        assert version==1
    print('T4 V8 coherentmodel/BN codec PASS',gpu,flush=True)
    del model; torch.cuda.empty_cache()
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--owner42', required=True)
    parser.add_argument('--owner43', required=True)
    args = parser.parse_args()
    base = ROOT / 'papers/v8'
    runtime = json.loads((ROOT / 'knowledge/runtime.json').read_text())
    key = netrc.netrc().authenticators('api.wandb.ai')[2]
    assert key and '**/private_push/' in (ROOT / '.gitignore').read_text().splitlines()
    code = [p.name for p in sorted((ROOT / 'mh').glob('*.py'))]
    manifest = {name: hashlib.sha256((ROOT / 'mh' / name).read_bytes()).hexdigest() for name in code}
    manifest['meta.json'] = hashlib.sha256((ROOT / 'knowledge/meta.json').read_bytes()).hexdigest()
    manifest[DOC] = hashlib.sha256((ROOT / 'docs/research' / DOC).read_bytes()).hexdigest()
    for owner, seed in ((args.owner42, 42), (args.owner43, 43)):
        rid = f'v8{args.family}-s{seed}-20261012-v1'
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
        runner = RUN.replace("'mh.v2_run'", "'mh.v8_run'")
        runner = runner.replace("'--max_hours','4'", "'--max_hours','2.2'")
        runner = runner.replace('>=3*3600', '>=3.7*3600')
        runner = runner.replace('        if proc.poll() is not None:',
            "        if proc.poll() is None and time.time()-T0>=4.7*3600: proc.terminate()\n"
            "        if proc.poll() is None and time.time()-T0>=4.75*3600: proc.kill()\n"
            '        if proc.poll() is not None:')
        # Preserve identity for unstarted arms so terminal collection can audit all slots.
        extra = r'''
for arm in METHODS:
    if arm in summary: continue
    folder=OUT/f'{arm}_s{SEED}'; folder.mkdir(exist_ok=True)
    cfg=dict(arm=arm,seed=SEED,private='dagsnet',rounds=20,batch=512,lr=.001,lr_min=.00001,
             clip=1.,eval_every=5,aux_batch=256,aux_weight=1.,temperature=2.,tau=.05,weight_cap=10.,smoke=False)
    identity=dict(config=cfg,source={name:manifest[name] for name in CODE},
                  fixture=hashlib.sha256((SRC/'manifest.json').read_bytes()).hexdigest())
    identity['fingerprint']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    (folder/'config.json').write_text(json.dumps(identity,indent=2))
    (folder/'log.txt').write_text('NOT STARTED: parent wall-time budget reached before this slot\n')
    summary[arm]=dict(rc=-998,complete=False)
'''
        runner = runner.replace("(OUT/f'summary_s{SEED}.json').write_text", extra + "\n(OUT/f'summary_s{SEED}.json').write_text")
        nb = nbf.v4.new_notebook(cells=[nbf.v4.new_markdown_cell(
            '# MobiHet V8 Gaussian-proxy hybrid\n'
            'Frozen pilot-v8-hybrid contract. All learned models DAGSNet. '
            'No raw/perturbed/latent/per-record gradients exported. FP32 statistics uplink, coherent paid proxy snapshot, '
            'actual FP16 Gaussian factors and FP32 coherent proxy downlink, cohort>=5 trusted emulator, no crypto/DP. '
            'Internal group/gp, r20, ten mean-client classification metrics. Stage U; R1/R2 not proven. '
            'W&B online authentication before training; private inline key owner-authorized.'),
            nbf.v4.new_code_cell(AUTH), nbf.v4.new_code_cell(f'CODE={code!r}\n'+setup+PREFLIGHT),
            nbf.v4.new_code_cell(f'SEED={seed}\nMETHODS={FAMILIES[args.family]!r}\n'+runner)])
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
