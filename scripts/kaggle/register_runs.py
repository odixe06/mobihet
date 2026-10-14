"""Freeze expected W&B identities from the exact public pilot notebook and manifests."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--build',required=True)
    args = parser.parse_args(); base = Path(__file__).resolve().parents[2] / 'papers' / args.build
    records = []
    for folder in sorted((base/'notebook').iterdir()):
        meta = json.loads((folder/'kernel-metadata.json').read_text())
        manifest = json.loads((folder/'source_manifest.json').read_text())
        notebook = json.loads((folder/meta['code_file']).read_text())
        cells = ''.join(notebook['cells'][-1]['source'])
        assignments = ast.parse(cells).body[:2]
        seed, methods = [ast.literal_eval(node.value) for node in assignments]
        for arm in methods:
            cfg = dict(arm=arm,seed=seed,private='dagsnet',rounds=20,batch=512,lr=.001,lr_min=.00001,
                       clip=1.,eval_every=5,aux_batch=256,aux_weight=1.,temperature=2.,tau=.05,weight_cap=10.,smoke=False)
            identity = dict(config=cfg,source={name:sha for name,sha in manifest.items() if name.endswith('.py')},
                            fixture=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest())
            fp = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            rid = f'{arm}_s{seed}-{fp[:8]}'
            records.append(dict(family=folder.name[len(args.build)],seed=seed,arm=arm,run_id=rid,
                                fingerprint=fp,kernel=meta['id'],locator=f'21522798-uit/mobihet-{args.build}/{rid}'))
    (base/'wandb-runs.json').write_text(json.dumps(records,indent=2)+'\n')
    print(args.build,len(records),'exact W&B identities')


if __name__ == '__main__':
    main()
