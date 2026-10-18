"""Read exact full-Hybrid W&B identities; source/data checked against audited cache."""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path

import wandb

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
BASE = ROOT / '.compat/papers/hybrid-full'


def main():
    api = wandb.Api(timeout=25)
    entity = api.default_entity
    prior = ROOT / '.compat/papers/full-validation/kaggle/full1-s42-20261013-cont01-v1/source'
    # Both frozen preparations write the same cache manifest. Do not guess a latest run.
    assert (BASE / 'proj/full_prepare.py').read_bytes() == (prior / 'full_prepare.py').read_bytes()
    data = json.loads((prior / 'data_manifest.json').read_text())
    fixture = hashlib.sha256((json.dumps(data, indent=2) + '\n').encode()).hexdigest()
    observations = []
    for folder in sorted((BASE / 'notebook').iterdir()):
        if not (folder / 'PUSHED').exists():
            continue
        manifest = json.loads((folder / 'source_manifest.json').read_text())
        project = {p.name for p in (BASE / 'proj').glob('*.py')}
        source = {('proj/' if name in project else 'mh/') + name: sha
                  for name, sha in manifest.items() if name.endswith('.py')}
        import nbformat
        meta = json.loads((folder / 'kernel-metadata.json').read_text())
        notebook = nbformat.read(folder / meta['code_file'], as_version=4)
        import ast
        arms = ast.literal_eval(ast.parse(notebook.cells[3].source).body[0].value)
        for arm in arms:
            config = dict(arm=arm, seed=42, rounds=50, batch=512, private='dagsnet',
                          lr=.001, lr_min=.00001, clip=1., eval_every=1, aux_batch=256,
                          aux_weight=1., temperature=2., tau=.05, weight_cap=10.,
                          smoke=False, backend='compiled')
            fp = hashlib.sha256(json.dumps(dict(config=config, source=source, fixture=fixture),
                                           sort_keys=True).encode()).hexdigest()
            observations.append(dict(arm=arm, kernel=meta['id'], fingerprint=fp,
                                     locator=f'{entity}/mobihet-hybrid-full/{arm}_s42-{fp[:8]}'))

    def read(item):
        try:
            run = api.run(item['locator'])
        except wandb.errors.CommError:
            return dict(item, state='NOT_VISIBLE_YET')
        assert run.config['fingerprint'] == item['fingerprint']
        keys = ('round', 'loss', 'lr', 'seconds', 'applied', 'skipped', 'preflight/online',
                'progress/round_in_flight', 'progress/client', 'progress/applied', 'progress/skipped',
                'progress/loss', 'progress/grad_norm', 'progress/max_logit', 'progress/softmax_entropy',
                'progress/real_ce', 'progress/kd', 'progress/synth_ce', '_timestamp',
                'private/test/gp/f1_macro', 'private/test/gp/accuracy')
        return dict(item, state=run.state, url=run.url,
                    summary={key: run.summary.get(key) for key in keys})

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(read, observations))
    record = dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), runs=rows,
                  evidence='read-only startup snapshot; not terminal scientific audit')
    (BASE / 'startup-health-latest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
