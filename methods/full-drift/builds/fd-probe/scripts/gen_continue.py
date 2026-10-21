"""Prepare a continuation and private checkpoint bridge from the last audited committed round.

Preparation only. Upload/verify inputs, then push after the parent's evaluation kernels show progress.
    python gen_continue.py --parent fd-s42za-20261015-r24-v1 --owner dinhan25 --session za02
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import gen_fd


def main():
    p = argparse.ArgumentParser(); p.add_argument('--parent', required=True); p.add_argument('--owner', required=True)
    p.add_argument('--session', required=True, help='new immutable session code, e.g. za02'); a = p.parse_args()
    assert a.session.isalnum() and len(a.session) <= 8
    base = gen_fd.BASE; parent = base / 'artifacts/kaggle' / a.parent
    audit = json.loads((parent / 'AUDITED.json').read_text()); assert len(audit['audits']) == 1
    run = parent / 'runs' / audit['audits'][0]['run']
    cfg = json.loads((run / 'config.json').read_text())['config']
    assert cfg['final'] and cfg['defer_eval'] and cfg['seed'] == 42, 'seed42 final pipeline only'
    rows = sorted((run / 'complete').glob('round_*.json')); assert rows
    last = json.loads(rows[-1].read_text())['round']; assert last < 50, 'training already reached round 50'
    assert last in audit['audits'][0]['kept_weight_rounds'], 'parent round not verified'
    clients = cfg.get('clients', 20)
    metadata = json.loads((base / 'notebooks' / a.parent / 'kernel-metadata.json').read_text())
    tag = metadata['dataset_sources'][0].split('mobihet-fulldrift-src-')[1]
    round_tag = tag.removeprefix(gen_fd.TAG + '-')
    rid = f'fd-s42{a.session}-{tag}-v1'; dataset = f'{a.owner}/mh-parent-{rid}'
    queues = dict(workers=2, arms=[cfg['arm'] + ':50'], fast=cfg.get('fast', False), final=True, clients=clients,
                  defer_eval=True,   # owner 20/10 ~19:40Z: ~11.5h sessions, fd.py stops on its time budget (no round cap)
                  parent=dict(dataset=dataset, session=a.parent, run=run.name, round=last))
    checked = gen_fd.parent_files(queues['parent'])
    stage = base / 'artifacts/input' / f'{rid}-parent'; assert not stage.exists(); stage.mkdir()
    for rel in checked['files']:
        flat = ('resume_' if rel.startswith('resume/') else '') + Path(rel).name
        shutil.copyfile(run / rel, stage / flat)
    (stage / 'dataset-metadata.json').write_text(json.dumps(dict(id=dataset, title=dataset.split('/')[1], licenses=[dict(name='CC0-1.0')])))
    gen_fd.PLANS[round_tag] = [(a.owner, 42, a.session, queues, 11.5)]
    sys.argv = ['gen_fd.py', '--round', round_tag, '--sessions', a.session]
    gen_fd.main()
    print('PREPARED continuation from round', last + 1, '(time budget 11.5h) checkpoint stage', stage,
          'upload/verify the stage as', dataset, 'before push', flush=True)


if __name__ == '__main__': main()
