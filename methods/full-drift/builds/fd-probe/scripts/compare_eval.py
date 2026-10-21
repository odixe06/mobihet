"""Equivalence evidence for fd_eval_fast.py (owner 21/10, tolerance class): an audited fast evaluation vs the audited
fd_eval.py evaluation of the same rounds. Per round: prediction agreement of every client/rule array and the largest change
of any of the 10 mean metrics; written to <fast rid>/EQUIVALENCE.json.

    python compare_eval.py fd-e01-04-s42za-20261015-r25-v1-v1 fd-e01-04-s42za-20261015-r25-v1-f1
"""
import json
import sys

import numpy as np

from collect_fd import BASE


def main():
    old, new = (BASE / 'artifacts/kaggle' / rid for rid in sys.argv[1:3])
    a, b = (json.loads((d / 'AUDITED.json').read_text()) for d in (old, new))
    assert a['session'] == b['session'] and a['run'] == b['run'] and a['fingerprint'] == b['fingerprint']
    rounds = sorted(set(a['rounds']) & set(b['rounds'])); assert rounds
    rows = {}
    for r in rounds:
        x, y = (json.loads((d / 'runs' / a['run'] / 'eval' / f'round_{r:03d}.json').read_text()) for d in (old, new))
        assert x['weights_sha'] == y['weights_sha']
        delta = max(abs(x['eval'][role]['test'][rule]['mean_metrics'][key] - value)
                    for role, holds in y['eval'].items() for rule, ev in holds['test'].items() for key, value in ev['mean_metrics'].items())
        with np.load(old / 'runs' / a['run'] / 'predictions' / f'round_{r:03d}.npz') as p, \
             np.load(new / 'runs' / a['run'] / 'predictions' / f'round_{r:03d}.npz') as q:
            assert set(p.files) == set(q.files) and np.array_equal(p['test_y'], q['test_y'])
            diff = {k: (int((p[k] != q[k]).sum()), p[k].size) for k in p.files if k != 'test_y'}
        agree = {k: 1 - d / n for k, (d, n) in diff.items()}
        rows[r] = dict(max_metric_delta=delta, differing_predictions=sum(d for d, _ in diff.values()),
                       predictions=sum(n for _, n in diff.values()), min_array_agreement=min(agree.values()),
                       worst_array=min(agree, key=agree.get), fast_gate=y['fast_gate'])
        print('round', r, rows[r], flush=True)
    out = dict(reference=sys.argv[1], fast=sys.argv[2], session=a['session'], run=a['run'], rounds=rows,
               max_metric_delta=max(v['max_metric_delta'] for v in rows.values()),
               agreement=1 - sum(v['differing_predictions'] for v in rows.values()) / sum(v['predictions'] for v in rows.values()))
    (new / 'EQUIVALENCE.json').write_text(json.dumps(out, indent=2) + '\n')
    print('EQUIVALENCE', sys.argv[2], 'agreement', out['agreement'], 'max metric delta', out['max_metric_delta'], flush=True)


if __name__ == '__main__': main()
