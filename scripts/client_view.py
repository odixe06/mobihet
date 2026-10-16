"""Per-client view of audited round JSONs: global macro F1 vs client-local label prior (test CM rows reweighted by
pi_k/pi_test). Label shift only: per-client feature shift needs a real per-client holdout.

    python scripts/client_view.py <round_json>[::holdout] ...   # holdout defaults to 'test'
"""
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'methods/shared-baselines/builds/federated-baselines-full/artifacts/kaggle/full1-s42-20261014-cont02-v1/source/data_manifest.json'


def f1(cm, keep):
    tp, pred, true = np.diag(cm), cm.sum(0), cm.sum(1)
    prec = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    rec = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    f = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=prec + rec > 0)
    return f[keep].mean(), rec


def view(cms, pi):
    glob, local_f1, local_acc = [], [], []
    for k, cm in enumerate(np.asarray(cms, float)):
        test_pi = cm.sum(1) / cm.sum(); present = cm.sum(1) > 0
        g, rec = f1(cm, present); glob.append(g)
        w = np.divide(pi[k], test_pi, out=np.zeros(len(pi[k])), where=test_pi > 0)
        local_f1.append(f1(cm * w[:, None], present & (pi[k] > 0))[0]); local_acc.append(float((pi[k] * rec).sum()))
    return ' | '.join(f'{name} {np.mean(v):.4f} (min {np.min(v):.4f})'
                      for name, v in (('global_f1', glob), ('local_f1', local_f1), ('local_acc', local_acc)))


def prior(manifest):
    counts = np.array([json.loads(Path(manifest).read_text())['counts'][f'train_{k}'] for k in range(20)], float)
    return counts / counts.sum(1, keepdims=True)


def main():
    pi = prior(MANIFEST)
    for arg in sys.argv[1:]:
        path, _, hold = arg.partition('::'); hold = hold or 'test'
        d = json.loads(Path(path).read_text()); name = f"{Path(path).parts[-3]} r{d['round']} {hold}"
        for role, holds in d['eval'].items():
            for rule, ev in holds.get(hold, {}).items():
                print(f'{name:34s} {role:11s} {rule:4s}', view(ev['confusion_clients'], pi))


if __name__ == '__main__':
    main()
