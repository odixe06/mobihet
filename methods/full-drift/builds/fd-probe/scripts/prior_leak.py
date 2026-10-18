"""Local label-prior leakage into the private decision (audited round JSONs, no GPU): per client k and class c, over-prediction
log((pred_c + 1) / (true_c + 1)) on `group` vs the local prior excess log pi_k,c - log pi_g,c, both centred per class across clients.
Slope 1 = the private predicts with its full local prior, 0 = none, < 0 = over-corrected (design §Luật triển khai head vs gp).

    python methods/full-drift/builds/fd-probe/scripts/prior_leak.py
"""
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1] / 'artifacts/kaggle'
RUNS = {'m4b@e15': ('fd-s42j-20261015-r5-v1', 'fd_dml_m4b_s42_e15'), 'gpool@e15': ('fd-s42lc-20261015-r11-v1', 'fd_dml_m4b_gpool_s42_e15'),
        'frz@e15': ('fd-s42oa-20261015-r15-v1', 'fd_dml_m4b_frz_s42_e15'),
        'gpool_frz@e15': ('fd-s42ob-20261015-r15-v1', 'fd_dml_m4b_gpool_frz_s42_e15'),
        'gkd@e15': ('fd-s42oc-20261015-r15-v1', 'fd_dml_m4b_gpool_frz_gkd_s42_e15'),
        'gkd_la@e15': ('fd-s42ra-20261015-r17-v1', 'fd_dml_m4b_gpool_frz_gkd_la_s42_e15'),
        'gkd_kdpx@e15': ('fd-s42rb-20261015-r17-v1', 'fd_dml_m4b_gpool_frz_gkd_kdpx_s42_e15'),
        'gkd_la_kdpx@e15': ('fd-s42rc-20261015-r17-v1', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx_s42_e15')}


def leak(cms, counts):
    true, pred = cms.sum(2), cms.sum(1)
    over = np.log((pred + 1) / (true + 1))
    excess = np.log((counts + 1) / (counts + 1).sum(1, keepdims=True)) - np.log((counts.sum(0) + 1) / (counts.sum() + 16))
    o, e = over - over.mean(0), excess - excess.mean(0); keep = true.sum(0) > 0
    o, e = o[:, keep], e[:, keep]
    return float(np.corrcoef(o.ravel(), e.ravel())[0, 1]), float((o * e).sum() / (e * e).sum())


def main():
    for name, (session, run) in RUNS.items():
        folder = BASE / session
        counts = np.array([json.loads((folder / 'source/data_manifest.json').read_text())['counts'][f'train_{k}'] for k in range(20)], float)
        row = json.loads(sorted((folder / 'runs' / run / 'complete').glob('round_*.json'))[-1].read_text())
        for rule in ('head', 'gp'):
            corr, slope = leak(np.array(row['eval']['private']['group'][rule]['confusion_clients'], float), counts)
            print(f'{name:16s} ep{row["round"]} {rule:4s} corr {corr:+.3f} slope {slope:+.3f}')


if __name__ == '__main__':
    main()
