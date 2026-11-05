"""P20 locked contrasts, fixed endpoint20, verified artifacts only; no training or tuning."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics
from scripts.analyse_m20 import COHORTS, conditional_accuracy
from scripts.research_metrics import client_f1

PROBES = (1, 2, 5, 10, 15, 20)
CONTRASTS = [(0, 15, 'local_LA'), (1, 0, 'augmentation'), (2, 1, 'soft_class_Q'),
             (3, 2, 'binary_real_KD'), (4, 3, 'margin_vs_gate'), (4, 5, 'margin_vs_strength'),
             (5, 3, 'reduced_strength'), (7, 6, 'input_vs_class_same_bank'), (8, 7, 'server_proxy_fit'),
             (14, 8, 'carrier_control'), (9, 14, 'streaming_proxy_extrapolation'), (10, 8, 'hybrid_server'),
             (11, 1, 'finite_bank_hard'), (7, 11, 'soft_input_vs_bank_hard'),
             (12, 1, 'projection_hard'), (13, 2, 'projection_soft'), (13, 12, 'soft_Q_projected')]


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    study = ROOT / 'runs/p-study'
    out = study / 'analysis'
    out.mkdir(exist_ok=True)
    closure_path = study / 'final-audit/closure.json'
    closure = json.loads(closure_path.read_text())
    assert closure['full20_pass'] == 31 and closure['unresolved_audit_failures'] == 0
    for name, h in closure['report_sha256'].items(): assert sha(study / 'final-audit' / name) == h
    batch = json.loads((study / 'batch.json').read_text())
    counts = np.load(ROOT / 'evidence/research-data/train_counts.npy')
    support = np.load(ROOT / 'evidence/research-data/test_support.npy')
    n = counts.sum(1)
    accepted = {r['arm'] for r in closure['runs'] if r['status'] == 'PASS_BOUNDED'}
    snapshots = {r['arm']: r for r in json.loads((study / 'final-audit/diagnostic-audit.json').read_text())}
    arms, finals, csvrows = {}, {}, []
    for spec in batch:
        name = spec['arm']
        if name not in accepted: continue
        folder = ROOT / 'artifacts/p20' / spec['slug']
        d = folder / spec['slug']
        cfg = json.loads((d / 'config.json').read_text())
        rows = [json.loads(line) for line in (d / 'rounds.jsonl').read_text().splitlines()]
        curve = []
        for row in rows:
            r = row['round']
            assert sha(d / 'models' / f'r{r:03}.pt') == snapshots[name]['snapshots'][r - 1]['sha256']
            with np.load(d / f'cms_r{r:03}.npz') as z:
                assert np.array_equal(z['ids'], np.arange(5343))
                cm = z['private']
            assert np.array_equal(cm.sum(2), support)
            pooled, f = metrics.ten(cm.sum(0)), client_f1(cm)
            for k in metrics.TEN: np.testing.assert_allclose(pooled[k], row['private/' + k], atol=1e-12, rtol=0)
            point = {'round': r, 'f1': pooled['f1_macro'], 'mean': float(f.mean()), 'p10': float(np.percentile(f, 10)),
                     'std': float(f.std()), 'low256_mean': float(f[n < 256].mean()), 'ge4096_mean': float(f[n >= 4096].mean()),
                     'timeDelay_f1': pooled['f1_per_class'][13], 'other15_f1': float(np.delete(pooled['f1_per_class'], 13).mean())}
            curve.append(point)
            csvrows.append({'arm': name, 'mode': spec['mode'], 'seed': cfg['hp']['seed'], **point,
                            **{k: pooled[k] for k in metrics.TEN}})
            if r == 20:
                finals[name] = cm
                cohorts = []
                for lo, hi in COHORTS:
                    sel = (n >= lo) & (n < hi)
                    cohorts.append({'lo': lo, 'hi': hi, 'clients': int(sel.sum()), 'mean': float(f[sel].mean()),
                                    'p10': float(np.percentile(f[sel], 10)), 'by_local_support': conditional_accuracy(cm[sel], counts[sel])})
                endpoint = {**point, **pooled, 'cohorts': cohorts, 'by_local_support': conditional_accuracy(cm, counts)}
        train, teacher, shadow, external = {}, {}, {}, {}
        diagnostics = [json.loads(line) for rank in (0, 1) for line in (d / f'diagnostics_rank{rank}.jsonl').read_text().splitlines()]
        for r in PROBES:
            with np.load(d / f'train_cms_r{r:03}.npz') as z:
                tr = z['private']
            assert np.array_equal(tr.sum(2), counts)
            train[r] = metrics.ten(tr.sum(0))
            if r == 20:
                recall = np.divide(tr.diagonal(axis1=1, axis2=2), counts, out=np.zeros_like(counts, dtype=float), where=counts > 0)
                matched = {}
                for label, selected in [('all', n >= 0), ('low256', n < 256), ('ge4096', n >= 4096)]:
                    weights = support[:, 13] * selected * (counts[:, 13] > 0)
                    matched[label] = {'rows': int(weights.sum()), 'train_recall': float((recall[:, 13] * weights).sum() / weights.sum()),
                                      'test_recall': float((finals[name][:, 13, 13] * selected * (counts[:, 13] > 0)).sum() / weights.sum())}
            probes = [c for record in diagnostics if record['round'] == r for c in record['clients']]
            shadow[r] = {'clients': len(probes), 'median_gradient_norm': float(np.median([p['gradient_norm'] for p in probes])),
                         'exceeds_clip': sum(p['exceeds_clip'] for p in probes), 'scope': 'Shadow fp32/dropout0, not actual AMP/clipping'}
            paired = [c for rank in (0, 1) for c in json.loads((d / f'teacher_probe_rank{rank}_r{r:03}.json').read_text())]
            if paired:
                teacher[r] = {}
                for fold in ['train', 'test']:
                    teacher[r][fold] = {}
                    for label, recs in [('all', paired), ('low256', [p for p in paired if n[p['client_id']] < 256])]:
                        t = np.asarray([p[fold]['teacher_cm'] for p in recs])
                        s = np.asarray([p[fold]['student_cm'] for p in recs])
                        assert np.array_equal(t.sum(2), s.sum(2))
                        teacher[r][fold][label] = {'clients': len(recs), 'rows': int(t.sum()),
                            'teacher': metrics.ten(t.sum(0)), 'student': metrics.ten(s.sum(0)),
                            'teacher_correct_student_wrong': sum(p[fold]['teacher_correct_student_wrong'] for p in recs),
                            'student_correct_teacher_wrong': sum(p[fold]['student_correct_teacher_wrong'] for p in recs)}
        for r in (1, 10, 20):
            values = []
            for rank in (0, 1):
                with np.load(d / f'external_cms_rank{rank}_r{r:03}.npz') as z:
                    values.extend({'id': int(cid), 'f1': metrics.ten(c)['f1_macro'], 'selection_sha256': str(h)}
                                  for cid, c, h in zip(z['ids'], z['private'], z['selection_sha256']))
            external[r] = {'clients': values, 'mean_f1': float(np.mean([p['f1'] for p in values]))}
        log = json.loads((folder / (spec['slug'] + '.log')).read_text())
        timing = {key: float(sum(row.get(key, 0) for row in rows)) for key in
                  ['t_train', 't_eval', 't_diagnostics', 'server_seconds', 'scoring_seconds_local', 'bank_seconds_local']}
        timing['notebook_log_elapsed_seconds'] = max(p['time'] for p in log)
        timing['unattributed_seconds'] = timing['notebook_log_elapsed_seconds'] - sum(timing[k] for k in ['t_train', 't_eval', 't_diagnostics'])
        # Scoring/server/bank times are inside training, never add them again to walltime.
        boot = cfg['payload_bytes']['bootstrap']
        arms[name] = {'mode': spec['mode'], 'cell': spec['cell'], 'seed': cfg['hp']['seed'], 'final': endpoint,
                      'curve': curve, 'train': train, 'teacher_probe': teacher, 'shadow': shadow, 'external': external,
                      'matched_timeDelay': matched, 'delta_f1_15_20': curve[-1]['f1'] - curve[14]['f1'],
                      'delta_low256_15_20': curve[-1]['low256_mean'] - curve[14]['low256_mean'],
                      'bytes': {'bootstrap_up': boot['up'], 'bootstrap_down': boot['down'],
                                'rounds_up': sum(row['bytes_up'] for row in rows), 'rounds_down': sum(row['bytes_down'] for row in rows)},
                      'timing': timing, 'max_skip': max(row['skipped_frac'] for row in rows)}
        arms[name]['bytes']['total'] = sum(arms[name]['bytes'].values())
        print('P20_ANALYSED', name, flush=True)
    effects = []
    for target, base, label in CONTRASTS:
        for offset, seed in [(0, 42), (1, 43)]:
            name, reference = f'P{2 * target + offset}', f'P{2 * base + offset}'
            if name not in arms or reference not in arms: continue
            delta = client_f1(finals[name]) - client_f1(finals[reference])
            effect = {'label': label, 'seed': seed, 'target': name, 'reference': reference,
                      **{k: arms[name]['final'][k] - arms[reference]['final'][k] for k in
                         ['f1', 'mean', 'p10', 'std', 'low256_mean', 'ge4096_mean', 'timeDelay_f1', 'other15_f1']},
                      'better_clients': int((delta > 1e-12).sum()), 'worse_clients': int((delta < -1e-12).sum()),
                      'paired_delta_p10': float(np.percentile(delta, 10)), 'low256_better_fraction': float((delta[n < 256] > 1e-12).mean()),
                      'per_class_delta': (np.asarray(arms[name]['final']['f1_per_class']) - arms[reference]['final']['f1_per_class']).tolist()}
            if label == 'carrier_control':
                limits = {'f1': .001, 'timeDelay_f1': .01, 'p10': .01, 'low256_mean': .01}
                effect['tripwire_pass'] = all(abs(effect[k]) <= v for k, v in limits.items())
                effect['tripwire_limits'] = limits
            effects.append(effect)
    result = {'closure_sha256': sha(closure_path), 'arms': arms, 'effects': effects,
              'missing': ['P21: failed before round1; C10 has seed42 only'],
              'scope': 'Fixed20round screening, 2seeds except hybrid, repeated officialtest; live total effects, no significance/exactresume/mixedarchitecture/contact proof'}
    (out / 'analysis.json').write_text(json.dumps(result, indent=2) + '\n')
    with (out / 'metrics-all-rounds.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(csvrows[0])); writer.writeheader(); writer.writerows(csvrows)
    metricdoc = ['# P20 — 10 metric own-test tại round20', '', '31run PASS_BOUNDED; P21 thiếu, không nội suy. Số thập phân, không phần trăm.', '',
                 '| arm | mode | seed | ' + ' | '.join(metrics.TEN) + ' |', '|---|---|---|' + '---|' * len(metrics.TEN)]
    for name, arm in arms.items():
        metricdoc.append(f"| {name} | {arm['mode']} | {arm['seed']} | " + ' | '.join(f"{arm['final'][k]:.6f}" for k in metrics.TEN) + ' |')
    (ROOT / 'docs/research/metrics-p20.md').write_text('\n'.join(metricdoc) + '\n')
    print('P20_ANALYSIS_VERIFIED_DONE', flush=True)


if __name__ == '__main__': main()
