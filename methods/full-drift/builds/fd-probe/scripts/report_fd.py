"""Locked FullDrift gates (docs/methods/full-drift/design.md) + all ten metrics from audited sessions only.

    python methods/full-drift/builds/fd-probe/scripts/report_fd.py
"""
import csv
import json
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
from mh.metrics import METRIC_KEYS
from scripts.client_view import prior, view

# round 6 seed-43 pairs (design §Vòng 6): one tech may have two controls, so gate names get '_vs_<control>' on collision
S43 = [('fd_dml_m4b', 'fd_dml'), ('fd_dml_m4b_sub4_q4', 'fd_dml_m4b'), ('fd_dml_m4b_sub4_q4', 'fd_dml_sub4_q4'),
       ('fd_dml_sub4_q4', 'fd_dml'), ('fd_dml_swappos', 'fd_dml')]   # last pair: round 7 (design §Vòng 7, locked ~17:29Z)
PRIMARY = {'dml': ('proxy', 'head'), 'fedavg': ('proxy', 'head'), 'gaussian': ('private', 'gp')}
# (technique, control, epoch, rule) — design.md round 1 and §Vòng 2; 15-epoch arms are keyed '<arm>@e15', seed 43 '<arm>@s43'.
GATES = [('fd_dml_sub4', 'fd_dml', 6, 'gain'), ('fd_dml_prox', 'fd_dml', 6, 'gain'), ('fd_dml_avgm', 'fd_dml', 6, 'gain'),
         ('fd_fedavg_sub4', 'fd_fedavg', 6, 'gain'), ('fd_gaussian_swappos', 'fd_gaussian', 6, 'gain'),
         ('fd_dml_sub4_q4', 'fd_dml_sub4', 6, 'noninferior'), ('fd_fedavg_sub4_q4', 'fd_fedavg_sub4', 6, 'noninferior'),
         ('fd_dml_sub16_q4', 'fd_dml_sub4_q4', 6, 'gain'), ('fd_fedavg_sub16_q4', 'fd_fedavg_sub4_q4', 6, 'gain'),
         ('fd_dml_sub4_prox', 'fd_dml_sub4', 6, 'benign'), ('fd_fedavg_sub4_prox', 'fd_fedavg_sub4', 6, 'benign'),
         ('fd_dml_sub4@e15', 'fd_dml@e15', 15, 'horizon'), ('fd_fedavg_sub4@e15', 'fd_fedavg@e15', 15, 'horizon'),
         ('fd_dml_q4', 'fd_dml', 6, 'noninferior'),   # round 3 (R1 bytes at one aggregation/epoch)
         ('fd_dml_m4b_q4', 'fd_dml_m4b', 6, 'noninferior'), ('fd_dml_m4b@e15', 'fd_dml@e15', 15, 'horizon'),   # round 5
         *[(t + '@s43', c + '@s43', 6, 'gain') for t, c in S43]]   # round 6
# Owner 16/10 ~03:00Z: target = client (private gp, R2). Locked ~04:00Z before any round-2 epoch-6 result (W&B r2 seen).
# DML arms mirror the proxy gate rules on private/gp; FedAvg has no private model (descriptive only).
PRIVATE_GATES = [('fd_dml_sub4_q4', 'fd_dml_sub4', 6, 'noninferior'), ('fd_dml_sub16_q4', 'fd_dml_sub4_q4', 6, 'gain'),
                 ('fd_dml_sub4_prox', 'fd_dml_sub4', 6, 'benign'), ('fd_dml_sub4@e15', 'fd_dml@e15', 15, 'horizon'),
                 # round 3 (design §Vòng 3, locked ~04:05Z): client-side only, no extra bytes/contacts
                 ('fd_dml_m4b', 'fd_dml', 6, 'gain'), ('fd_dml_swappos', 'fd_dml', 6, 'gain'), ('fd_dml_ntd', 'fd_dml', 6, 'gain'),
                 ('fd_dml_q4', 'fd_dml', 6, 'noninferior'),
                 # round 4 (design §Vòng 4, locked ~06:30Z)
                 ('fd_dml_hkd', 'fd_dml', 6, 'gain'), ('fd_gaussian_m4b', 'fd_gaussian', 6, 'gain'),
                 # round 5 (design §Vòng 5, locked ~08:25Z)
                 ('fd_dml_m4b@e15', 'fd_dml@e15', 15, 'horizon'), ('fd_dml_m4b_q4', 'fd_dml_m4b', 6, 'noninferior'),
                 ('fd_dml_m4b_sub4_q4', 'fd_dml_m4b', 6, 'gain'),
                 *[(t + '@s43', c + '@s43', 6, 'gain') for t, c in S43]]   # round 6 (design §Vòng 6, locked ~16:50Z)
PRIVATE_SEEN = [('fd_dml_sub4', 'fd_dml'), ('fd_dml_prox', 'fd_dml'), ('fd_dml_avgm', 'fd_dml')]   # round 1, seen before lock
RULES = {'gain': lambda d, b: d >= .010 and b >= -.005, 'noninferior': lambda d, b: d >= -.010 and b >= -.010,
         'benign': lambda d, b: b >= .010 and d >= -.005, 'horizon': lambda d, b: d >= .010}


def runs():
    found, manifests = {}, {}
    for marker in sorted((BASE / 'artifacts/kaggle').glob('fd-s42*/AUDITED.json')):
        for run in sorted((marker.parent / 'runs').glob('fd_*_s4[23]*')):
            rows = [json.loads(m.read_text()) for m in sorted((run / 'complete').glob('round_*.json'))]
            arm, seed, epochs = re.fullmatch(r'(fd_.+)_s(4[23])(?:_(e15))?', run.name).groups()
            key = arm + (f'@{epochs}' if epochs else '') + ('@s43' if seed == '43' else '')   # seed 42 keys unchanged
            assert key not in found, ('duplicate arm across sessions', key)
            if rows: found[key], manifests[key] = rows, marker.parent / 'source/data_manifest.json'
    return found, manifests


def primary(arm):
    return PRIMARY[arm.split('@')[0].split('_')[1]]


def score(rows, role, rule, hold='group', epoch=6):
    row = next((r for r in rows if r['round'] == epoch), None)
    if not row or role not in row['eval']: return None
    x = row['eval'][role][hold][rule]; return x['f1_macro'], x['benign_f1']


def main():
    out = ROOT / 'docs/methods/full-drift/results'; out.mkdir(parents=True, exist_ok=True)
    data, manifests = runs(); metrics, classes, gates = [], [], {}
    for arm, rows in data.items():
        for row in rows:
            for role, holds in row['eval'].items():
                for hold, rules in holds.items():
                    for rule, ev in rules.items():
                        key = dict(arm=arm, epoch=row['round'], role=role, holdout=hold, rule=rule)
                        metrics.append(dict(key, **ev['mean_metrics']))
                        classes.extend(dict(key, class_id=c, f1=f) for c, f in enumerate(ev['f1_per_class']))
    for tech, ctl, epoch, kind in GATES:
        role, rule = primary(ctl)
        a, b = score(data.get(ctl, []), role, rule, epoch=epoch), score(data.get(tech, []), role, rule, epoch=epoch)
        gates[tech if tech not in gates else f'{tech}_vs_{ctl}'] = dict(control=ctl, epoch=epoch, rule=kind, primary=f'{role}/{rule}', delta=b[0] - a[0] if a and b else None,
                           benign_delta=b[1] - a[1] if a and b else None,
                           promising=bool(a and b and RULES[kind](b[0] - a[0], b[1] - a[1])) if a and b else None)
    for arm in [k for k in data if k.endswith('@e15')]:   # determinism: 15-epoch rerun vs round 1 on epochs 1-6
        base = arm.split('@')[0]; role, rule = primary(base)
        pairs = [(score(data[arm], role, rule, epoch=e), score(data.get(base, []), role, rule, epoch=e)) for e in range(1, 7)]
        if all(x and y for x, y in pairs):
            gates[f'{arm}:replay_ep1-6'] = dict(descriptive=True, delta=max(abs(x[0] - y[0]) for x, y in pairs))
    for arm, rows in data.items():
        a, b = score(rows, 'private', 'gp'), score(rows, 'private_ema', 'gp')
        if a and b:
            gates[f'{arm}:private_ema'] = dict(control='private/gp', delta=b[0] - a[0], benign_delta=b[1] - a[1],
                                               promising=b[0] - a[0] >= .010 and b[1] - a[1] >= -.005)
    for tech, ctl, epoch, kind in PRIVATE_GATES:
        a, b = score(data.get(ctl, []), 'private', 'gp', epoch=epoch), score(data.get(tech, []), 'private', 'gp', epoch=epoch)
        gates[f'{tech}:private_gp' if f'{tech}:private_gp' not in gates else f'{tech}:private_gp_vs_{ctl}'] = dict(control=ctl, epoch=epoch, rule=kind, primary='private/gp',
                                           delta=b[0] - a[0] if a and b else None, benign_delta=b[1] - a[1] if a and b else None,
                                           promising=bool(RULES[kind](b[0] - a[0], b[1] - a[1])) if a and b else None)
    for tech, ctl in PRIVATE_SEEN:
        a, b = score(data.get(ctl, []), 'private', 'gp'), score(data.get(tech, []), 'private', 'gp')
        if a and b: gates[f'{tech}:private_gp'] = dict(descriptive=True, delta=b[0] - a[0], benign_delta=b[1] - a[1])
    a, b = score(data.get('fd_dml', []), 'proxy', 'head'), score(data.get('fd_fedavg', []), 'proxy', 'head')
    gates['fd_fedavg_vs_fd_dml'] = dict(descriptive=True, delta=b[0] - a[0] if a and b else None)
    with (out / 'metrics-all.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['arm', 'epoch', 'role', 'holdout', 'rule', *METRIC_KEYS]); w.writeheader(); w.writerows(metrics)
    with (out / 'per-class-f1.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['arm', 'epoch', 'role', 'holdout', 'rule', 'class_id', 'f1']); w.writeheader(); w.writerows(classes)
    (out / 'gates.json').write_text(json.dumps(gates, indent=2) + '\n')
    lines = ['# FullDrift — seed42 sàng lọc, group holdout (audited)', '',
             'Primary: DML/FedAvg proxy head, Gaussian private gp. Test chỉ epoch 6, mô tả. Drift = ‖θ_k−θ_chung‖/‖θ_chung‖.', '',
             '| arm | epoch | primary group | benign | private gp | private_ema gp | old primary | drift | test primary |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm, rows in sorted(data.items()):
        role, rule = primary(arm)
        for row in rows:
            ev = row['eval']; g = lambda r, u, h='group': f"{ev[r][h][u]['f1_macro']:.4f}" if r in ev and h in ev[r] else ''
            drift = row['diagnostics'].get('drift_mean')
            lines.append(f"| {arm} | {row['round']} | {g(role, rule)} | {ev[role]['group'][rule]['benign_f1']:.4f} | "
                         f"{g('private', 'gp')} | {g('private_ema', 'gp')} | {g(role, rule, 'old')} | "
                         f"{'' if drift is None else f'{drift:.4f}'} | {g(role, rule, 'test')} |")
    lines += ['', '| so sánh (group; epoch theo cổng) | Δ primary | Δ benign | cổng (rule trong gates.json) |', '|---|---:|---:|---|']
    for name, gate in gates.items():
        d, bd = gate.get('delta'), gate.get('benign_delta')
        lines.append(f"| {name} | {'' if d is None else f'{d:+.4f}'} | {'' if bd is None else f'{bd:+.4f}'} | "
                     f"{'mô tả' if gate.get('descriptive') else gate['promising']} |")
    lines += ['', '## Góc nhìn client (epoch cuối, group): mean20client (client tệ nhất)', '',
              'local = confusion đánh trọng số lại theo prior nhãn train của từng client (chỉ lệch nhãn).', '',
              '| arm | epoch | model/rule | global F1 (min) | local F1 (min) | local acc (min) |', '|---|---:|---|---|---|---|']
    for arm, rows in sorted(data.items()):
        row, pi = rows[-1], prior(manifests[arm])
        for role, holds in row['eval'].items():
            for rule, ev in holds['group'].items():
                lines.append(f"| {arm} | {row['round']} | {role}/{rule} | {view(ev['confusion_clients'], pi)} |")
    lines += ['', '[gates.json](gates.json) · [10 metrics](metrics-all.csv) · [per-class](per-class-f1.csv)']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('FullDrift report arms', sorted(data), json.dumps({k: v.get('promising', v.get('delta')) for k, v in gates.items()}))


if __name__ == '__main__':
    main()
