"""Locked K5 quality, compression and joint contrasts from audited terminal artifacts."""
import csv
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mh.metrics import METRIC_KEYS


def main():
    base = ROOT / 'papers/v6'
    registry = {v['run_id']: v for v in json.loads((base / 'wandb-runs.json').read_text())}
    runs, failures, evaluations, classes = {}, {}, [], []
    audits = sorted((base / 'kaggle').glob('*/AUDITED.json'))
    for marker in audits:
        audit = json.loads(marker.read_text())
        for item in audit['audits']:
            arm, seed = item['arm'], audit['seed']
            path = marker.parent / 'runs' / f'{arm}_s{seed}'
            cfg = json.loads((path / 'config.json').read_text())
            fp = cfg['fingerprint']
            assert registry[f'{arm}_s{seed}-{fp[:8]}']['fingerprint'] == fp
            history = [json.loads(p.read_text()) for p in sorted((path / 'complete').glob('round_*.json'))]
            assert len(history) == item['rounds']
            if not item['complete']:
                failures[(arm, seed)] = item
            result = json.loads((path / 'result.json').read_text()) if (path / 'result.json').exists() else None
            if result:
                assert result['fingerprint'] == fp and result['complete'] == item['complete']
                ledger = result['bytes']
                minutes = result['minutes']
            elif history:
                state = torch.load(path / f'resume/round_{len(history):03d}.pt', map_location='cpu', weights_only=True)
                ledger = state['bytes']
                minutes = sum(h['sec'] for h in history) / 60
            else:
                ledger, minutes = [], None
            runs[(arm, seed)] = dict(history=history, complete=item['complete'], bytes=ledger,
                                     minutes=minutes, fingerprint=fp)
    local = {}
    for seed in (42, 43):
        audit = json.loads((ROOT / f'papers/v2/kaggle/v2a-s{seed}-20261011-v1/AUDITED.json').read_text())
        assert next(v for v in audit['audits'] if v['arm'] == 'local')['complete']
        local[seed] = json.loads((ROOT / f'papers/v2/runs/local_s{seed}/result.json').read_text())

    def ev(arm, seed, hold='group'):
        run = local[seed] if arm == 'local' else runs.get((arm, seed))
        return run['history'][-1]['eval']['private'][hold]['gp'] if run and run['complete'] else None

    def bootstrap(arm, seed):
        return sum(v[-1] for v in runs[(arm, seed)]['bytes'] if v[0] == 0) / 20

    gates, joint = {}, {}
    for seed in (42, 43):
        for dimension in (59, 61):
            for mask in ('marginal', 'joint'):
                label = f'G{dimension}_{mask}'
                arm = f'k{dimension}_{mask}_fl'
                fail = failures.get((arm, seed))
                if fail:
                    scientific = fail.get('failure_category') == 'numerical' or fail['rounds'] > 0
                    gates[f'{label}_s{seed}'] = dict(pass_quality=False if scientific else None,
                        pass_compression=False if scientific else None, rounds=fail['rounds'],
                        reason='numerical/partial under locked configuration' if scientific else 'startup failure needs diagnosis')
                    continue
                support = f'k66_{mask}_fl'
                own, shuffle = f'k{dimension}_{mask}_own', f'k{dimension}_{mask}_shuffle'
                needed = (arm, support, 'k66_plain_fl', own, shuffle)
                if any(ev(a, seed) is None for a in needed):
                    continue
                v, control, plain = ev(arm, seed), ev(support, seed), ev('k66_plain_fl', seed)
                reference = ev('local', seed)
                common = (v['f1_macro'] - reference['f1_macro'] >= .03
                          and ev(arm, seed, 'old')['f1_macro'] > ev('local', seed, 'old')['f1_macro']
                          and v['benign_f1'] >= reference['benign_f1'] - .03)
                gain = v['f1_macro'] - max(control['f1_macro'], plain['f1_macro'])
                own_gain = v['f1_macro'] - ev(own, seed)['f1_macro']
                shuffle_gain = v['f1_macro'] - ev(shuffle, seed)['f1_macro']
                ratio = bootstrap(arm, seed) / bootstrap(support, seed)
                quality = common and gain >= .015 and own_gain >= .01 and shuffle_gain >= .01
                compression = v['f1_macro'] - control['f1_macro'] >= -.005 and ratio <= .65
                gates[f'{label}_s{seed}'] = dict(pass_quality=bool(quality), pass_compression=bool(compression),
                    common_pass=bool(common), gain_vs_stronger_full66=gain, gain_vs_own=own_gain,
                    gain_vs_shuffle=shuffle_gain, bootstrap_bytes_per_client=bootstrap(arm, seed),
                    control_bootstrap_bytes_per_client=bootstrap(support, seed), byte_ratio=ratio)
            a, b = f'k{dimension}_joint_fl', f'k{dimension}_marginal_fl'
            if all(ev(name, seed) is not None for name in (a, b, 'k66_joint_fl', 'k66_marginal_fl')):
                contrast = ev(a, seed)['f1_macro'] - ev(b, seed)['f1_macro']
                control = ev('k66_joint_fl', seed)['f1_macro'] - ev('k66_marginal_fl', seed)['f1_macro']
                joint[f'G{dimension}_s{seed}'] = dict(contrast=contrast,
                    contribution=contrast >= .005, full66_contrast=control, interaction=contrast - control)
    paired = {}
    for dimension in (59, 61):
        for mask in ('marginal', 'joint'):
            label = f'G{dimension}_{mask}'
            values = [gates.get(f'{label}_s{seed}', {}) for seed in (42, 43)]
            paired[label] = {key: all(v[key] for v in values) if all(v.get(key) is not None for v in values) else None
                             for key in ('pass_quality', 'pass_compression')}
    (base / 'gates.json').write_text(json.dumps(dict(per_seed=gates, paired=paired, joint=joint), indent=2))
    fields = ['arm', 'seed', 'round', 'role', 'holdout', 'rule']
    for (arm, seed), run in sorted(runs.items()):
        for h in run['history']:
            for role, holds in h.get('eval', {}).items():
                for hold, rules in holds.items():
                    for rule, value in rules.items():
                        identity = dict(arm=arm, seed=seed, round=h['round'], role=role, holdout=hold, rule=rule)
                        evaluations.append(dict(identity, **value['mean_metrics']))
                        classes.extend(dict(identity, class_id=c, f1=v) for c, v in enumerate(value['f1_per_class']))
    for name, data, extra in [('metrics-all.csv', evaluations, list(METRIC_KEYS)),
                              ('per-class-f1.csv', classes, ['class_id', 'f1'])]:
        with (base / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields + extra)
            w.writeheader()
            w.writerows(data)
    lines = ['# V6 — geometry artifact report', '',
        '[Frozen contract](../../docs/research/pilot-v6-geometry.md), [ledger](rebuild.md), [gates](gates.json).',
        f'{len(audits)}/4 audited kernels; {len(runs)}/30 audited arms; {len(failures)} failed/partial.',
        'Source/payload/weights/prediction audit required; status/W&B alone excluded. StageU, no R1/R2 proof.', '',
        '| variant | paired quality | paired compression |', '|---|---|---|']
    verdict = lambda v: 'PENDING' if v is None else 'PASS' if v else 'FAIL'
    lines.extend(f'| {label} | {verdict(v["pass_quality"])} | {verdict(v["pass_compression"])} |'
                 for label, v in paired.items())
    lines += ['', '## Complete r20', '',
        '| arm | seed | group/gp F1 | benign F1 | minutes | bootstrap up B/client | down B/client |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for (arm, seed), run in sorted(runs.items()):
        if not run['complete']:
            continue
        v = ev(arm, seed)
        sizes = [sum(x[-1] for x in run['bytes'] if x[0] == 0 and x[2] == direction) / 20
                 for direction in ('up', 'down')]
        lines.append(f'| {arm} | {seed} | {v["f1_macro"]:.6f} | {v["benign_f1"]:.6f} | '
                     f'{run["minutes"]:.2f} | {sizes[0]:.0f} | {sizes[1]:.0f} |')
    lines += ['', '## Failed/partial', '', '| arm | seed | committed rounds | category |', '|---|---:|---:|---|']
    lines.extend(f'| {arm} | {seed} | {v["rounds"]} | {v.get("failure_category", "unknown")} |'
                 for (arm, seed), v in sorted(failures.items()))
    lines += ['', '## All 10 metrics', '', '[Full precision CSV](metrics-all.csv), [per-class F1](per-class-f1.csv).',
        'Mean per-client confusion metrics; pooled confusion is illustration only.', '',
        '| ' + ' | '.join(fields + list(METRIC_KEYS)) + ' |', '|---|---:|---:|---|---|---|' + '---:|' * 10]
    lines.extend('| ' + ' | '.join([str(row[k]) for k in fields] + [f'{row[k]:.6f}' for k in METRIC_KEYS]) + ' |'
                 for row in evaluations)
    lines += ['', 'Joint-minus-marginal and its interaction with full66 are in gates.json; '
        'joint contribution does not replace geometry utility. Own retains paid global prior. '
        'Application bytes include actual FP64 statistics/FP16 factors, exclude contact ACK/frame/security/retry. '
        'Fresh full66 controls only; differences from historical P1 do not isolate geometry. '
        'Two seeds are not a confidence interval. Numerical failures preserved; no unchanged rerun or gate rescue.']
    (base / 'report.md').write_text('\n'.join(lines) + '\n')
    print('V6 report:', len(runs), 'arms;', len(evaluations), 'ten-metric rows;', paired)


if __name__ == '__main__':
    main()
