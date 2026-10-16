"""Locked AntiMem/QProxy gates + all ten metrics from audited sessions only (docs/methods/anti-memorization/design.md)."""
import csv
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
sys.path.insert(0, str(ROOT))
from mh.metrics import METRIC_KEYS

PRIMARY = {'dml': ('proxy', 'head'), 'hybrid': ('proxy', 'head'), 'gaussian': ('private', 'gp')}
EMA_ROLE = {'proxy': 'proxy_ema', 'private': 'private_ema'}


def sessions(seed):
    return [p.parent for p in sorted((BASE / 'artifacts/kaggle').glob(f'am-s{seed}-20261015-*/AUDITED.json'))]


def final(seed, arm):
    for session in sessions(seed):   # audited sessions only; an arm r20 in two sessions is reported from the first
        marker = session / f'runs/{arm}_s{seed}/complete/round_020.json'
        if marker.exists():
            return json.loads(marker.read_text())['eval']
    return None


def score(ev, role, rule):
    x = ev[role]['group'][rule]; return x['f1_macro'], x['benign_f1']


def main():
    out = ROOT / 'docs/methods/anti-memorization/results'; out.mkdir(parents=True, exist_ok=True)
    audited = {s: [json.loads((d / 'AUDITED.json').read_text()) for d in sessions(s)] for s in (42, 43) if sessions(s)}
    rows, pc, gates = [], [], {}
    for seed in audited:
        for run in sorted(r for d in sessions(seed) for r in (d / 'runs').glob(f'am_*_s{seed}')):
            for m in sorted((run / 'complete').glob('round_*.json')):
                d = json.loads(m.read_text())
                for role, holds in d.get('eval', {}).items():
                    for hold, rules in holds.items():
                        for rule, ev in rules.items():
                            key = dict(seed=seed, arm=run.name.rsplit('_s', 1)[0], round=d['round'], role=role, holdout=hold, rule=rule)
                            rows.append(dict(key, **ev['mean_metrics']))
                            pc.extend(dict(key, class_id=c, f1=f) for c, f in enumerate(ev['f1_per_class']))
    for base, (role, rule) in PRIMARY.items():
        for tech in ('swap', 'm4b', 'swappos'):
            per = {}
            for seed in audited:
                a, b = final(seed, f'am_{base}'), final(seed, f'am_{base}_{tech}')
                if a and b:
                    (fa, ba), (fb, bb) = score(a, role, rule), score(b, role, rule)
                    per[seed] = dict(delta=fb - fa, benign_delta=bb - ba, pass_=fb - fa >= .010 and bb - ba >= -.005)
            gates[f'{base}_{tech}'] = dict(primary=f'{role}/{rule}', seeds=per,
                                           pass_all=len(per) == 2 and all(v['pass_'] for v in per.values()))
        per = {}
        for seed in audited:
            for tech in ('', '_swap', '_m4b', '_swappos'):
                ev = final(seed, f'am_{base}{tech}')
                if ev and EMA_ROLE[role] in ev:
                    (f0, b0), (f1, b1) = score(ev, role, rule), score(ev, EMA_ROLE[role], rule)
                    per[f'{seed}{tech}'] = dict(delta=f1 - f0, benign_delta=b1 - b0, pass_=f1 - f0 >= .010)
        gates[f'{base}_ema'] = dict(primary=f'{EMA_ROLE[role]}/{rule}', runs=per,
                                    pass_all_base=all(per.get(f'{s}', {}).get('pass_', False) for s in (42, 43)))
    per = {}
    for seed in audited:
        dense = final(seed, 'am_dml')
        for bits, tol in ((8, -.005), (4, -.010)):
            q = final(seed, f'am_dml_q{bits}')
            if dense and q:
                dp, dg = score(q, 'proxy', 'head')[0] - score(dense, 'proxy', 'head')[0], score(q, 'private', 'gp')[0] - score(dense, 'private', 'gp')[0]
                db = score(q, 'proxy', 'head')[1] - score(dense, 'proxy', 'head')[1]
                per[f'{seed}_q{bits}'] = dict(proxy_delta=dp, private_gp_delta=dg, benign_delta=db,
                                             pass_=dp >= tol and dg >= tol and db >= -.010)
    gates['qproxy'] = dict(seeds=per, q4_pass_all=all(per.get(f'{s}_q4', {}).get('pass_', False) for s in (42, 43)))
    with (out / 'metrics-all.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['seed', 'arm', 'round', 'role', 'holdout', 'rule', *METRIC_KEYS]); w.writeheader(); w.writerows(rows)
    with (out / 'per-class-f1.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['seed', 'arm', 'round', 'role', 'holdout', 'rule', 'class_id', 'f1']); w.writeheader(); w.writerows(pc)
    (out / 'gates.json').write_text(json.dumps(dict(audited={s: [a['kernel'] for a in v] for s, v in audited.items()}, gates=gates), indent=2) + '\n')
    lines = ['# AntiMem/QProxy — r20 group (audited)', '', '| seed | arm | proxy head | private gp | proxy_ema | private_ema gp | benign (primary) |',
             '|---|---|---:|---:|---:|---:|---:|']
    for seed in audited:
        for arm in sorted({r.name.rsplit('_s', 1)[0] for d in sessions(seed) for r in (d / 'runs').glob(f'am_*_s{seed}')}):
            ev = final(seed, arm)
            if not ev: lines.append(f'| {seed} | {arm} | partial | | | | |'); continue
            g = lambda r, u: f"{ev[r]['group'][u]['f1_macro']:.4f}" if r in ev else ''
            role, rule = PRIMARY[arm.split('_')[1]]
            lines.append(f"| {seed} | {arm} | {g('proxy','head')} | {g('private','gp')} | {g('proxy_ema','head')} | {g('private_ema','gp')} | {ev[role]['group'][rule]['benign_f1']:.4f} |")
    lines += ['', 'Gates: [gates.json](gates.json) · 10 metrics: [metrics-all.csv](metrics-all.csv) · [per-class](per-class-f1.csv)']
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('AntiMem report rows', len(rows), json.dumps({k: v.get('pass_all', v.get('pass_all_base', v.get('q4_pass_all'))) for k, v in gates.items()}))


if __name__ == '__main__':
    main()
