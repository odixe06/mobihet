"""Pilot P1 table + locked gates (docs/research/pilot-plan.md §4). Reads papers/pilot/runs/*/result.json.

    python scripts/report_pilot.py [--round 20] > papers/pilot/report.md
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
SEEDS = (42, 43)


def load(runs, rnd):
    out = {}
    for f in sorted(runs.glob('*_s*/result.json')):
        r = json.loads(f.read_text()); arm, seed = r['config']['arm'], r['config']['seed']
        h = [x for x in r['history'] if x['round'] == rnd and 'eval' in x]
        if h:
            out[arm, seed] = dict(ev=h[0]['eval'], res=r)
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--round', type=int, default=20)
    p.add_argument('--runs', default=str(ROOT / 'papers/pilot/runs')); a = p.parse_args()
    D = load(Path(a.runs), a.round)
    arms = sorted({k[0] for k in D}, key=lambda s: (s != 'local', s))

    def v(arm, seed, hold='group', rule='gp', model='private', key='f1_macro'):
        e = D.get((arm, seed))
        if e is None:
            return np.nan
        m = e['ev'].get(model) or e['ev'][next(iter(e['ev']))]
        return m[hold][rule][key]

    print(f'# Pilot P1 — round {a.round}\n')
    print('Δ = arm − local, cùng seed. Model chính: private (fedavg: global). `gp` trừ khi ghi khác.\n')
    print('| arm | group gp s42 | s43 | Δ s42 | Δ s43 | old gp Δ s42 | s43 | benign F1 Δ s42 | s43 | head group s42/s43 | byte lên/xuống mỗi round | một lần lên/xuống | G1 | G2 |')
    print('|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|')
    gate = {}
    for arm in arms:
        g = [v(arm, s) for s in SEEDS]; dl = [v(arm, s) - v('local', s) for s in SEEDS]
        do = [v(arm, s, 'old') - v('local', s, 'old') for s in SEEDS]
        db = [v(arm, s, key='benign_f1') - v('local', s, key='benign_f1') for s in SEEDS]
        hd = [v(arm, s, rule='head') for s in SEEDS]
        e = D.get((arm, 42)) or D.get((arm, 43))
        br, bo = e['res']['bytes_per_client_round'], e['res']['bytes_once_per_client']
        g1 = all(x >= 0.03 for x in dl) and all(x > 0 for x in do)
        g2 = all(x >= -0.03 for x in db)
        gate[arm] = (g1, g2)
        f = lambda xs: [f'{x:+.4f}' if np.isfinite(x) else '—' for x in xs]
        print(f'| `{arm}` | ' + ' | '.join(f'{x:.4f}' for x in g) + ' | ' + ' | '.join(f(dl) + f(do) + f(db)) +
              f" | {hd[0]:.4f}/{hd[1]:.4f} | {br['up']/1e3:.1f}/{br['down']/1e3:.1f} kB | {bo['up']/1e3:.1f}/{bo['down']/1e3:.1f} kB"
              f" | {'PASS' if g1 else 'fail'} | {'PASS' if g2 else 'fail'} |")
    print('\n## Model phụ\n')
    print('| arm | model | group gp s42 | s43 | Δ local s42 | s43 |')
    print('|---|---|---:|---:|---:|---:|')
    for arm in arms:
        e = D.get((arm, 42))
        for model in (e['ev'] if e else {}):
            if model == 'private' or (arm == 'fedavg'):
                continue
            g = [v(arm, s, model=model) for s in SEEDS]
            print(f'| `{arm}` | {model} | {g[0]:.4f} | {g[1]:.4f} | {g[0]-v("local",42):+.4f} | {g[1]-v("local",43):+.4f} |')
    print('\n## Cổng riêng\n')
    pairs = [('G3', f, 'h2_bal', 0.01) for f in ('h2_full', 'h2_lr4', 'h2_diag')] + \
            [('G3', f, 'h2_mean', 0.0) for f in ('h2_full', 'h2_lr4', 'h2_diag')] + \
            [('G3', 'h2_full', 'h2_own', 0.01)] + \
            [('G4', 'dsfl', 'h2_full', 0.01), ('G4', 'fedgen', 'lg', 0.01)]
    print('| cổng | arm − đối chứng | Δ s42 | Δ s43 | ngưỡng | kết quả |')
    print('|---|---|---:|---:|---:|---|')
    for gname, x, y, th in pairs:
        d = [v(x, s) - v(y, s) for s in SEEDS]
        ok = all(t >= th for t in d) if th > 0 else all(t > 0 for t in d)
        print(f'| {gname} | `{x}` − `{y}` | {d[0]:+.4f} | {d[1]:+.4f} | {"≥" if th > 0 else ">"} {th} | {"PASS" if ok else "fail"} |')
    print('\n## Đường cong (group gp, model chính, trung bình 2 seed) và loss local\n')
    print('| arm | r5 | r10 | r15 | r20 | loss r1 | loss r20 |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for arm in arms:
        cells, l1, l20 = [], [], []
        for rr in (5, 10, 15, 20):
            xs = []
            for s in SEEDS:
                e = D.get((arm, s))
                if not e:
                    continue
                h = [x for x in e['res']['history'] if x['round'] == rr and 'eval' in x]
                if h:
                    m = h[0]['eval'].get('private') or h[0]['eval'][next(iter(h[0]['eval']))]
                    xs.append(m['group']['gp']['f1_macro'])
            cells.append(f'{np.mean(xs):.4f}' if xs else '—')
        for s in SEEDS:
            e = D.get((arm, s))
            if e:
                l1.append(e['res']['history'][0]['loss']); l20.append(e['res']['history'][-1]['loss'])
        print(f'| `{arm}` | ' + ' | '.join(cells) + f' | {np.mean(l1):.3f} | {np.mean(l20):.3f} |')
    print('\n## F1 theo lớp (group gp, trung bình 2 seed, Δ so với local)\n')
    classes = json.loads((ROOT / 'knowledge/meta.json').read_text())['class_names']
    print('| arm | ' + ' | '.join(c[:10] for c in classes) + ' |')
    print('|---|' + '---:|' * len(classes))
    base = np.mean([D[('local', s)]['ev']['private']['group']['gp']['f1_per_class'] for s in SEEDS if ('local', s) in D], 0)
    for arm in arms:
        rows = []
        for s in SEEDS:
            e = D.get((arm, s))
            if e:
                m = e['ev'].get('private') or e['ev'][next(iter(e['ev']))]
                rows.append(m['group']['gp']['f1_per_class'])
        pc = np.mean(rows, 0)
        cells = [f'{x:.2f}' for x in base] if arm == 'local' else [f'{x - b:+.2f}' for x, b in zip(pc, base)]
        print(f'| `{arm}` | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
