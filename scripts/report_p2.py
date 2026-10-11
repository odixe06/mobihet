"""P2 table + locked gates (docs/research/pilot-plan.md §5). Reads papers/p2/runs/*/result.json and the P1 runs as reference.

    python scripts/report_p2.py > papers/p2/report.md
P2-a: (arm - P1 local) >= 0.70 * (P1 same form - P1 local), group gp, both seeds.
P2-b: h6 - naive >= +0.01 both seeds, where naive kept < 70%. DML also reported with the `head` rule.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43)


def load(d):
    out = {}
    for f in sorted(Path(d).glob('*_s*/result.json')):
        r = json.loads(f.read_text()); out[r['config']['arm'], r['config']['seed']] = r
    return out


def final(r, rule='gp', model='private', hold='group'):
    e = [h for h in r['history'] if 'eval' in h][-1]['eval']
    return e[model][hold][rule]['f1_macro']


def main():
    P1, P2 = load(ROOT / 'papers/pilot/runs'), load(ROOT / 'papers/p2/runs')
    print('# Pilot P2 — kênh B0/B2 (round 20)\n')
    print('Δ = arm − P1 `local` cùng seed. Giữ = Δ / (P1 cùng dạng − P1 local). `group`.\n')
    print('| arm | gp s42 | s43 | giữ s42 | s43 | P2-a | head s42/s43 | proxy s42/s43 | byte dùng/client/round | xe đã kích hoạt r20 |')
    print('|---|---:|---:|---:|---:|---|---|---|---|---|')
    keep = {}
    for arm in sorted({a for a, _ in P2}):
        ref = arm.split('_B')[0]
        cells, ks = [], []
        for s in SEEDS:
            r = P2.get((arm, s))
            if r is None:
                cells.append('—'); ks.append(np.nan); continue
            base = final(P1[('local', s)]); full = final(P1[(ref, s)]) - base
            cells.append(f'{final(r):.4f}'); ks.append((final(r) - base) / full)
        keep[arm] = ks
        ok = all(np.isfinite(k) and k >= 0.70 for k in ks)
        rr = [P2.get((arm, s)) for s in SEEDS]
        head = '/'.join(f'{final(r, "head"):.4f}' if r else '—' for r in rr)
        prox = '/'.join(f'{final(r, "head", "proxy"):.4f}' if r and 'proxy' in r['history'][-1]['eval'] else '—' for r in rr)
        used = [np.mean([h['bytes_used'] for h in r['history']]) / len(r['train_rows']) for r in rr if r]
        act = '/'.join(str(r['history'][-1]['active']) if r else '—' for r in rr)
        print(f'| `{arm}` | {cells[0]} | {cells[1]} | ' + ' | '.join(f'{k:.0%}' if np.isfinite(k) else '—' for k in ks) +
              f" | {'PASS' if ok else 'fail'} | {head} | {prox} | {np.mean(used)/1e3:.1f} kB | {act} |")
    print('\n## P2-b (h6 − naive)\n')
    print('| arm | Δ s42 | Δ s43 | naive giữ < 70%? | P2-b |')
    print('|---|---:|---:|---|---|')
    for arm in sorted(a for a in keep if a.endswith('_h6')):
        nv = arm[:-3] + '_naive'
        if nv not in keep:
            continue
        d = [final(P2[(arm, s)]) - final(P2[(nv, s)]) if (arm, s) in P2 and (nv, s) in P2 else np.nan for s in SEEDS]
        lost = any(np.isfinite(k) and k < 0.70 for k in keep[nv])
        ok = lost and all(np.isfinite(x) and x >= 0.01 for x in d)
        print(f'| `{arm}` | {d[0]:+.4f} | {d[1]:+.4f} | {"có" if lost else "không"} | {"PASS" if ok else ("n/a" if not lost else "fail")} |')


if __name__ == '__main__':
    main()
