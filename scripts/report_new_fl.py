"""Report only verified S-a runs; locked gates are in docs/research/pilot-new-fl.md."""
import csv, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('local', 'diag', 'kme_gauss', 'kme', 'dm_gauss', 'dm')
SEEDS = (42, 43)
KEYS = ('accuracy', 'precision_macro', 'precision_micro', 'precision_weighted', 'recall_macro',
        'recall_micro', 'recall_weighted', 'f1_macro', 'f1_micro', 'f1_weighted')


def main():
    data = {}
    for method in METHODS:
        for seed in SEEDS:
            run = ROOT / 'papers/new-fl/runs' / f'{method}_s{seed}'
            assert (run.resolve().parents[1] / 'VERIFIED.json').exists(), f'not yet verified: {run}'
            data[method, seed] = json.loads((run / 'result.json').read_text())
    def ev(m, s, hold='group', rule='gp'):
        return data[m, s]['history'][-1]['eval']['private'][hold][rule]
    def f1(m, s, hold='group', rule='gp'):
        return ev(m, s, hold, rule)['f1_macro']
    gates = {}
    for m, control, threshold in [('kme', 'kme_gauss', .015), ('dm', 'dm_gauss', .01)]:
        gains = [f1(m, s) - f1('local', s) for s in SEEDS]
        contrasts = [f1(m, s) - (max(f1(control, s), f1('diag', s)) if m == 'kme' else f1(control, s)) for s in SEEDS]
        benign = [ev(m, s)['benign_f1'] - ev('local', s)['benign_f1'] for s in SEEDS]
        passed = all(g >= .03 and c >= threshold and b >= -.03 and f1(m, s, 'old') > f1('local', s, 'old')
                     for s, g, c, b in zip(SEEDS, gains, contrasts, benign))
        gates[m] = dict(pass_gate=passed, gain=gains, contrast=contrasts, benign_delta=benign)
    base = ROOT / 'papers/new-fl'
    (base / 'gates.json').write_text(json.dumps(gates, indent=2) + '\n')
    lines = ['# Pilot S-a — kết quả đã verify', '', '20round,20client,DAGSNet,holdout nội bộ; không giới hạn kênh. Không chứng minh R1.', '',
             '| arm | group gp s42/s43 | head s42/s43 | Δlocal s42/s43 | payload lên/xuống mỗi client |',
             '|---|---|---|---|---|']
    for m in METHODS:
        r = data[m, 42]
        byte = [sum(b[-1] for b in r['bytes'] if b[2] == d) / 20 for d in ('up', 'down')]
        vals = ' / '.join(f'{f1(m, s):.4f}' for s in SEEDS)
        head = ' / '.join(f'{f1(m, s, rule="head"):.4f}' for s in SEEDS)
        gain = ' / '.join(f'{f1(m, s) - f1("local", s):+.4f}' for s in SEEDS)
        lines.append(f'| {m} | {vals} | {head} | {gain} | {byte[0]:.0f} / {byte[1]:.0f} B |')
    lines += ['', 'Payload một lần gồm object header32B và prior; chưa gồm frame/ACK/retransmission (S-a không kênh).', '', '## Cổng khóa', '']
    for m, g in gates.items():
        lines.append(f'- **{m}: {"PASS" if g["pass_gate"] else "FAIL"}**; Δcơ chế {g["contrast"]}; Δbenign {g["benign_delta"]}.')
    lines += ['', '## Đủ10metric, group gp r20', '', '| arm | seed | ' + ' | '.join(KEYS) + ' |', '|---|---|' + '---|' * len(KEYS)]
    with (base / 'metrics-final.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['arm', 'seed', 'holdout', 'rule', *KEYS]); writer.writeheader()
        for m in METHODS:
            for s in SEEDS:
                for hold in ('old', 'group'):
                    for rule in ('head', 'gp'):
                        vals = ev(m, s, hold, rule)['mean_metrics']
                        writer.writerow(dict(arm=m, seed=s, holdout=hold, rule=rule, **vals))
                vals = ev(m, s)['mean_metrics']
                lines.append(f'| {m} | {s} | ' + ' | '.join(f'{vals[k]:.6f}' for k in KEYS) + ' |')
    classes = json.loads((ROOT / 'knowledge/meta.json').read_text())['class_names']
    with (base / 'per-class-final.csv').open('w', newline='') as f:
        w = csv.writer(f); w.writerow(['arm', 'seed', 'class', 'group_gp_f1', 'group_head_f1'])
        for m in METHODS:
            for s in SEEDS:
                for c, name in enumerate(classes):
                    w.writerow([m, s, name, ev(m, s)['f1_per_class'][c], ev(m, s, rule='head')['f1_per_class'][c]])
    lines += ['', '## Mọi round đánh giá, đủ10metric', '',
              'Chỉ đánh giá r5/10/15/20 theo pilot khóa; round khác không có metric đánh giá. ',
              '`accuracy = precision_micro = recall_micro = recall_weighted = f1_micro` cho single-label16class.', '',
              '| arm | seed | round | holdout | rule | ' + ' | '.join(KEYS) + ' |', '|---|---|---|---|---|' + '---|' * len(KEYS)]
    with (base / 'metrics-all.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['arm', 'seed', 'round', 'holdout', 'rule', *KEYS]); w.writeheader()
        for m in METHODS:
            for s in SEEDS:
                for h in data[m, s]['history']:
                    for hold, rules in h.get('eval', {}).get('private', {}).items():
                        for rule, e in rules.items():
                            vals = e['mean_metrics']
                            w.writerow(dict(arm=m, seed=s, round=h['round'], holdout=hold, rule=rule, **vals))
                            lines.append(f'| {m} | {s} | {h["round"]} | {hold} | {rule} | ' +
                                         ' | '.join(f'{vals[k]:.6f}' for k in KEYS) + ' |')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = base / 'figures'; figures.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in METHODS:
        values = []; rounds = None
        for s in SEEDS:
            hist = [h for h in data[m, s]['history'] if 'eval' in h]
            rounds = [h['round'] for h in hist]
            values.append([h['eval']['private']['group']['gp']['f1_macro'] for h in hist])
        values = np.asarray(values)
        ax.plot(rounds, values.mean(0), marker='o', label=m)
        ax.fill_between(rounds, values.min(0), values.max(0), alpha=.12)
    ax.set(xlabel='Round', ylabel='Mean client macro-F1 (group, gp)', title='S-a: unlimited channel; band = range of two seeds')
    ax.set_xticks(rounds); ax.grid(alpha=.2); ax.legend(ncol=2); fig.tight_layout()
    fig.savefig(figures / 'convergence.png', dpi=180); fig.savefig(figures / 'convergence.svg'); plt.close(fig)
    contrast = np.asarray([np.mean([np.asarray(ev(m, s)['f1_per_class']) - np.asarray(ev('local', s)['f1_per_class']) for s in SEEDS], 0)
                           for m in METHODS[1:]])
    fig, ax = plt.subplots(figsize=(12, 4))
    limit = max(.03, float(np.abs(contrast).max()))
    im = ax.imshow(contrast, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_yticks(range(len(METHODS) - 1), METHODS[1:]); ax.set_xticks(range(16), classes, rotation=60, ha='right')
    ax.set_title('Final r20 per-class F1 gain over local; mean of two seeds, group gp')
    fig.colorbar(im, ax=ax, label='Delta F1'); fig.tight_layout()
    fig.savefig(figures / 'per-class.png', dpi=180); fig.savefig(figures / 'per-class.svg'); plt.close(fig)
    lines += ['', '## Đường học và từng lớp', '', '![Convergence](figures/convergence.png)', '',
              'Dải là min/max hai seed, không phải confidence interval. F1 là mean20private, không metric từ confusion gộp.', '',
              '![Per-class gain](figures/per-class.png)', '',
              'Delta so local cùng seed, r20 cố định; không chọn checkpoint tốt nhất. ', '',
              'Pilot dùng2,5%train và holdout receiver-group/temporal, không official test20%. Dataset có shortcut Sybil/session, '
              'benign từ run Sybil, fingerprint receiver; timeDelay có trần thông tin. Hai seed không chứng minh significance.', '',
              'Thiết lập/cài đặt/nguồn/attempt: [rebuild.md](rebuild.md). '
              'Weights, predictions và confusion từng client nằm trong [runs/](runs/). '
              'Hardware/source từng kernel đã kiểm qua output `source/`, log2×T4 và executed notebook v1.']
    (base / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:17])); print(json.dumps(gates, indent=2))


if __name__ == '__main__':
    main()
