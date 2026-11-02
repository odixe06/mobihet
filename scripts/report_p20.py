"""Render verified P20 tables and standalone figures; no training or parameter selection."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ROOT / 'runs/p-study/analysis'
    result = json.loads((base / 'analysis.json').read_text())
    arms = result['arms']
    docs = ROOT / 'docs/research'
    figures = docs / 'figures'
    figures.mkdir(exist_ok=True)
    costs = []
    for name, arm in arms.items():
        costs.append({'arm': name, 'mode': arm['mode'], 'seed': arm['seed'], **arm['bytes'], **arm['timing']})
    with (base / 'costs.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(costs[0])); writer.writeheader(); writer.writerows(costs)
    lines = ['# P20 — byte và thời gian', '',
        'Byte tính mỗi xe, bootstrap + upload/download20round. Payload protocol, chưa transport/ACK/crypto/retransmission.',
        'Thời gian giây của cả run mô phỏng5343xe/2T4; không latency từng xe hoặc thời gian gửi qua mạng thật.', '',
        '| arm | bootstrap up | bootstrap down | round up tổng | round down tổng | tổng B/xe |', '|---|---:|---:|---:|---:|---:|']
    for row in costs:
        lines.append('| ' + row['arm'] + ' | ' + ' | '.join(str(int(row[k])) for k in ['bootstrap_up', 'bootstrap_down', 'rounds_up', 'rounds_down', 'total']) + ' |')
    lines += ['', 'Scoring/server/bank nằm trong train, không cộng lần hai. Phần chưa phân bổ gồm decode/setup/compile/checkpoint/finalization và lệch mốc log; không gọi là setup đo chính xác.', '',
              '| arm | train | eval | diagnostic | log elapsed | chưa phân bổ | server (trong train) | scoring local rank0 (trong train) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in costs:
        lines.append('| ' + row['arm'] + ' | ' + ' | '.join(f'{row[k]:.2f}' for k in ['t_train', 't_eval', 't_diagnostics', 'notebook_log_elapsed_seconds', 'unattributed_seconds', 'server_seconds', 'scoring_seconds_local']) + ' |')
    lines += ['', 'P21 lỗi trước round1, không ghép vào chi phí20round. Không có measured GPUkernel-time/utilization hoặc bandwidth/contactlatency.',
              'Full CSV: `runs/p-study/analysis/costs.csv`; [10metric](metrics-p20.md), [kết quả](results-p20.md).']
    (docs / 'cost-p20.md').write_text('\n'.join(lines) + '\n')
    table = ['| cell | pooled F1 s42/s43 | mean F1 xe<256 s42/s43 | total MB/xe/20r |', '|---|---|---|---:|']
    for cell in range(16):
        pair = [arms.get(f'P{2 * cell + i}') for i in (0, 1)]
        def values(key): return '/'.join(f"{a['final'][key]:.6f}" if a else 'thiếu' for a in pair)
        present = next(a for a in pair if a)
        table.append(f"| C{cell} {present['mode']} | {values('f1')} | {values('low256_mean')} | {present['bytes']['total'] / 1e6:.6f} |")
    (base / 'endpoint-table.md').write_text('\n'.join(table) + '\n')
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    cells = [0, 1, 2, 4, 7, 8, 9]
    colors = dict(zip(cells, plt.cm.tab10.colors))
    for cell in cells:
        pair = [arms[f'P{2 * cell + i}'] for i in (0, 1)]
        x = np.arange(1, 21)
        for ax, metric in zip(axes[0], ['f1', 'low256_mean']):
            y = np.array([[p[metric] for p in a['curve']] for a in pair])
            ax.plot(x, y.mean(0), label=f'C{cell} {pair[0]["mode"]}', color=colors[cell])
            ax.fill_between(x, y.min(0), y.max(0), color=colors[cell], alpha=.12)
            ax.set_xlabel('Round (cosine schedule50)'); ax.grid(alpha=.2)
    axes[0, 0].set(title='Own-test pooled macro F1', ylabel='F1')
    axes[0, 0].legend(fontsize=8, loc='lower right')
    axes[0, 1].set(title='848 clients with <256 train rows', ylabel='Mean per-client F1')
    labels = ['augmentation', 'soft_class_Q', 'binary_real_KD', 'margin_vs_strength', 'input_vs_class_same_bank', 'server_proxy_fit', 'streaming_proxy_extrapolation']
    for i, label in enumerate(labels):
        selected = [e for e in result['effects'] if e['label'] == label]
        axes[1, 0].scatter([e['f1'] for e in selected], [e['low256_mean'] for e in selected], label=label, color=plt.cm.tab10(i))
    axes[1, 0].axhline(0, c='grey', lw=.7); axes[1, 0].axvline(0, c='grey', lw=.7)
    axes[1, 0].set(title='Locked paired contrasts (each dot = one seed)', xlabel='Delta pooled F1', ylabel='Delta mean F1, clients<256')
    axes[1, 0].legend(fontsize=7, loc='upper left'); axes[1, 0].grid(alpha=.2)
    names = [name for name, a in arms.items() if a['teacher_probe']]
    x = np.arange(len(names)); ax = axes[1, 1]
    teacher = [arms[name]['teacher_probe']['20']['test']['all']['teacher']['f1_macro'] for name in names]
    student = [arms[name]['teacher_probe']['20']['test']['all']['student']['f1_macro'] for name in names]
    ax.bar(x - .18, teacher, width=.36, label='Server proxy'); ax.bar(x + .18, student, width=.36, label='Private student')
    ax.set_xticks(x, names); ax.set(title='Same rows: 34 diagnostic clients, r20', ylabel='Pooled test F1 (diagnostic only)'); ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.suptitle('P20: fixed endpoint20; bands = two-seed range, not confidence intervals')
    fig.tight_layout()
    for ext in ['png', 'pdf']: fig.savefig(figures / ('p20-summary.' + ext), dpi=180, bbox_inches='tight')
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    offsets = {1:(8,-15),2:(8,7),4:(8,23),7:(8,2),8:(8,-15),9:(-45,-5),14:(-45,7)}
    for cell in [1, 2, 4, 7, 8, 9, 14]:
        pair = [arms[f'P{2 * cell + i}'] for i in (0, 1)]
        x = pair[0]['bytes']['total'] / 1e6
        ys = [a['final']['f1'] for a in pair]
        ax.scatter([x, x], ys, s=30)
        ax.annotate(f'C{cell}', (x, np.mean(ys)), xytext=offsets[cell], textcoords='offset points')
    ax.set(xlabel='Total payload MB/client/20round (bootstrap included)', ylabel='Own-test pooled macro F1',
           title='P20 communication tradeoff: selected copula cells, both seeds')
    ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(figures / 'p20-cost.png', dpi=180)
    plt.close(fig)
    print('P20_REPORT_TABLES_FIGURES_DONE')


if __name__ == '__main__': main()
