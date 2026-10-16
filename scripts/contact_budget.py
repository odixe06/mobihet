"""R1 contact budget: how many RSU contacts fit in one local epoch, and does one exchange fit in one contact?

Measured inputs (no training): vehicle speeds from the VeReMi NextGen InTAS ground truth (test files carry spd/pos),
per-client epoch seconds on a T4 and the q4 proxy payload from an audited FullDrift marker. Assumed inputs are swept,
not fixed: RSU spacing along the route, coverage radius, link rate, on-board compute slowdown vs T4.

    python scripts/contact_budget.py --gt "/mnt/g/My Drive/NCKH/veremi_nextgen/data/ground_truth" \
        --marker methods/full-drift/builds/fd-probe/artifacts/kaggle/fd-s42j-20261015-r5-v1/runs/fd_dml_m4b_q4_s42/complete/round_006.json

contacts/epoch = f * T_epoch / (spacing / v); contact seconds = (pi r / 2) / v (mean chord of a disk crossed by a
straight route through it); one exchange = proxy down + proxy up. Limits: straight-road RSU line, no queueing/MAC loss,
no waiting for the other 19 clients (synchronous cohort), speeds of stopped vehicles excluded via the median.
"""
import argparse
import json
import math
from pathlib import Path
import statistics as st

SPACING_M = (500, 1000, 2000, 5000)        # RSU every ... metres along the route
RADIUS_M = (150, 300, 700)                 # coverage radius (field 802.11p ~138-700 m, design target ~1000 m)
RATE_BPS = (80e3, 750e3)                   # old simulator max 80 kB/s; 6 Mbps PHY = 750 kB/s
SLOWDOWN = (1, 5, 20)                      # on-board epoch time / T4 epoch time
DENSE_OVER_Q4 = 7.65                       # QProxy q4 byte ratio (AntiMem, audited)


def speeds(gt):
    out = {}
    for f in sorted(Path(gt).glob('InTAS_*_Test_ground_truth.json')):
        per = {}
        for m in json.loads(f.read_text()):
            if 'spd' in m: per.setdefault(m['sender_id'], []).append(float(m['spd']))
        out[f.name.split('_Test')[0][6:]] = st.median(st.median(v) for v in per.values())
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gt', required=True); p.add_argument('--marker', required=True)
    a = p.parse_args()
    marker = json.loads(Path(a.marker).read_text())
    t_epoch = st.median(c['seconds'] for c in marker['clients'])
    q4 = 2 * marker['diagnostics']['q_bytes_down']               # one exchange: down + up, same int4 size
    payload = {'q4': q4, 'dense': q4 * DENSE_OVER_Q4}
    v = speeds(a.gt)
    print(f'T4 epoch/client median {t_epoch:.1f} s; exchange bytes q4 {q4/1e3:.0f} kB, dense {payload["dense"]/1e6:.2f} MB')
    print('median vehicle speed m/s', {k: round(x, 1) for k, x in v.items()})
    vm = st.median(v.values())
    print(f'\n## contacts per local epoch (v = {vm:.1f} m/s)\n\n| RSU spacing | ' +
          ' | '.join(f'slowdown {f}x' for f in SLOWDOWN) + ' |\n|---|' + '---:|' * len(SLOWDOWN))
    for s in SPACING_M:
        print(f'| {s} m | ' + ' | '.join(f'{f * t_epoch / (s / vm):.1f}' for f in SLOWDOWN) + ' |')
    print(f'\n## one exchange per contact? (contact s = pi r/2 / v; capacity = rate x s)\n\n'
          '| radius | contact s | rate | capacity | q4 fits | dense fits |\n|---|---:|---|---:|---|---|')
    for r in RADIUS_M:
        sec = math.pi * r / 2 / vm
        for rate in RATE_BPS:
            cap = rate * sec
            print(f'| {r} m | {sec:.0f} | {rate/1e3:.0f} kB/s | {cap/1e6:.2f} MB | '
                  f'{cap / payload["q4"]:.1f}x | {cap / payload["dense"]:.1f}x |')


if __name__ == '__main__':
    main()
