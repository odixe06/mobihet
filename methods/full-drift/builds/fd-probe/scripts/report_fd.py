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
from mh.metrics import METRIC_KEYS, metrics_from_confusion
from scripts.client_view import prior, view

# round 6 seed-43 pairs (design §Vòng 6): one tech may have two controls, so gate names get '_vs_<control>' on collision
S43 = [('fd_dml_m4b', 'fd_dml'), ('fd_dml_swappos', 'fd_dml')]   # last pair: round 7 (design §Vòng 7, locked ~17:29Z)
PRIMARY = {'dml': ('proxy', 'head'), 'fedavg': ('proxy', 'head'), 'gaussian': ('private', 'gp')}
# (technique, control, epoch, rule) — design.md round 1 and §Vòng 2; 15-epoch arms are keyed '<arm>@e15', seed 43 '<arm>@s43'.
GATES = [('fd_dml_prox', 'fd_dml', 6, 'gain'), ('fd_dml_avgm', 'fd_dml', 6, 'gain'), ('fd_gaussian_swappos', 'fd_gaussian', 6, 'gain'),
         ('fd_dml_q4', 'fd_dml', 6, 'noninferior'),   # round 3 (R1 bytes at one aggregation/epoch)
         ('fd_dml_m4b_q4', 'fd_dml_m4b', 6, 'noninferior'), ('fd_dml_m4b@e15', 'fd_dml@e15', 15, 'horizon'),   # round 5
         *[(t + '@s43', c + '@s43', 6, 'gain') for t, c in S43]]   # round 6
# Owner 16/10 ~03:00Z: target = client (private gp, R2). Locked ~04:00Z before any round-2 epoch-6 result (W&B r2 seen).
# DML arms mirror the proxy gate rules on private/gp; FedAvg has no private model (descriptive only).
PRIVATE_GATES = [# round 3 (design §Vòng 3, locked ~04:05Z): client-side only, no extra bytes
                 ('fd_dml_m4b', 'fd_dml', 6, 'gain'), ('fd_dml_swappos', 'fd_dml', 6, 'gain'), ('fd_dml_ntd', 'fd_dml', 6, 'gain'),
                 ('fd_dml_q4', 'fd_dml', 6, 'noninferior'),
                 # round 4 (design §Vòng 4, locked ~06:30Z)
                 ('fd_dml_hkd', 'fd_dml', 6, 'gain'), ('fd_gaussian_m4b', 'fd_gaussian', 6, 'gain'),
                 # round 5 (design §Vòng 5, locked ~08:25Z)
                 ('fd_dml_m4b@e15', 'fd_dml@e15', 15, 'horizon'), ('fd_dml_m4b_q4', 'fd_dml_m4b', 6, 'noninferior'),
                 *[(t + '@s43', c + '@s43', 6, 'gain') for t, c in S43],   # round 6 (design §Vòng 6, locked ~16:50Z)
                 ('fd_dml_m4b_q4@s43', 'fd_dml_m4b@s43', 6, 'noninferior'),   # round 10 (design §Vòng 10, locked ~22:20Z)
                 # round 11 (design §Vòng 11, locked ~06:00Z 17/10): gpool horizon
                 ('fd_dml_m4b_gpool@e15', 'fd_dml_m4b@e15', 15, 'horizon'),
                 # round 12 (design §Vòng 12, locked ~14:10Z 17/10): headline over the real int4 QPacket exchange
                 ('fd_dml_m4b_gpool_q4', 'fd_dml_m4b_gpool', 6, 'noninferior'),
                 # round 13 (design §Vòng 13, locked ~16:40Z 17/10): ternary 2-bit proxy codec
                 ('fd_dml_m4b_gpool_q2', 'fd_dml_m4b_gpool', 6, 'noninferior'),
                 # round 14 (design §Vòng 14, locked ~19:40Z 17/10): horizon / seed-43 repeats of the round-13 PASS arms
                 ('fd_dml_m4b_gpool_rej@e15', 'fd_dml_m4b@e15', 15, 'horizon'), ('fd_dml_m4b_gpool_q2@e15', 'fd_dml_m4b_gpool@e15', 15, 'noninferior')]
PRIVATE_SEEN = [('fd_dml_prox', 'fd_dml'), ('fd_dml_avgm', 'fd_dml')]   # round 1, seen before lock
# rounds 8-9 (design §Vòng 8–9, locked ~21:00Z/~21:10Z 16/10): primary private gp, gain also needs the worst client
# (global macro-F1, private gp) not below control by more than 0.010. SCAFFOLD pairs use the LR chosen on the old holdout.
R8 = [('fd_dml_m4b_kdcap', 'fd_dml_m4b'), ('fd_dml_m4b_kdcap_frz', 'fd_dml_m4b_kdcap_wu'), ('fd_dml_m4b', 'fd_dml_m4b_own'),
      ('fd_gaussian_m4b', 'fd_gaussian_m4b_own'), ('fd_dml_m4b_dyn', 'fd_dml_m4b'), ('fd_dml_swappos_cons', 'fd_dml_swappos_cons0'),
      ('fd_gaussian_m4b@s43', 'fd_gaussian@s43'), ('fd_gaussian_swappos@s43', 'fd_gaussian@s43'),
      ('fd_dml_m4b_gpool', 'fd_dml_m4b'), ('fd_dml_m4b_gpool', 'fd_dml_m4b_gown'), ('fd_dml_m4b_gpool', 'fd_dml_m4b_replay'),
      # round 10 (design §Vòng 10, locked ~22:20Z): seed-43 repeats against the audited s43 controls
      *[(t + '@s43', c + '@s43') for t, c in [('fd_dml_m4b', 'fd_dml_m4b_own'), ('fd_gaussian_m4b', 'fd_gaussian_m4b_own'),
        ('fd_dml_m4b_kdcap', 'fd_dml_m4b'), ('fd_dml_m4b_kdcap_frz', 'fd_dml_m4b_kdcap_wu'), ('fd_dml_m4b_gpool', 'fd_dml_m4b'),
        ('fd_dml_m4b_gpool', 'fd_dml_m4b_gown'), ('fd_dml_m4b_gpool', 'fd_dml_m4b_replay'), ('fd_dml_m4b_dyn', 'fd_dml_m4b')]]]
# round 11: improvements of the headline vs the audited gpool control
R8 += [(f'fd_dml_m4b_gpool_{k}', 'fd_dml_m4b_gpool') for k in ('kd2', 'kdpx', 'gw2')]
# round 13 (design §Vòng 13): A fill, B1 rej, P1 and its two controls vs the audited gpool control (gain + worst)
R8 += [(f'fd_dml_m4b_gpool_{k}', 'fd_dml_m4b_gpool') for k in ('fill', 'rej', 'p1', 'p1self', 'p1cm')]
R8 += [('fd_dml_m4b_gpool_rej@s43', 'fd_dml_m4b_gpool@s43')]   # round 14 seed-43 repeat
# P1 mechanism contrasts (read only when p1 passes its gain gate): global teacher beats the own private snapshot and the
# x-independent class-mean target by >= 0.005 private gp
R13_CONTRAST = [('fd_dml_m4b_gpool_p1', 'fd_dml_m4b_gpool_p1self'), ('fd_dml_m4b_gpool_p1', 'fd_dml_m4b_gpool_p1cm')]
# round 15 (design §Vòng 15, locked ~00:10Z 18/10): the private learns from the aggregated snapshot frozen for the epoch, so it carries
# the global prior -> primary rule head for these arms (LESSONS: deployment rule follows the teacher source); controls keep their locked
# gp rule; gain + worst at epoch 15 (horizon); the same pairs at epoch 6 are descriptive
R15 = [('fd_dml_m4b_frz@e15', 'head', 'fd_dml_m4b@e15', 'gp'), ('fd_dml_m4b_gpool_frz@e15', 'head', 'fd_dml_m4b_gpool@e15', 'gp'),
       ('fd_dml_m4b_gpool_frz_gkd@e15', 'head', 'fd_dml_m4b_gpool_frz@e15', 'head'),
       ('fd_dml_m4b_gpool_frz_gkd@e15', 'head', 'fd_dml_m4b_gpool@e15', 'gp')]
# round 16 (design §Vòng 16, locked ~04:55Z 18/10): seed-43 horizon repeats of the round-15 PASS arms (head) vs seed-43 controls (gp), and the
# gkd headline over the ternary 2-bit proxy wire vs dense gkd (noninferior, both head); epoch 15
R16 = [('fd_dml_m4b_gpool_frz_gkd@e15@s43', 'head', 'fd_dml_m4b_gpool@e15@s43', 'gp', 'gain'),
       ('fd_dml_m4b_gpool_frz_gkd@e15@s43', 'head', 'fd_dml_m4b@e15@s43', 'gp', 'gain'),
       ('fd_dml_m4b_frz@e15@s43', 'head', 'fd_dml_m4b@e15@s43', 'gp', 'gain'),
       ('fd_dml_m4b_gpool_q2_frz_gkd@e15', 'head', 'fd_dml_m4b_gpool_frz_gkd@e15', 'head', 'noninferior')]
# round 17 (design §Vòng 17, locked ~06:25Z 18/10; owner ~06Z: primary rule = the more accurate one -> head for the frz family): small
# improvements of the gkd headline vs gkd itself, both head, epoch 15, gain + worst (same loop as R16)
R16 += [(f'fd_dml_m4b_gpool_frz_gkd_{t}@e15', 'head', 'fd_dml_m4b_gpool_frz_gkd@e15', 'head', 'gain') for t in ('la', 'kdpx', 'la_kdpx')]
# round 18 (design §Vòng 18, locked ~07:30Z 18/10): one evidence-backed change each on gkd, same gate
R16 += [(f'fd_dml_m4b_gpool_frz_gkd_{t}@e15', 'head', 'fd_dml_m4b_gpool_frz_gkd@e15', 'head', 'gain') for t in ('cop', 'pce', 't2', 'g512', 'gk2')]
R16 += [('fd_dml_m4b_gpool_q4_frz_gkd@e15', 'head', 'fd_dml_m4b_gpool_frz_gkd@e15', 'head', 'noninferior')]
# round 19 (design §Vòng 19, locked ~14:35Z 18/10; owner ~09:15Z: --fast arms only vs --fast controls): end-to-end equivalence of the gkd
# control re-run with --fast, la_kdpx vs the --fast control (s42 and s43), la_kdpx over int4 vs la_kdpx (noninferior); all head, epoch 15
R16 += [('fd_dml_m4b_gpool_frz_gkd@e15@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd@e15', 'head', 'equiv'),
        ('fd_dml_m4b_gpool_frz_gkd@e15@s43@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd@e15@s43', 'head', 'equiv'),
        ('fd_dml_m4b_gpool_frz_gkd_la_kdpx@e15@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd@e15@fast', 'head', 'gain'),
        ('fd_dml_m4b_gpool_frz_gkd_la_kdpx@e15@s43@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd@e15@s43@fast', 'head', 'gain'),
        ('fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx@e15@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx@e15@fast', 'head', 'noninferior')]
# round 20 (design §Vòng 20, locked ~08:30Z 19/10): each la/kdpx variant vs la_kdpx --fast s42 (tb), head, epoch 15, gain + worst
R16 += [(f'fd_dml_m4b_gpool_frz_gkd_{t}@e15@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx@e15@fast', 'head', 'gain')
        for t in ('la15_kdpx', 'la05_kdpx', 'lau_kdpx', 'la_lap_kdpx', 'la_kdpx_kd2')]
# round 21 (design §Vòng 21): compressed wires vs dense la_kdpx --fast s42 (tb): noninferior (performance kept), bytes reported
R16 += [(f'fd_dml_m4b_gpool_{t}@e15@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx@e15@fast', 'head', 'noninferior')
        for t in ('q4_frz_gkd_la_kdpx_zl', 'q4_frz_gkd_la_kdpx_ef_zl', 'q2_frz_gkd_la_kdpx_ef_zl')]
# round 22 (design §Vòng 22): 3-bit / 2-bit-uplink wires vs tb (noninferior); q4_zl on seed 43 vs la_kdpx --fast s43 (te)
R16 += [(f'fd_dml_m4b_gpool_{t}@e15@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx@e15@fast', 'head', 'noninferior')
        for t in ('q3_frz_gkd_la_kdpx_zl', 'q4_u2_frz_gkd_la_kdpx_ef_zl', 'q3_u2_frz_gkd_la_kdpx_ef_zl')]
R16 += [('fd_dml_m4b_gpool_q4_frz_gkd_la_kdpx_zl@e15@s43@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx@e15@s43@fast', 'head', 'noninferior')]
# round 23 (design §Vòng 23): q3 on seed 43 vs te; 2-bit with the RMS step vs tb (noninferior)
R16 += [('fd_dml_m4b_gpool_q3_frz_gkd_la_kdpx_zl@e15@s43@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx@e15@s43@fast', 'head', 'noninferior'),
        ('fd_dml_m4b_gpool_q2_rs_frz_gkd_la_kdpx_zl@e15@fast', 'head', 'fd_dml_m4b_gpool_frz_gkd_la_kdpx@e15@fast', 'head', 'noninferior')]
R8_SEEN = [('fd_dml_m4b_kdcap_frz', 'fd_dml_m4b_kdcap'), ('fd_dml_m4b_kdcap_frz', 'fd_dml_m4b'), ('fd_dml_m4b_kdcap_wu', 'fd_dml_m4b_kdcap'),
           ('fd_dml_swappos_cons', 'fd_dml_m4b'), ('fd_dml_m4b@r10', 'fd_dml_m4b'),
           ('fd_dml_m4b_kdcap_frz@s43', 'fd_dml_m4b_kdcap@s43'), ('fd_dml_m4b_kdcap_wu@s43', 'fd_dml_m4b_kdcap@s43'),
           *[(f'fd_dml_m4b_{k}{lr}', 'fd_dml_m4b') for k in ('sgd', 'scaf') for lr in (1, 3, 10)]]
RULES = {'gain': lambda d, b: d >= .010 and b >= -.005, 'noninferior': lambda d, b: d >= -.010 and b >= -.010,
         'benign': lambda d, b: b >= .010 and d >= -.005, 'horizon': lambda d, b: d >= .010,
         'equiv': lambda d, b: abs(d) <= .010}   # round 19: --fast control vs old-code control within the seed spread (0.008)


def runs():
    found, manifests = {}, {}
    for marker in sorted((BASE / 'artifacts/kaggle').glob('fd-s4[23]*/AUDITED.json')):
        for run in sorted((marker.parent / 'runs').glob('fd_*_s4[23]*')):
            rows = [json.loads(m.read_text()) for m in sorted((run / 'complete').glob('round_*.json'))]
            arm, seed, epochs = re.fullmatch(r'(fd_.+)_s(4[23])(?:_(e15|e50))?', run.name).groups()   # e50: continued via --parent
            if {'sub4', 'sub16'} & set(arm.split('_')): continue   # one aggregation per full local epoch only (owner 17/10)
            key = arm + (f'@{epochs}' if epochs else '') + ('@s43' if seed == '43' else '')   # seed 42 keys unchanged
            if json.loads((run / 'config.json').read_text())['config'].get('fast'): key += '@fast'   # owner 18/10: --fast vs --fast only
            if json.loads((run / 'config.json').read_text())['config'].get('final'): key += '@final'   # owner 19/10: final protocol
            if key in found: key += '@' + marker.parent.name.split('-')[3]   # rerun of an audited arm: '<arm>@<round tag>' (y: '@r10')
            assert key not in found, ('duplicate arm across sessions', key)
            if rows: found[key], manifests[key] = rows, marker.parent / 'source/data_manifest.json'
    return found, manifests


def primary(arm):
    base = arm.split('@')[0]
    return ('private', 'gp') if base.endswith('_own') else PRIMARY[base.split('_')[1]]


def worst(rows, role='private', rule='gp', hold='group', epoch=6):
    row = next((r for r in rows if r['round'] == epoch), None)
    if not row or role not in row['eval']: return None
    return min(metrics_from_confusion(__import__('numpy').array(c))['f1_macro'] for c in row['eval'][role][hold][rule]['confusion_clients'])


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
    sgd = {lr: score(data.get(f'fd_dml_m4b_sgd{lr}', []), 'proxy', 'head', hold='old') for lr in (1, 3, 10)}
    pairs = list(R8)
    if all(sgd.values()):
        best = max(sgd, key=lambda lr: sgd[lr][0])
        gates['scaffold_lr_choice'] = dict(descriptive=True, rule='argmax old-holdout proxy head ep6 of sgd*', lr=best / 100,
                                           old_proxy={lr / 100: v[0] for lr, v in sgd.items()})
        pairs += [(f'fd_dml_m4b_scaf{best}', f'fd_dml_m4b_sgd{best}'), (f'fd_dml_m4b_scaf{best}', 'fd_dml_m4b')]
    for tech, ctl in pairs:
        a, b = score(data.get(ctl, []), 'private', 'gp'), score(data.get(tech, []), 'private', 'gp')
        wa, wb = worst(data.get(ctl, [])), worst(data.get(tech, []))
        pa, pb = score(data.get(ctl, []), 'proxy', 'head'), score(data.get(tech, []), 'proxy', 'head')
        name = f'{tech}:private_gp' if f'{tech}:private_gp' not in gates else f'{tech}:private_gp_vs_{ctl}'
        gates[name] = dict(control=ctl, epoch=6, rule='gain+worst', primary='private/gp',
                           delta=b[0] - a[0] if a and b else None, benign_delta=b[1] - a[1] if a and b else None,
                           worst_delta=wb - wa if wa is not None and wb is not None else None,
                           proxy_delta=pb[0] - pa[0] if pa and pb else None,
                           promising=bool(RULES['gain'](b[0] - a[0], b[1] - a[1]) and wb - wa >= -.010) if a and b else None)
    for tech, ctl in R13_CONTRAST:
        a, b = score(data.get(ctl, []), 'private', 'gp'), score(data.get(tech, []), 'private', 'gp')
        gates[f'{tech}:private_gp_vs_{ctl}'] = dict(control=ctl, epoch=6, rule='contrast >= +0.005', primary='private/gp',
                                                   delta=b[0] - a[0] if a and b else None,
                                                   benign_delta=b[1] - a[1] if a and b else None,
                                                   promising=b[0] - a[0] >= .005 if a and b else None)
    for arm, rows in data.items():   # round 13 C: gpe rule vs the locked gp rule on the same private weights (gain + worst)
        a, b = score(rows, 'private', 'gp'), (score(rows, 'private', 'gpe') if rows and 'gpe' in rows[-1]['eval'].get('private', {}).get('group', {}) else None)
        if a and b:
            wa, wb = worst(rows), worst(rows, rule='gpe')
            gates[f'{arm}:private_gpe_vs_gp'] = dict(control='private/gp', epoch=6, rule='gain+worst', delta=b[0] - a[0],
                                                     benign_delta=b[1] - a[1], worst_delta=wb - wa,
                                                     promising=bool(RULES['gain'](b[0] - a[0], b[1] - a[1]) and wb - wa >= -.010))
    for tech, trule, ctl, crule in R15:
        for epoch in (15, 6):
            a, b = score(data.get(ctl, []), 'private', crule, epoch=epoch), score(data.get(tech, []), 'private', trule, epoch=epoch)
            wa, wb = worst(data.get(ctl, []), rule=crule, epoch=epoch), worst(data.get(tech, []), rule=trule, epoch=epoch)
            pa, pb = score(data.get(ctl, []), 'proxy', 'head', epoch=epoch), score(data.get(tech, []), 'proxy', 'head', epoch=epoch)
            gates[f'{tech}:private_{trule}_vs_{ctl}_{crule}@ep{epoch}'] = dict(
                control=f'{ctl} private/{crule}', epoch=epoch, rule='gain+worst' if epoch == 15 else 'descriptive', primary=f'private/{trule}',
                delta=b[0] - a[0] if a and b else None, benign_delta=b[1] - a[1] if a and b else None,
                worst_delta=wb - wa if wa is not None and wb is not None else None, proxy_delta=pb[0] - pa[0] if pa and pb else None,
                promising=(bool(RULES['gain'](b[0] - a[0], b[1] - a[1]) and wb - wa >= -.010) if a and b else None) if epoch == 15 else None)
    for tech, trule, ctl, crule, kind in R16:
        a, b = score(data.get(ctl, []), 'private', crule, epoch=15), score(data.get(tech, []), 'private', trule, epoch=15)
        wa, wb = worst(data.get(ctl, []), rule=crule, epoch=15), worst(data.get(tech, []), rule=trule, epoch=15)
        pa, pb = score(data.get(ctl, []), 'proxy', 'head', epoch=15), score(data.get(tech, []), 'proxy', 'head', epoch=15)
        ok = (lambda: RULES[kind](b[0] - a[0], b[1] - a[1]) and (kind != 'gain' or wb - wa >= -.010))
        gates[f'{tech}:private_{trule}_vs_{ctl}_{crule}@ep15'] = dict(
            control=f'{ctl} private/{crule}', epoch=15, rule=kind + ('+worst' if kind == 'gain' else ''), primary=f'private/{trule}',
            delta=b[0] - a[0] if a and b else None, benign_delta=b[1] - a[1] if a and b else None,
            worst_delta=wb - wa if wa is not None and wb is not None else None, proxy_delta=pb[0] - pa[0] if pa and pb else None,
            promising=bool(ok()) if a and b else None)
    for tech, ctl in R8_SEEN:
        a, b = score(data.get(ctl, []), 'private', 'gp'), score(data.get(tech, []), 'private', 'gp')
        if a and b and f'{tech}:private_gp_vs_{ctl}' not in gates:   # never overwrite a locked gate of the same pair
            gates[f'{tech}:private_gp_vs_{ctl}'] = dict(descriptive=True, delta=b[0] - a[0], benign_delta=b[1] - a[1])
    a, b = score(data.get('fd_dml@e15@s43', []), 'private', 'gp', epoch=15), score(data.get('fd_dml_m4b@e15@s43', []), 'private', 'gp', epoch=15)
    gates['fd_dml_m4b@e15@s43:private_gp'] = dict(control='fd_dml@e15@s43', epoch=15, rule='horizon', delta=b[0] - a[0] if a and b else None,
                                                  promising=bool(RULES['horizon'](b[0] - a[0], 0)) if a and b else None)
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
