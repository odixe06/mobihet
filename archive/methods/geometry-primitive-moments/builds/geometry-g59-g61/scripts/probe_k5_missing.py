#!/usr/bin/env python3
"""Bounded synthetic-only fidelity probe against the original feature extractor."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
SOURCE = ROOT.parent / 'dataset/script/preprocessed/vnx_features.py'
OUTPUT = ROOT / 'proposals/geometry-primitive-moments/fidelity-missing.json'
spec = importlib.util.spec_from_file_location('primary_k5_features', SOURCE)
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)

NOMINAL_REMOVE = (
    'f_dx', 'f_dy', 'f_dist', 'f_bearing_sin', 'f_bearing_cos',
    'f_hed_diff_cos', 'f_rcv_vx', 'f_rcv_vy', 'f_snd_vx', 'f_snd_vy',
    'f_rel_speed', 'f_closing_speed', 'f_spd_diff',
    'f_rcv_noise_mag', 'f_snd_noise_mag',
)
# Conservative policy: no missingness inference from redundant derived fields.
RETAIN = (
    'f_dx', 'f_dy', 'f_dist', 'f_rel_speed', 'f_closing_speed',
    'f_spd_diff', 'f_rcv_noise_mag', 'f_snd_noise_mag',
)
ELIGIBLE = tuple(name for name in NOMINAL_REMOVE if name not in RETAIN)
ELIGIBLE61 = tuple(name for name in ELIGIBLE if name not in ('f_bearing_sin', 'f_bearing_cos'))
STRUCTURAL = extractor.ZERO_IMPUTED_FEATURES


def base():
    row = {column: 1.0 for _, column in extractor.RAW_SOURCES}
    row.update(
        rcv_pos_x=0.0, rcv_pos_y=0.0, snd_pos_x=3.0, snd_pos_y=4.0,
        rcv_spd=2.0, snd_spd=3.0, rcv_hed=0.0, snd_hed=0.0,
        rcv_time_ns=1_000_000_000, send_time_ns=900_000_000,
        message_id=1.0, receiver_id='toy-receiver', sender_alias='toy-alias',
        rcv_profile='NORMAL', snd_profile='NORMAL',
    )
    return row


def features(row, two_rows=False):
    rows = [row]
    if two_rows:
        following = dict(row)
        following.update(rcv_time_ns=2_000_000_000, send_time_ns=1_900_000_000,
                         message_id=2.0)
        rows.append(following)
    table = pa.table({key: [item[key] for item in rows] for key in row})
    return extractor.impute_zero(extractor.feature_matrix(table, 0.0, 0.0))


def reconstruct(values):
    result = {}
    for side in ('rcv', 'snd'):
        result[f'f_{side}_vx'] = values[f'f_{side}_spd'] * values[f'f_{side}_hed_sin']
        result[f'f_{side}_vy'] = values[f'f_{side}_spd'] * values[f'f_{side}_hed_cos']
    result['f_hed_diff_cos'] = (
        values['f_snd_hed_sin'] * values['f_rcv_hed_sin']
        + values['f_snd_hed_cos'] * values['f_rcv_hed_cos']
    )
    with np.errstate(invalid='ignore', divide='ignore'):
        inv = np.where(values['f_dist'] > 0, 1.0 / values['f_dist'], np.nan)
        result['f_bearing_sin'] = values['f_dx'] * inv
        result['f_bearing_cos'] = values['f_dy'] * inv
    return extractor.impute_zero(result)


def same_primitives(a, b, remove):
    return all(np.array_equal(a[name], b[name])
               for name in extractor.FEATURE_COLUMNS if name not in remove)


cases = [('finite', base())]
for side in ('rcv', 'snd'):
    for heading in (0.0, 90.0, 180.0, 270.0):
        row = base(); row[f'{side}_hed'] = heading
        cases.append((f'compass_{side}_{int(heading)}', row))
coincident = base(); coincident.update(snd_pos_x=0.0, snd_pos_y=0.0)
cases.append(('coincident', coincident))
unknown = base(); unknown.update(rcv_profile='UNKNOWN', snd_profile=None)
cases.append(('unknown_profiles', unknown))
for field in ('rcv_pos_x', 'rcv_pos_y', 'snd_pos_x', 'snd_pos_y',
              'rcv_spd', 'snd_spd', 'rcv_hed', 'snd_hed',
              'rcv_pos_noise_x', 'rcv_pos_noise_y',
              'snd_pos_noise_x', 'snd_pos_noise_y'):
    for name, value in (('nan', np.nan), ('posinf', np.inf), ('neginf', -np.inf)):
        row = base(); row[field] = value
        cases.append((f'{field}_{name}', row))

case_results = []
with np.errstate(all='ignore'):
    for name, row in cases:
        observed = features(row, two_rows=True)
        rebuilt = reconstruct(observed)
        failed = [feature for feature in ELIGIBLE
                  if not np.allclose(rebuilt[feature], observed[feature],
                                     rtol=1e-12, atol=1e-12)]
        finite = all(np.isfinite(value).all() for value in observed.values())
        first = observed['f_first_in_session'] == 1
        structural = all((observed[feature][first] == 0).all()
                         for feature in STRUCTURAL)
        case_results.append(dict(case=name, rows=2, first_rows=int(first.sum()),
                                 nonfirst_rows=int((~first).sum()), finite=finite,
                                 first_structural_pass=structural,
                                 eligible_reconstruction_pass=not failed,
                                 g61_reconstruction_pass=not any(feature in ELIGIBLE61 for feature in failed),
                                 failed_formulas=failed))

    counterexamples = []
    for field, receiver_position in (
        ('snd_spd', 0.0), ('rcv_spd', 0.0),
        ('snd_pos_noise_x', 0.0), ('rcv_pos_noise_x', 0.0),
        ('snd_pos_x', 2.0), ('snd_pos_y', 2.0),
        ('snd_pos_x', 0.0),
    ):
        worlds = []
        for value in (np.nan, 0.0):
            row = base(); row[field] = value
            row.update(rcv_pos_x=receiver_position, rcv_pos_y=receiver_position)
            worlds.append(features(row))
        changed = [feature for feature in NOMINAL_REMOVE
                   if not np.array_equal(worlds[0][feature], worlds[1][feature])]
        removed53 = tuple(feature for feature in NOMINAL_REMOVE
                          if feature not in ('f_dx', 'f_dy'))
        counterexamples.append(dict(
            case=f'{field}_nan_vs_valid_zero_receiver_{int(receiver_position)}',
            g51_same_primitives=same_primitives(*worlds, NOMINAL_REMOVE),
            g53_same_primitives=same_primitives(*worlds, removed53),
            different_derived_features=changed,
        ))

primitive51 = tuple(feature for feature in extractor.FEATURE_COLUMNS
                    if feature not in NOMINAL_REMOVE or feature in RETAIN)
primitive53 = tuple(feature for feature in extractor.FEATURE_COLUMNS
                    if feature not in NOMINAL_REMOVE or feature in RETAIN
                    or feature in ('f_dx', 'f_dy'))
assert primitive51 == primitive53 and len(primitive51) == 59
assert all(item['finite'] and item['first_structural_pass']
           and item['eligible_reconstruction_pass'] for item in case_results)
for name in ('g51', 'g53'):
    assert any(item[f'{name}_same_primitives'] and item['different_derived_features']
               for item in counterexamples)

result = dict(
    probe='K5 primary extractor synthetic missing/nonfinite/coincident fidelity',
    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    source=str(SOURCE.relative_to(ROOT.parent)),
    real_records_used=0, learned_models_used=0, dtype='float64',
    comparison=dict(rtol=1e-12, atol=1e-12),
    case_count=len(cases), toy_rows=sum(item['rows'] for item in case_results),
    finite_and_eligible_toy_pass=True,
    nominal_g51_fidelity_pass=False, nominal_g53_fidelity_pass=False,
    conservative_retention=RETAIN, eligible_derived=ELIGIBLE,
    actual_primitive_dimension=59, nominal_designs_collide=True,
    g61_candidate=dict(
        status='owner selected both G59 and G61 quality experiments; toy only',
        actual_primitive_dimension=61,
        eligible_derived=ELIGIBLE61,
        all_toy_cases_pass=all(item['g61_reconstruction_pass'] for item in case_results),
        retained_bearings=True,
        inverse_distance_used=False,
    ),
    cases=case_results, indistinguishability_counterexamples=counterexamples,
    limits=[
        'Synthetic toy only; no actual dataset branch-frequency or F1 evidence.',
        'Eight retained fields follow conservative no-missingness-inference policy; not a proof of globally minimal sufficient dimension.',
        'Seven removed fields use retained dx/dy/dist; no reconstruction of geometry from positions is claimed.',
        'No FP16 quantization, covariance codec, synthetic Gaussian support or deadline proof.',
        'Aggregate case outcomes and field names only; no raw toy rows or real records exported.',
    ],
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({key: result[key] for key in (
    'case_count', 'toy_rows', 'finite_and_eligible_toy_pass',
    'nominal_g51_fidelity_pass', 'nominal_g53_fidelity_pass',
    'actual_primitive_dimension', 'nominal_designs_collide',
)}))
