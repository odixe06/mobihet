"""Bounded local block-swap/extractor closure; no model training or row export."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT.parent / 'dataset/script/preprocessed/vnx_features.py'
spec = importlib.util.spec_from_file_location('response_extractor', SOURCE)
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)
BLOCK = ['f_rcv_pos_noise_x', 'f_rcv_pos_noise_y', 'f_rcv_spd_noise',
         'f_rcv_acl_noise', 'f_rcv_hed_noise', 'f_rcv_noise_mag',
         'f_rcv_profile_normal', 'f_rcv_profile_cautious', 'f_rcv_profile_aggressive']


def raw_row():
    row = {column: 1. for _, column in extractor.RAW_SOURCES}
    row.update(rcv_pos_x=0., rcv_pos_y=0., snd_pos_x=3., snd_pos_y=4.,
               rcv_spd=2., snd_spd=3., rcv_hed=0., snd_hed=0.,
               rcv_time_ns=1_000_000_000, send_time_ns=900_000_000,
               message_id=1., receiver_id='toy', sender_alias='alias',
               rcv_profile='NORMAL', snd_profile='NORMAL')
    return row


def derive(row):
    table = pa.table({key: [value] for key, value in row.items()})
    return extractor.impute_zero(extractor.feature_matrix(table, 0., 0.))


def main():
    original = raw_row()
    original_features = derive(original)
    raw_block = [name[2:] for name in BLOCK[:5]] + ['rcv_profile']
    cases = []
    for name, nx, ny, profile in [('finite', 3., 4., 'CAUTIOUS'),
                                 ('nan_x', np.nan, 4., 'AGGRESSIVE'),
                                 ('nan_y', 3., np.nan, 'NORMAL'),
                                 ('infinite', np.inf, 4., 'UNKNOWN')]:
        donor = raw_row()
        donor.update(rcv_pos_noise_x=nx, rcv_pos_noise_y=ny, rcv_profile=profile,
                     rcv_spd_noise=2., rcv_acl_noise=3., rcv_hed_noise=4.)
        perturbed = dict(original)
        perturbed.update({key: donor[key] for key in raw_block})
        actual, donor_features = derive(perturbed), derive(donor)
        failed = [feature for feature in extractor.FEATURE_COLUMNS
                  if not np.array_equal(actual[feature],
                                        donor_features[feature] if feature in BLOCK else original_features[feature])]
        cases.append(dict(name=name, closure_pass=not failed, failed_features=failed))
    assert all(case['closure_pass'] for case in cases)
    fx = ROOT/'papers/pilot/fixture'
    manifest = json.loads((fx/'manifest.json').read_text())
    columns = [manifest['features'].index(name) for name in BLOCK]
    outside = [i for i in range(66) if i not in columns]
    rng = np.random.default_rng(1729)
    rows = valid = donor_groups = 0
    by_class = np.zeros((16, 2), dtype=np.int64)
    for k in range(20):
        xx = np.load(fx/f'train_{k}_X.npy', mmap_mode='r')
        yy = np.load(fx/f'train_{k}_y.npy', mmap_mode='r')
        gg = np.load(fx/f'train_{k}_g.npy', mmap_mode='r')
        selected = rng.choice(len(yy), min(2048, len(yy)), replace=False)
        x, y, g = xx[selected].copy(), yy[selected], gg[selected]
        changed = x.copy()
        donor = np.full(len(y), -1)
        for c in range(16):
            idx = np.flatnonzero(y == c)
            groups = np.unique(g[idx])
            donor_groups += len(groups)
            by_class[c, 0] += len(idx)
            if len(groups) < 2:
                continue
            for j, group in enumerate(groups):
                recipients = idx[g[idx] == group]
                pool = idx[g[idx] == groups[(j+1) % len(groups)]]
                donor[recipients] = rng.choice(pool, len(recipients), replace=True)
        ok = donor >= 0
        changed[np.ix_(np.flatnonzero(ok), columns)] = x[np.ix_(donor[ok], columns)]
        assert np.array_equal(x[:, outside], changed[:, outside])
        assert np.isfinite(changed).all()
        assert (y[ok] == y[donor[ok]]).all() and (g[ok] != g[donor[ok]]).all()
        assert np.array_equal(changed[np.ix_(np.flatnonzero(ok), columns)], x[np.ix_(donor[ok], columns)])
        rows += len(y)
        valid += int(ok.sum())
        by_class[:, 1] += np.bincount(y[ok], minlength=16)
    report = dict(stage='EXTRACTOR_AND_BOUNDED_SWAP_ONLY', block=BLOCK, seed=1729,
                  extractor_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  script_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  fixture_manifest_sha=hashlib.sha256((fx/'manifest.json').read_bytes()).hexdigest(),
                  toy_cases=cases, rows=rows, valid_swaps=valid,
                  class_rows_valid=by_class.tolist(), toy_closure_pass=True,
                  no_changes_outside_block=True, same_label_client_different_hashed_receiver=True,
                  rows_exported=0, official_test_read=False, model_training=False,
                  physical_counterfactual_or_causal_proof=False,
                  note='Copying annotations preserves extractor identities, not sensor-generation/label causality. '
                       'Hashed receiver scope is scenario|receiver as in existing fixture; no raw IDs exported.')
    path = ROOT/'papers/receiver-response/fidelity-probe.json'
    path.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
