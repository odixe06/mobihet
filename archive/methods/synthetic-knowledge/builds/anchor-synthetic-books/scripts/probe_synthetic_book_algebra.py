"""Bounded aggregate-only algebra audit; no model/training or record export."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def neg(a):
    return -a[1], -a[0]


def mul(a, b):
    z = np.stack([a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]])
    return z.min(0), z.max(0)


def hyp(a, b):
    def abs_bounds(x):
        lo = np.where((x[0] <= 0) & (x[1] >= 0), 0, np.minimum(abs(x[0]), abs(x[1])))
        return lo, np.maximum(abs(x[0]), abs(x[1]))
    aa, bb = abs_bounds(a), abs_bounds(b)
    return np.hypot(aa[0], bb[0]), np.hypot(aa[1], bb[1])


def div_positive(a, b):
    # Caller excludes denominator intervals containing zero; no synthetic repair.
    safe = (np.maximum(b[0], 1e-300), np.maximum(b[1], 1e-300))
    return mul(a, (1/safe[1], 1/safe[0]))


def quantiles(v):
    return dict(zip(('p50', 'p90', 'p99', 'max'), map(float, np.quantile(v, [.5, .9, .99, 1])))) if len(v) else None


def main():
    fx = ROOT/'papers/pilot/fixture'
    meta = json.loads((ROOT/'knowledge/meta.json').read_text())
    scaler = json.loads((ROOT/'knowledge/scaler.json').read_text())['features']
    cols = meta['feature_cols']
    assert len(cols) == 66
    batches = []
    for cid in range(20):
        x = np.load(fx/f'train_{cid}_X.npy', mmap_mode='r')
        assert x.dtype == np.float16, x.dtype
        ix = np.random.default_rng(5501+cid).choice(len(x), min(2048, len(x)), replace=False)
        batches.append(np.asarray(x[ix]).copy())
    z16 = np.concatenate(batches)
    z = z16.astype(np.float64)
    assert np.isfinite(z).all()
    prev = np.nextafter(z16, np.float16(-np.inf)).astype(np.float64)
    nxt = np.nextafter(z16, np.float16(np.inf)).astype(np.float64)
    # Midpoints of neighboring fp16 representables cover round-to-nearest source value.
    zl, zh = (prev+z)/2, (nxt+z)/2
    mu = np.array([scaler[c]['mean'] for c in cols])
    sd = np.array([scaler[c]['std_used'] for c in cols])
    raw, lo, hi = z*sd+mu, zl*sd+mu, zh*sd+mu
    # Float32 zscore/extractor rounding allowance, explicit rather than exact equality.
    margin = 8*np.finfo(np.float32).eps*np.maximum(1, np.maximum(abs(lo), abs(hi)))
    lo -= margin; hi += margin
    i = {c: k for k, c in enumerate(cols)}
    def val(c): return raw[:, i['f_'+c]]
    def bounds(c): return lo[:, i['f_'+c]], hi[:, i['f_'+c]]
    def zero(c):
        a = bounds(c)
        return (a[0] <= 0) & (a[1] >= 0)

    dx = add(bounds('snd_x_rel'), neg(bounds('rcv_x_rel')))
    dy = add(bounds('snd_y_rel'), neg(bounds('rcv_y_rel')))
    d = hyp(dx, dy)
    dx0, dy0 = val('snd_x_rel')-val('rcv_x_rel'), val('snd_y_rel')-val('rcv_y_rel')
    d0 = np.hypot(dx0, dy0)
    pred = {'dx': (dx0, dx), 'dy': (dy0, dy), 'dist': (d0, d)}
    valid_div = d[0] > 0
    headzero = {s: zero(s+'_hed_sin') & zero(s+'_hed_cos') for s in ('rcv', 'snd')}
    v, vb = {}, {}
    for side in ('rcv', 'snd'):
        for axis, trig in [('x', 'sin'), ('y', 'cos')]:
            name = side+'_v'+axis
            v[name] = val(side+'_spd')*val(side+'_hed_'+trig)
            vb[name] = mul(bounds(side+'_spd'), bounds(side+'_hed_'+trig))
            pred[name] = v[name], vb[name]
        unit = add(mul(bounds(side+'_hed_sin'), bounds(side+'_hed_sin')),
                   mul(bounds(side+'_hed_cos'), bounds(side+'_hed_cos')))
        pred[side+'_heading_unit'] = val(side+'_hed_sin')**2+val(side+'_hed_cos')**2, unit
    rvb = {a: add(vb['snd_v'+a], neg(vb['rcv_v'+a])) for a in ('x', 'y')}
    rv = {a: v['snd_v'+a]-v['rcv_v'+a] for a in ('x', 'y')}
    pred['rel_speed'] = np.hypot(rv['x'], rv['y']), hyp(rvb['x'], rvb['y'])
    dot = add(mul(rvb['x'], dx), mul(rvb['y'], dy))
    pred['closing_speed'] = -(rv['x']*dx0+rv['y']*dy0)/np.maximum(d0, 1e-300), neg(div_positive(dot, d))
    for name, component, b in [('bearing_sin', dx0, dx), ('bearing_cos', dy0, dy)]:
        pred[name] = component/np.maximum(d0, 1e-300), div_positive(b, d)
    hc = add(mul(bounds('snd_hed_cos'), bounds('rcv_hed_cos')),
             mul(bounds('snd_hed_sin'), bounds('rcv_hed_sin')))
    pred['hed_diff_cos'] = val('snd_hed_cos')*val('rcv_hed_cos')+val('snd_hed_sin')*val('rcv_hed_sin'), hc
    pred['spd_diff'] = val('snd_spd')-val('rcv_spd'), add(bounds('snd_spd'), neg(bounds('rcv_spd')))
    for side in ('rcv', 'snd'):
        pred[side+'_noise_mag'] = np.hypot(val(side+'_pos_noise_x'), val(side+'_pos_noise_y')), hyp(bounds(side+'_pos_noise_x'), bounds(side+'_pos_noise_y'))

    first = val('first_in_session') >= .5
    any_headzero = headzero['rcv'] | headzero['snd']
    geom_zero = zero('dist')
    patterns = {f'first{int(f)}_heading_zero{int(h)}_dist_zero{int(g)}':
                (first == f) & (any_headzero == h) & (geom_zero == g)
                for f in (False, True) for h in (False, True) for g in (False, True)}
    results = {}
    for name, (center, interval) in pred.items():
        if name.endswith('_heading_unit'):
            actual = np.ones(len(z)); al, ah = actual, actual
            use = ~headzero[name.split('_')[0]]
        else:
            actual = val(name); al, ah = bounds(name); use = np.ones(len(z), dtype=bool)
        if name in ('bearing_sin', 'bearing_cos', 'closing_speed'):
            use &= valid_div
        if name in ('hed_diff_cos', 'rel_speed', 'closing_speed'):
            use &= ~any_headzero
        gap = np.maximum(0, np.maximum(interval[0]-ah, al-interval[1]))
        violation = gap > 0
        tol = np.maximum(abs(center-interval[0]), abs(interval[1]-center)) + np.maximum(abs(actual-al), abs(ah-actual))
        groups = {'all': use, **{k: use & v for k, v in patterns.items()}}
        results[name] = {k: {'rows': int(mask.sum()), 'violations': int((mask & violation).sum()),
                            'violation_fraction': float(violation[mask].mean()) if mask.any() else None,
                            'abs_residual_raw': quantiles(abs(center[mask]-actual[mask])),
                            'propagated_tolerance_raw': quantiles(tol[mask]),
                            'interval_gap_raw': quantiles(gap[mask])}
                         for k, mask in groups.items() if mask.any()}
        results[name]['excluded_rows'] = int((~use).sum())
    sources = ['../dataset/script/preprocessed/vnx_features.py', '../dataset/centralized/feature_schema.json',
               'knowledge/meta.json', 'knowledge/scaler.json', 'papers/pilot/fixture/manifest.json']
    out = {'evidence': 'bounded_local_algebra_only_no_training', 'rows': len(z), 'clients': 20,
           'max_rows_per_client': 2048, 'sampling_seed_recipe': '5501+cid; uniform_without_replacement',
           'fp16_interval': 'neighbor_midpoints; plus8*float32eps*max(1,raw_interval_abs)',
           'sources_sha256': {s: hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           'patterns': {k: int(v.sum()) for k, v in patterns.items()},
           'branch_counts': {'heading_pair_zero_rcv': int(headzero['rcv'].sum()),
                             'heading_pair_zero_snd': int(headzero['snd'].sum()),
                             'raw_dist_atom_zero_consistent': int(geom_zero.sum()),
                             'primitive_distance_interval_contains_zero': int((~valid_div).sum())},
           'geometry_identities': results,
           'limits': ['Aggregates only; no record/index/label/latent/gradient export.',
                      'Inverse zscore does not recover original nonfinite flags or raw float64 precision.',
                      'Raw zero speed/noise can be legitimate or imputed: branch not identifiable.',
                      'Excluded denominator/heading branches retained in counts, never called PASS.',
                      'Sample fidelity cannot establish all-data branch validity or F1/DP/R1.']}
    dest = ROOT/'papers/v5/algebra-probe.json'; dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'rows': len(z), 'branch_counts': out['branch_counts'],
                      'violations': {k: v['all']['violations'] for k, v in results.items()}}))


if __name__ == '__main__':
    main()
