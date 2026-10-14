"""Copied from archive3:papers/build20/.../proj/prepare.py (2026-10-10), plus train_frac.
Bounded real-data pilot; exact old temporal cut plus globally hashed receiver groups.
Build9: holdout=False trains on every row of train80% (final protocol); prepare_test builds the official test20% array."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SCEN = ('highway_2', 'highway_7', 'urban_2', 'urban_7')
CLIENTS = (0, 5, 10, 15)

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def codes(c):
    a = pc.index_in(c, value_set=pa.array(SCEN))
    assert not a.null_count
    return a.to_numpy(zero_copy_only=False).astype(np.uint8)

def groups(c, s):
    enc = pc.dictionary_encode(c.combine_chunks())
    # Scope receiver IDs by scenario; metadata only, never an input feature or wire field.
    h = np.array([[int.from_bytes(hashlib.blake2b((sc+'|'+r).encode(), digest_size=8).digest(), 'little')
                   for sc in SCEN] for r in enc.dictionary.to_pylist()], np.uint64)
    return h[enc.indices.to_numpy(), s]

def prepare(root, out, features, classes, train_cap=32768, eval_cap=16384, affine=None, index_dir=None, clients=CLIENTS, zero=None, holdout=True, train_frac=None):
    """affine=(mean, std, label): optional per-feature refit applied in shipped z-space; rows unchanged.
    index_dir: optionally save each set's global partition row indices (outside the fixture manifest).
    clients: partition clients to train; train_cap=None keeps every non-holdout row. zero: feature indices set to 0 (mean).
    holdout=False: no temporal/group holdout, every row trains (final protocol before test20%); no old/group sets.
    train_frac: MobiHet 2026-10-10 pilot -- keep this fraction of each client's candidate rows instead of train_cap, so client sizes keep their partition ratio."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    parts = sorted(Path(root).glob('train/client_id=*/*.parquet'))
    assert parts
    ts, ys, ss, entries, offset = [], [], [], [], 0
    identity = hashlib.sha256()
    for p in parts:
        t = pq.read_table(p, columns=['label','scenario','rcv_time_ns'])
        n = len(t); cid = int(p.parent.name.split('=')[1]); entries.append((p,cid,offset,offset+n)); offset += n
        ys.append(t['label'].to_numpy().astype(np.uint8)); ss.append(codes(t['scenario']))
        ts.append(t['rcv_time_ns'].to_numpy().astype(np.int64))
        identity.update(str(p.relative_to(root)).encode())
        with p.open('rb') as f:
            f.seek(-8,2); tail=f.read(8); assert tail[4:]==b'PAR1'
            size=int.from_bytes(tail[:4],'little');f.seek(-8-size,2);identity.update(f.read(size+8))
    y, s, times = np.concatenate(ys), np.concatenate(ss), np.concatenate(ts)
    del ys, ss, ts
    val = np.zeros(len(y), bool)
    for k in (np.unique(y.astype(np.uint16)*4+s) if holdout else ()):
        ix = np.flatnonzero(y.astype(np.uint16)*4+s == k)
        cut = np.partition(times[ix], len(ix)-int(np.ceil(.02*len(ix))))[len(ix)-int(np.ceil(.02*len(ix)))]
        val[ix[times[ix]>=cut]] = True
    del times
    rng = np.random.default_rng(1729)
    vi = np.flatnonzero(val); old = np.sort(rng.choice(vi, min(eval_cap,len(vi)), replace=False))
    selected = {}; train_ix = {}; novel = []; group_counts = {}
    for p,cid,lo,hi in entries:
        if cid not in clients: continue
        t = pq.read_table(p, columns=['receiver_id']); g = groups(t['receiver_id'], s[lo:hi]); held = (g%20==0) & holdout
        novel.append(lo+np.flatnonzero(held & ~val[lo:hi]))
        selected[lo] = (g, held)
    for cid in clients:
        cand = np.concatenate([lo+np.flatnonzero(~val[lo:hi] & ~selected[lo][1])
                               for _,c,lo,hi in entries if c==cid])
        if train_cap is None:
            train_ix[cid] = cand; continue
        # Reserve minimum class coverage, then fill uniformly; prior uses actual sample.
        extra = [rng.choice(cand[y[cand]==c], min(64, int((y[cand]==c).sum())), replace=False) for c in range(16)]
        extra = np.unique(np.concatenate(extra)); rest = np.setdiff1d(cand,extra)
        cap = int(np.ceil(train_frac*len(cand))) if train_frac else train_cap
        idx = rng.choice(rest,max(min(cap,len(cand))-len(extra),0),replace=False)
        train_ix[cid] = np.sort(np.concatenate([idx,extra]))
    ni = np.concatenate(novel); new = np.sort(rng.choice(ni,min(eval_cap,len(ni)),replace=False))
    sets = {**({'old':old,'group':new} if holdout else {}), **{f'train_{c}': ix for c,ix in train_ix.items()}}
    arrays = {k:np.empty((len(ix),len(features)),np.float16) for k,ix in sets.items()}
    gh = {k:np.empty(len(ix),np.uint64) for k,ix in sets.items()}
    for p,cid,lo,hi in entries:
        gaps = {k:(np.searchsorted(ix,lo),np.searchsorted(ix,hi)) for k,ix in sets.items()}
        gaps = {k:v for k,v in gaps.items() if v[1]>v[0]}
        if not gaps: continue
        t = pq.read_table(p,columns=features+['receiver_id'])
        g = groups(t['receiver_id'],s[lo:hi])
        for k,(a,b) in gaps.items():
            r = sets[k][a:b]-lo; sub = t.take(r)
            v = np.stack([sub[f].to_numpy() for f in features],1)
            if affine is not None: v = (v.astype(np.float64)-affine[0])/affine[1]
            if zero is not None: v = v.copy(); v[:, zero] = 0
            arrays[k][a:b] = v.astype(np.float16)
            gh[k][a:b] = g[r]
    seen = np.unique(np.concatenate([gh[f'train_{c}'] for c in clients]))
    if holdout: assert not np.intersect1d(seen,np.unique(gh['group'])).size
    assert not np.intersect1d(np.concatenate(list(train_ix.values())),vi).size
    hist = {}
    for k,ix in sets.items():
        assert np.isfinite(arrays[k]).all()
        np.save(out/f'{k}_X.npy',arrays[k]); np.save(out/f'{k}_y.npy',y[ix]); np.save(out/f'{k}_g.npy',gh[k])
        hist[k] = np.bincount(y[ix],minlength=16).tolist()
    if index_dir is not None:
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        for k,ix in sets.items(): np.save(Path(index_dir)/f'{k}_ix.npy',ix)
    files = {p.name:sha(p) for p in sorted(out.glob('*.npy'))}
    manifest = dict(source_id=identity.hexdigest(),source_rows=len(y),old_val_total=len(vi),features=features,
        classes=classes,clients=list(clients),train_cap=train_cap,train_frac=train_frac,eval_cap=eval_cap,counts=hist,
        group_scope='scenario,receiver_id; blake2b hash mod20==0 held out',group_overlap=0,
        protocol=('temporal2% whole partition; extra5% receiver-group TRAIN exclusion; sampled pilot' if holdout
                  else 'final: every train80% row trains, no internal holdout'),files=files)
    if zero is not None:
        manifest['zeroed_features'] = [features[j] for j in zero]
    if affine is not None:
        manifest['scaler_refit'] = dict(label=affine[2],mean=list(map(float,affine[0])),std=list(map(float,affine[1])))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared real pilot', {k:sum(v) for k,v in hist.items()}, 'old cut:',len(vi),flush=True)
    return manifest


def prepare_test(test_dir, out, features, scaler, zero=None, rows=None, sub=1 << 20, affine=None):
    """Official test20% (raw) -> shipped z-space with scaler.json, applied exactly once; zero: same schema zeroing as train.
    rows: optional uniform subsample size (local preflight only). Writes test_X.npy (fp16), test_y.npy; returns manifest part.
    affine=(mean, std, label): the same in-z-space refit the training fixture used (fixture B), applied after scaler.json."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    parts = sorted(Path(test_dir).glob('part-*.parquet')); assert parts
    mean = np.array([scaler['features'][f]['mean'] for f in features]); std = np.array([scaler['features'][f]['std_used'] for f in features])
    n = sum(pq.ParquetFile(p).metadata.num_rows for p in parts)
    keep = np.sort(np.random.default_rng(2718).choice(n, rows, replace=False)) if rows else None
    total = len(keep) if rows else n
    X = np.lib.format.open_memmap(out/'test_X.npy', mode='w+', dtype=np.float16, shape=(total, len(features)))
    y = np.empty(total, np.uint8); a = lo = 0; spd = 0.
    for p in parts:
        for t in pq.ParquetFile(p).iter_batches(batch_size=1 << 19, columns=features+['label']):
            m = t.num_rows
            r = np.arange(m) if keep is None else keep[(keep >= lo) & (keep < lo+m)] - lo
            v = np.stack([t.column(f).to_numpy() for f in features], 1)[r].astype(np.float64)
            assert np.isfinite(v).all(); spd += float(v[:, features.index('f_snd_spd')].sum())
            v = (v - mean) / std
            if affine is not None: v = (v - affine[0]) / affine[1]
            if zero is not None: v[:, zero] = 0
            X[a:a+len(r)] = v.astype(np.float16); y[a:a+len(r)] = t.column('label').to_numpy()[r]; a += len(r); lo += m
    assert a == total and np.isfinite(X).all() and y.max() < 16
    X.flush(); del X; np.save(out/'test_y.npy', y)
    # Fixed uniform subsample for descriptive learning curves; the full set is scored at the pre-registered cycles.
    pick = np.sort(np.random.default_rng(3141).choice(total, min(sub, total), replace=False))
    np.save(out/'testsub_X.npy', np.load(out/'test_X.npy', mmap_mode='r')[pick]); np.save(out/'testsub_y.npy', y[pick])
    raw_mean_spd = spd / total
    if rows is None: assert abs(raw_mean_spd - 7.4493) < 1e-3, ('test must be RAW before scaling', raw_mean_spd)
    return dict(test_rows=int(total), test_source_rows=int(n), test_counts=np.bincount(y, minlength=16).tolist(),
                test_raw_mean_f_snd_spd=raw_mean_spd, test_scaler='scaler.json (fit on all train80%), applied once'
                + ('' if affine is None else f'; then refit {affine[2]}'),
                testsub_rows=int(len(pick)), test_files={k: sha(out/k) for k in ('test_X.npy', 'test_y.npy', 'testsub_X.npy', 'testsub_y.npy')})
