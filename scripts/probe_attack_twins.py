"""Train fold only, one scenario. For attack rows: does the same vehicle hold a benign twin (sender, message_id, rcv_time)?
In the scaled 77-d input, which features differ attack - twin, per class? Reads no test file."""
import glob, json, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
R = '/home/odixe/project/mobihet-features/data/v4-scaled/train/'
cols = json.load(open('/home/odixe/project/mobihet/knowledge/input_order.json'))['inputs']
names = json.load(open('/home/odixe/project/mobihet/knowledge/label_mapping.json'))
names = names.get('id_to_label', names); names = {int(k): v for k, v in names.items()}
scen = sys.argv[1] if len(sys.argv) > 1 else 'highway_2'
fs = sorted(glob.glob(R + f'scenario={scen}/world=*/part-*.parquet'))
ids = ['client_id', 'label', 'fold', 'world', 'sender_id', 'message_id', 'rcv_time_ns']
rng = np.random.default_rng(1)
first = pq.ParquetFile(fs[0]).read(columns=['client_id']).to_pandas()['client_id'].unique()
keep = None
parts = []
for f in fs:
    t = pq.ParquetFile(f).read(columns=cols + ids).to_pandas()
    if keep is None:
        allc = t['client_id'].unique(); keep = set(rng.choice(allc, min(400, len(allc)), replace=False))
    t = t[t['client_id'].isin(keep) & (t['fold'].astype(str) == 'train')]
    t['world'] = t['world'].astype(str); parts.append(t)
d = pd.concat(parts, ignore_index=True)
print(scen, 'rows', len(d), 'clients', d.client_id.nunique(), flush=True)
key = ['client_id', 'sender_id', 'message_id', 'rcv_time_ns']
ben = d[d.label == 0].drop_duplicates(key)
X = cols
out = {}
print('benign world spread per client (median #worlds of benign rows):', d[d.label == 0].groupby('client_id').world.nunique().median())
for c in range(1, 16):
    a = d[d.label == c]
    if len(a) == 0: continue
    m = a.merge(ben, on=key, how='left', suffixes=('', '_b'), indicator=True)
    hit = m['_merge'] == 'both'
    mm = m[hit]
    same_world = float((mm['world'] == mm['world_b']).mean()) if len(mm) else float('nan')
    delta = np.abs(mm[X].to_numpy(np.float32) - mm[[x + '_b' for x in X]].to_numpy(np.float32)) if len(mm) else np.zeros((0, 77))
    changed = (delta > 0.01).mean(0) if len(mm) else np.zeros(77)
    sw = mm['world'] == mm['world_b']
    changed_sw = (delta[sw.values] > 0.01).mean(0) if sw.any() else np.zeros(77)
    top = np.argsort(-changed)[:10]
    out[names[c]] = dict(rows=int(len(a)), twin_rate=float(hit.mean()), twin_same_world=same_world,
                         n_feat_changed_mean=float((delta > 0.01).sum(1).mean()) if len(mm) else None,
                         n_feat_changed_same_world=float((delta[sw.values] > 0.01).sum(1).mean()) if sw.any() else None,
                         top_changed_same_world=[(X[i], round(float(changed[i]), 3)) for i in top],
                         changed_frac=changed.tolist())
    o = out[names[c]]
    hb=np.stack([np.histogram(d.loc[d.label==0,f],bins=64,range=(-4,4))[0] for f in X]).astype(float); hb/=hb.sum(1,keepdims=True)
    ha=np.stack([np.histogram(a[f],bins=64,range=(-4,4))[0] for f in X]).astype(float); ha/=ha.sum(1,keepdims=True)
    tv=.5*np.abs(ha-hb).sum(1); o['tv_vs_benign']=tv.tolist()
    ch=changed>0.5; topt=set(np.argsort(-tv)[:max(1,int(ch.sum()))])
    o['tv_top_overlap_with_changed']=len(topt & set(np.flatnonzero(ch)))/max(1,int(ch.sum()))
    print(f"{c:2d} {names[c][:22]:22s} rows {o['rows']:6d} twin {o['twin_rate']:.3f} sameworld {same_world:.3f} "
          f"nchg {o['n_feat_changed_mean']} n>50% {int((changed>.5).sum())} TVtop-overlap {o['tv_top_overlap_with_changed']:.2f}\n    {o['top_changed_same_world']}", flush=True)
json.dump(out, open(f'/home/odixe/project/mobihet/runs/transport-probe/twins-{scen}.json', 'w'), indent=1)
