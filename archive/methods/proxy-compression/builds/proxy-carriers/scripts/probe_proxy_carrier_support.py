"""Small schema/support audit from internal training fixture, no model fitting or record export."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])


def main():
    meta=json.loads((ROOT/'knowledge/meta.json').read_text()); cols=meta['feature_cols']
    scaler=json.loads((ROOT/'knowledge/scaler.json').read_text())['features']
    schema_path=ROOT.parent/'dataset/centralized/feature_schema.json'
    schema=json.loads(schema_path.read_text()); assert schema['feature_columns']==cols
    atoms=np.array([[-scaler[c]['mean']/scaler[c]['std_used'],(1-scaler[c]['mean'])/scaler[c]['std_used']] for c in cols])
    groups=[[cols.index(f'f_{side}_profile_{name}') for name in ('normal','cautious','aggressive')] for side in ('rcv','snd')]
    flag=cols.index('f_first_in_session'); missing=[cols.index(c) for c in schema['zero_imputed_features']]
    rows=0; first_rows=0; violation=0; profile_invalid=[0,0]; unknown=[0,0]; binary_invalid=0
    zero_counts=np.zeros(66,dtype=np.int64); total_classes=np.zeros(16,dtype=np.int64)
    for k in range(20):
        fx=ROOT/'papers/pilot/fixture'; source=np.load(fx/f'train_{k}_X.npy',mmap_mode='r')
        labels=np.load(fx/f'train_{k}_y.npy',mmap_mode='r')
        take=np.random.default_rng(np.random.SeedSequence([4301,k])).choice(len(labels),min(2048,len(labels)),replace=False)
        x=np.asarray(source[take]); y=labels[take]; rounded=atoms.astype(x.dtype)
        atzero=x==rounded[:,0]; atone=x==rounded[:,1]; rows+=len(x)
        zero_counts+=atzero.sum(0); total_classes+=np.bincount(y,minlength=16)
        first=atone[:,flag]; first_rows+=int(first.sum())
        violation+=int((~atzero[first][:,missing]).any(axis=1).sum())
        binary_invalid+=int((~(atzero[:,flag]|atone[:,flag])).sum())
        for g,indices in enumerate(groups):
            count=atone[:,indices].sum(1); all_binary=(atzero[:,indices]|atone[:,indices]).all(1)
            profile_invalid[g]+=int((~all_binary|(count>1)).sum())
            unknown[g]+=int((all_binary&(count==0)).sum())
    record=dict(evidence='sample 2048/client from training fixture, not full scan; no model training',
        schema_sha256=hashlib.sha256(schema_path.read_bytes()).hexdigest(),rows=rows,first_rows=first_rows,
        first_session_structural_violations=violation,binary_invalid=binary_invalid,
        profile_invalid=profile_invalid,profile_allzero_unknown=unknown,class_counts=total_classes.tolist(),
        zero_fraction={c:float(n/rows) for c,n in zip(cols,zero_counts)},
        raw_zero_atoms={c:float(atoms[j,0]) for j,c in enumerate(cols)},
        structural_features=schema['zero_imputed_features'])
    out=ROOT/'papers/v3'; out.mkdir(exist_ok=True); (out/'support-probe.json').write_text(json.dumps(record,indent=2))
    print('SUPPORT AUDIT',rows,'rows; first',first_rows,'structural violations',violation,
          'binary invalid',binary_invalid,'profile invalid',profile_invalid,'unknown',unknown)
    assert violation==binary_invalid==0 and sum(profile_invalid)==0,'schema mismatch: stop sampler design'


if __name__=='__main__': main()
