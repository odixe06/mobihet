"""LA cohort check (24/11/2026, docs/research/screening-la.md): ablation A3 vs A4 (~L4) per-vehicle CM R50 from the
experiment repo pulls -> per-class recall/FP at vehicles that have / lack the class in train, absent-row recall.

    python scripts/analyse_la_cohort.py "name:<owner>/<a3 slug dir>:<owner>/<a4 slug dir>" ...
"""
import numpy as np, glob, sys
P='/home/odixe/project/experiment/runs/ablation-l4/pulls/'
cnt=np.load('/home/odixe/project/mobihet/cache/fold_counts.npy').sum(1)  # (5343,16) train+val
pk=(cnt+1)/(cnt.sum(1,keepdims=True)+16); tot=cnt.sum(0); pg=(tot+1)/(tot.sum()+16)
shift=np.log(pk)-np.log(pg)
def cm(pat):
    fs=sorted(glob.glob(P+pat+'/*/cms_r050.npz'))
    if not fs: return None
    z=np.load(fs[0]); return z['ids'], z.get('private') if 'private' in z.files else z[z.files[1]]
def f1(C):
    tp=np.diag(C); p=tp/np.maximum(C.sum(0),1); r=tp/np.maximum(C.sum(1),1); return 2*p*r/np.maximum(p+r,1e-12),p,r
for name,a3,a4 in [(s.split(':')[0],s.split(':')[1],s.split(':')[2]) for s in sys.argv[1:]]:
    A=cm(a3);B=cm(a4)
    if A is None or B is None: print(name,'missing'); continue
    assert (A[0]==B[0]).all(); ids=A[0]; CA,CB=A[1],B[1]
    pres=cnt[ids]>0  # (K,16)
    fa,pa,ra=f1(CA.sum(0)); fb,pb,rb=f1(CB.sum(0))
    print(f'== {name}: macro A3 {fa.mean():.5f} A4 {fb.mean():.5f} d {fa.mean()-fb.mean():+.5f}')
    for c in [11,13,1,6,15,0]:
        # recall on true-c rows at vehicles where c present / absent in train
        out=[]
        for lab,m in [('pres',pres[:,c]),('abs',~pres[:,c])]:
            ta,tb=CA[m,c,:].sum(),CB[m,c,:].sum()
            reca=CA[m,c,c].sum()/max(ta,1); recb=CB[m,c,c].sum()/max(tb,1)
            fpa=CA[m][:,:,c].sum()-CA[m,c,c].sum(); fpb=CB[m][:,:,c].sum()-CB[m,c,c].sum()
            out.append(f'{lab}: n{ta} R {reca:.3f}/{recb:.3f} FP {fpa}/{fpb}')
        mshift=shift[ids][pres[:,c],c]
        print(f' c{c:2d} F1 {fa[c]:.4f}/{fb[c]:.4f} P {pa[c]:.3f}/{pb[c]:.3f} R {ra[c]:.3f}/{rb[c]:.3f} | '+' | '.join(out)+f' | shift(pres) med {np.median(mshift):+.2f}')
    # absent-class rows total
    ra_=sum(CA[k,c,c] for k,c in zip(*np.where(~pres))); rb_=sum(CB[k,c,c] for k,c in zip(*np.where(~pres)))
    n_=sum(CA[k,c].sum() for k,c in zip(*np.where(~pres)))
    print(f' absent rows {n_}: recall A3 {ra_/max(n_,1):.3f} A4 {rb_/max(n_,1):.3f}')
