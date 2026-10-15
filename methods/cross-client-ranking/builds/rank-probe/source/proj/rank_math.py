"""Cohort score moments and a locked common-support logit-gradient probe; no optimizer."""
import struct
import zlib
import numpy as np

C=16
HEADER=struct.Struct('<4sBBHII16s')


def packet(body,kind,identity):
    assert len(identity)==16
    return HEADER.pack(b'MHCR',1,kind,2,len(body),zlib.crc32(body),identity)+body


def body(wire,kind,identity):
    magic,schema,got,version,n,crc,key=HEADER.unpack_from(wire)
    value=wire[HEADER.size:]
    assert (magic,schema,got,version,key)==(b'MHCR',1,kind,2,identity)
    assert n==len(value) and crc==zlib.crc32(value)
    return value


def validate(counts,sums):
    assert counts.shape==(2,C) and sums.shape==(4,C) and np.isfinite(sums).all()
    for n,s,q in ((counts[0],sums[0],sums[2]),(counts[1],sums[1],sums[3])):
        tolerance=1e-10*np.maximum(n,1)
        assert np.all(s>=-tolerance) and np.all(q>=-tolerance)
        assert np.all(q<=s+tolerance) and np.all(s<=n+tolerance)
        assert np.all(q*n-s*s>=-tolerance*np.maximum(n,1))


def encode_up(counts,sums,identity):
    validate(counts,sums)
    flags=(counts>0).astype('<u4')
    wire=packet(counts.astype('<u8').tobytes()+sums.astype('<f8').tobytes()+flags.tobytes(),1,identity)
    assert len(wire)==928
    return wire


def decode_up(wire,identity):
    value=body(wire,1,identity);assert len(value)==896
    counts=np.frombuffer(value,dtype='<u8',count=2*C).reshape(2,C).copy()
    sums=np.frombuffer(value,dtype='<f8',count=4*C,offset=16*C).reshape(4,C).copy()
    flags=np.frombuffer(value,dtype='<u4',count=2*C,offset=48*C).reshape(2,C).copy()
    assert np.array_equal(flags,counts>0);validate(counts,sums)
    assert np.all(counts.sum(0)==counts.sum(0)[0])
    return counts,sums,flags


def aggregate(wires,identity):
    counts=np.zeros((2,C),dtype=np.uint64);sums=np.zeros((4,C),dtype=np.float64)
    contributors=np.zeros((2,C),dtype=np.uint32)
    for wire in wires:
        n,s,flags=decode_up(wire,identity)
        assert np.all(n<=np.iinfo(np.uint64).max-counts),'count overflow'
        counts+=n;sums+=s;contributors+=flags
    validate(counts,sums)
    means=np.divide(sums[:2],counts,out=np.zeros((2,C)),where=counts>0)
    eligible=(counts>=128).all(0)&(contributors>=5).all(0)
    total=int(counts[0].sum());assert np.all(counts.sum(0)==total)
    prior=((counts[0]+1)/(total+C)).astype('<f4')
    value=means.astype('<f8').tobytes()+counts.astype('<u8').tobytes()+contributors.astype('<u4').tobytes()
    value+=np.packbits(eligible,bitorder='little').tobytes()+prior.tobytes()+struct.pack('<Q',total)
    wire=packet(value,2,identity);assert len(wire)==746
    return wire,dict(counts=counts.tolist(),contributors=contributors.tolist(),eligible=eligible.tolist(),
        means=means.tolist(),second_moments=np.divide(sums[2:],counts,out=np.zeros((2,C)),where=counts>0).tolist())


def decode_down(wire,identity):
    value=body(wire,2,identity);assert len(value)==714
    means=np.frombuffer(value,dtype='<f8',count=2*C).reshape(2,C).copy()
    counts=np.frombuffer(value,dtype='<u8',count=2*C,offset=16*C).reshape(2,C).copy()
    contributors=np.frombuffer(value,dtype='<u4',count=2*C,offset=32*C).reshape(2,C).copy()
    eligible=np.unpackbits(np.frombuffer(value,dtype='u1',count=2,offset=40*C),bitorder='little').astype(bool)
    prior=np.frombuffer(value,dtype='<f4',count=C,offset=40*C+2).copy()
    total=struct.unpack_from('<Q',value,len(value)-8)[0]
    assert np.isfinite(means).all() and np.all((means>=0)&(means<=1))
    assert np.all(counts.sum(0)==total) and int(counts[0].sum())==total
    assert np.array_equal(eligible,(counts>=128).all(0)&(contributors>=5).all(0))
    assert np.allclose(prior,(counts[0]+1)/(total+C),rtol=1e-6,atol=1e-8)
    return means,counts,eligible,total


def gradient(prob,y,means,counts,total,eligible,weight=.25):
    """Per-example logit gradient of rank term (before mean over local batch)."""
    k=prob.shape[1];positive=np.arange(k)[None,:]==y[:,None]
    numerator=np.where(positive,prob-1-means[1],prob+1-means[0])
    denominator=np.where(positive,counts[0],counts[1])
    score=np.divide(2*weight*total/k*numerator,denominator,out=np.zeros_like(prob),where=denominator>0)
    score*=eligible[None,:]
    return prob*(score-(prob*score).sum(1,keepdims=True))


def basis(prob,y,logits):
    k=prob.shape[1];ce=prob-np.eye(k)[y]
    class_ce=ce[:,:,None]*(y[:,None,None]==np.arange(k)[None,None,:])
    bias=np.eye(k)[:,:k-1]-1/k
    temperature=(logits-logits.mean(1,keepdims=True))[:,:,None]
    return np.concatenate([class_ce,np.broadcast_to(bias,(len(y),k,k-1)),temperature],axis=2)


def projection(prob,y,logits,delta,fit):
    design=basis(prob,y,logits);columns=design.shape[-1]
    x=design[fit].reshape(-1,columns);target=delta[fit].ravel()
    scale=np.sqrt((x*x).mean(0));keep=scale>1e-12
    assert keep.any()
    coeff=np.linalg.lstsq(x[:,keep]/scale[keep],target,rcond=1e-10)[0]
    prediction=np.einsum('nkd,d->nk',design[~fit][:,:,keep]/scale[keep],coeff)
    return delta[~fit]-prediction,dict(columns=columns,retained_columns=int(keep.sum()),
        fit_rows=int(fit.sum()),test_rows=int((~fit).sum()))


def gates(clients,eligible):
    rows=[]
    for c in range(C):
        valid=[v['classes'][c] for v in clients if v['classes'][c] is not None]
        supported=bool(eligible[c] and len(valid)>=5)
        own=float(np.mean([x['own_energy'] for x in valid])) if valid else 0.
        delta=float(np.mean([x['delta_energy'] for x in valid])) if valid else 0.
        residual=float(np.mean([x['residual_energy'] for x in valid])) if valid else 0.
        nondegenerate=supported and own>1e-20 and delta>1e-20
        rows.append(dict(class_id=c,clients=len(valid),supported=supported,nondegenerate=nondegenerate,
            rms_ratio=float(np.sqrt(delta/own)) if nondegenerate else None,
            residual_fraction=residual/delta if nondegenerate else None))
    usable=[r for r in rows if r['nondegenerate']]
    signal=sum(r['rms_ratio']>=.05 for r in usable)
    residual=float(np.mean([r['residual_fraction'] for r in usable])) if usable else None
    coverage=sum(v['valid_scan_rows'] for v in clients)/sum(v['scan_rows'] for v in clients)
    passed=sum(r['supported'] for r in rows)>=12 and len(usable)>=12 and signal>=12 and coverage>=.9 and residual>=.2
    return dict(pass_probe=bool(passed),supported_classes=sum(r['supported'] for r in rows),
        nondegenerate_classes=len(usable),signal_classes=signal,residual_macro_class=residual,
        scan_coverage=coverage,classes=rows,utility_measured=False)
