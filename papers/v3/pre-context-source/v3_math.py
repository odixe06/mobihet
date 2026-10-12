"""Public coordinate schedules and orthogonal update projections; no learned model."""
import numpy as np
import torch

FAMILIES={
    'r':[f'a_r_{s}_{m}' for s in ('small','medium') for m in ('noef','ef','own_ef')],
    't':[f'a_t_{s}_{m}' for s in ('small','medium') for m in ('comm','train','own_train')],
    'o':[f'a_o_{s}_{m}' for s in ('small','medium') for m in ('noef','ef','own_ef')],
    'p':['a_p_medium_noef','a_p_medium_ef','a_p_medium_own_ef','a_o_rank8_ef','a_o_rank8_own_ef']}
ARMS=[arm for arms in FAMILIES.values() for arm in arms]
D=395024; ROWS=512; COLS=772


def generator(seed,round_no):
    return torch.Generator().manual_seed(int(np.random.SeedSequence([seed,round_no]).generate_state(1)[0]))


def coordinates(layout,k,round_no,rolling=False):
    if not rolling:
        return torch.randperm(layout.size,generator=generator(2401,round_no))[:k]
    sizes=[n for _,_,n in layout.items]; remaining=k-len(sizes); assert remaining>=0
    allocation=np.asarray(sizes)/sum(sizes)*remaining
    counts=np.floor(allocation).astype(int)+1
    for j in np.argsort(-(allocation-np.floor(allocation)),kind='stable')[:k-int(counts.sum())]: counts[j]+=1
    assert sum(counts)==k and all(q<=n for q,n in zip(counts,sizes))
    indices=[]; offset=0
    for j,(n,q) in enumerate(zip(sizes,counts)):
        permutation=torch.randperm(n,generator=generator(2402,j))
        positions=(torch.arange(int(q))+(round_no-1)*int(q))%n
        indices.append(permutation[positions]+offset); offset+=n
    return torch.cat(indices)


def orthogonal(matrix):
    # CPU QR: same public recipe on both GPUs; no global RNG consumption.
    q,r=torch.linalg.qr(matrix.detach().cpu().float(),mode='reduced')
    sign=torch.where(r.diag()<0,-1.,1.)
    q=q*sign
    assert torch.linalg.matrix_norm(q.T@q-torch.eye(q.shape[1]))<=1e-5
    return q.to(matrix.device)


def basis(rank,round_no,device):
    return orthogonal(torch.randn(COLS,rank,generator=generator(2403,round_no))).to(device)


def matrix(vector):
    assert vector.numel()==D
    return torch.nn.functional.pad(vector,(0,ROWS*COLS-D)).reshape(ROWS,COLS)


def unpad(value):
    return value.reshape(-1)[:D]
