"""Frozen-head gradient diagnosis: teacher targets versus support in private feature coordinates.

No method launch. Real pooled support and class-mean targets are R3-ineligible diagnostic oracles.
Old/group labels measure directions, never choose inputs, teacher, step length or checkpoints.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from fd_probe23 import C, P, basis, decode_stats, encode_stats, fd, fit, forward, report, sha, virtual
from fd_probe24 import invert


def augmented(h):
    return torch.cat([h.double(), torch.ones(len(h), 1, device=h.device, dtype=torch.float64)], 1)


def gradient(a, h, q, shift=None, scale=None):
    z = h @ a.T
    if shift is not None: z = z + shift
    # Preserve the exact gradient of KL for decoded/fp32 probabilities whose row mass can differ from 1 by rounding.
    r = z.softmax(1) * q.sum(1, keepdim=True) - q
    weight = torch.ones(len(h), device=h.device, dtype=torch.float64) if scale is None else scale.double()
    return (r * weight[:, None]).T @ h / weight.sum()


def cosine(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def class_mean(x, labels):
    count = torch.bincount(labels, minlength=C).double()
    return torch.zeros(C, x.shape[1], device=x.device, dtype=torch.float64).index_add_(0, labels, x.double()) / count[:, None].clamp_min(1)


def moments(h, labels):
    mean = class_mean(h, labels)
    var = class_mean(h.double().square(), labels) - mean.square()
    return mean, var.clamp_min(0)


def moment_error(h, labels, reference):
    mean, var = moments(h, labels); rm, rv = reference
    return dict(mean_scaled_mse=float(((mean - rm).square() / rv.mean(0).clamp_min(.01)).mean()),
                variance_relative_mse=float(((var - rv).square() / rv.square().mean(0).clamp_min(.0001)).mean()))


def components(a, h, q, labels):
    residual = (h @ a.T).softmax(1) * q.sum(1, keepdim=True) - q
    hm, rm = class_mean(h, labels), class_mean(residual, labels)
    rho = torch.bincount(labels, minlength=C).double() / len(labels)
    mean_part = torch.einsum('c,ci,cj->ij', rho, rm, hm)
    centered = (residual - rm[labels]).T @ (h - hm[labels]) / len(h)
    total = residual.T @ h / len(h)
    assert torch.allclose(total, mean_part + centered, atol=1e-10, rtol=1e-9)
    return dict(mean_norm=float(mean_part.norm()), centered_norm=float(centered.norm()),
                total_norm=float(total.norm()), mean_centered_cos=cosine(mean_part, centered))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True); parser.add_argument('--fixture', required=True)
    parser.add_argument('--out', required=True)
    a = parser.parse_args(); start = time.monotonic()
    fd.deterministic(); torch.set_num_threads(2); dev = torch.device('cuda')
    log = lambda *v: print(round(time.monotonic() - start, 1), *v, flush=True)
    fx = Path(a.fixture); manifest = json.loads((fx / 'manifest.json').read_text())
    for name, expected in manifest['files'].items(): assert sha(fx / name) == expected
    assert manifest['train_frac'] == .025 and manifest['group_overlap'] == 0
    zero = [manifest['features'].index(n) for n in fd.BLOCK + fd.POSITION]
    def read(name):
        x = torch.from_numpy(np.load(fx / f'{name}_X.npy').astype('float32')).to(dev); x[:, zero] = 0
        return x, torch.from_numpy(np.load(fx / f'{name}_y.npy').astype('int64')).to(dev)
    weights = torch.load(a.weights, map_location=dev, weights_only=True)
    teacher = P.make('dagsnet').to(dev); teacher.load_state_dict(weights['proxy'][0], strict=True)
    teacher.eval().requires_grad_(False); projection = basis(teacher.head[5].weight, 48)
    own = {}; total = 0; ni = torch.zeros(C, device=dev, dtype=torch.float64)
    si = torch.zeros(C, 66, device=dev, dtype=torch.float64)
    qi = torch.zeros(C, 66, 66, device=dev, dtype=torch.float64)
    generator = torch.Generator().manual_seed(22)
    full_x, full_y = [], []
    for k in range(20):
        x, y = read(f'train_{k}'); full_x.append(x.cpu().numpy()); full_y.append(y.cpu().numpy())
        index = torch.randperm(len(y), generator=generator)[:16384].to(dev); x, y = x[index], y[index]
        z, h, _ = forward(teacher, x, feats=True)
        total = total + encode_stats(h, y, 'project48', projection)[0]
        own[k] = (x, y, h.float(), z)
        ni += torch.bincount(y, minlength=C)
        si.index_add_(0, y, x.double())
        for c in range(C):
            xx = x[y == c].double(); qi[c] += xx.T @ xx
    n, mu, cov = decode_stats(total, 'project48', projection)
    hbank, ybank = virtual(mu, cov, n / n.sum(), 200000, 23)
    head = torch.nn.Linear(48, C).to(dev)
    with torch.no_grad():
        head.weight.copy_(teacher.head[5].weight @ projection.float()); head.bias.copy_(teacher.head[5].bias)
    head = fit(head, hbank, ybank, 3000, 23); del hbank, ybank
    calibrated = copy.deepcopy(teacher)
    with torch.no_grad():
        calibrated.head[5].weight.copy_(head.weight @ projection.float().T); calibrated.head[5].bias.copy_(head.bias)
    im = si / ni[:, None]; ic = (qi - ni[:, None, None] * im[:, :, None] * im[:, None, :]) / (ni[:, None, None] - 1)
    share = torch.ones(C, device=dev) / C
    initial, labels = virtual(im, ic, share, 4096, 24)
    targets, target_labels = virtual(mu, cov, share, 4096, 24); assert torch.equal(labels, target_labels)
    initial[:, zero] = 0
    revised = invert(teacher, initial, targets, projection.float(),
                     ic.diagonal(dim1=1, dim2=2).mean(0).sqrt().float().clamp_min(.1),
                     cov.diagonal(dim1=1, dim2=2).mean(0).sqrt().float().clamp_min(.1), zero, 80)
    pooled_x, pooled_y = np.concatenate(full_x), np.concatenate(full_y); del full_x, full_y
    rows = np.empty((len(labels), 66), dtype='float32'); random = np.random.default_rng(27)
    for c in range(C):
        slots = np.where(labels.cpu().numpy() == c)[0]
        rows[slots] = pooled_x[random.choice(np.where(pooled_y == c)[0], len(slots), replace=True)]
    real = torch.from_numpy(rows).to(dev); del pooled_x, pooled_y
    supports = dict(gaussian=initial, inversion=revised, real_oracle=real)
    teachers = dict(original=teacher, calibrated=calibrated)
    qs = {(t, s): forward(model, x).softmax(1).double() for t, model in teachers.items() for s, x in supports.items()}
    references = {t: class_mean(qs[t, 'real_oracle'], labels) for t in teachers}
    evals = {s: read(s) for s in ('old', 'group')}
    out = dict(scope='frozen original private heads; full-bank surrogate gradients; real/support-target oracles R3-ineligible; no official test',
               protocol=dict(clients=20,bank=4096,inversion_steps=80,step_logit_rms=.01,reference_risks=['record_CE','balanced_CE','benign_CE']),
               weights_sha=sha(a.weights),fixture_sha=sha(fx/'manifest.json'),source_sha=sha(__file__),
               dependencies={name:sha(Path(__file__).with_name(name)) for name in ('fd_probe23.py','fd_probe24.py','fd_probe14.py','fd.py')},
               target_mass={f'{t}_{s}':q.mean(0).tolist() for (t,s),q in qs.items()}, teacher_moments={}, clients={})
    for s, x in supports.items():
        h = forward(teacher, x, feats=True)[1].double() @ projection
        out['teacher_moments'][s] = moment_error(h, labels, (mu, cov.diagonal(dim1=1,dim2=2)))
    path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    predictions = {s+'_y':yy.cpu().numpy().astype('uint8') for s,(_,yy) in evals.items()}
    for k in range(20):
        private = P.make('dagsnet').to(dev); private.load_state_dict(weights['private'][k], strict=True)
        private.eval().requires_grad_(False)
        matrix = torch.cat([private.head[5].weight, private.head[5].bias[:,None]], 1).double()
        x, y, ht, zt = own[k]; hp = augmented(forward(private, x, feats=True)[1])
        hs = {s: augmented(forward(private, xx, feats=True)[1]) for s, xx in supports.items()}
        counts = torch.tensor(manifest['counts'][f'train_{k}'], device=dev)
        global_counts = torch.tensor([sum(manifest['counts'][f'train_{j}'][c] for j in range(20)) for c in range(C)],device=dev)
        shift = ((counts+1)/(counts.sum()+C)).log() - ((global_counts+1)/(global_counts.sum()+C)).log()
        g_ce = gradient(matrix,hp,F.one_hot(y,C),shift=shift)
        ce = F.cross_entropy(hp @ matrix.T + shift,y); proxy_ce = F.cross_entropy(zt,y)
        scale = 1/(ce+proxy_ce)
        real_targets = {t: model.head[5](ht).softmax(1).double() for t, model in teachers.items()}
        g_real = {t: gradient(matrix,hp,q) for t,q in real_targets.items()}
        # Independent check of the analytical own-CE gradient, including the fixed logit adjustment.
        check = matrix.detach().clone().requires_grad_(True)
        reference = torch.autograd.grad(F.cross_entropy(hp @ check.T + shift,y),check)[0]
        error = float((reference-g_ce).abs().max()); assert error < 1e-9
        check = matrix.detach().clone().requires_grad_(True)
        qcheck = qs['calibrated','inversion']
        kl = F.kl_div((hs['inversion'] @ check.T).log_softmax(1),qcheck,reduction='batchmean')
        kl_error = float((torch.autograd.grad(kl,check)[0]-gradient(matrix,hs['inversion'],qcheck)).abs().max())
        assert kl_error < 1e-9
        test = {}; refs = {}
        for split,(xx,yy) in evals.items():
            h = augmented(forward(private,xx,feats=True)[1]); one = F.one_hot(yy,C)
            frequency = torch.bincount(yy,minlength=C).double()
            gradients = dict(record_CE=gradient(matrix,h,one),balanced_CE=gradient(matrix,h,one,scale=1/frequency[yy]),
                             benign_CE=gradient(matrix,h,one,scale=(yy==0).double()))
            refs[split]=gradients; test[split]=(h,yy)
        entry = dict(autograd_max_abs_error=error,kl_autograd_max_abs_error=kl_error,weight=float(scale),private_moments={},directions={})
        refmom = moments(hs['real_oracle'][:,:-1],labels)
        for s,h in hs.items(): entry['private_moments'][s]=moment_error(h[:,:-1],labels,refmom)
        directions = dict(own_ce=g_ce)
        for t in teachers:
            directions[t+'_own_kl']=g_real[t]
            for s,h in hs.items():
                for mode,q in (('input_target',qs[t,s]),('same_real_classmean',references[t][labels])):
                    key=f'{t}_{s}_{mode}'
                    gs=gradient(matrix,h,q)
                    directions[key+'_syn']=gs
                    directions[key+'_total']=g_ce+scale*(g_real[t]+gs)
                    entry.setdefault('decomposition',{})[key]=components(matrix,h,q,labels)
        for name,direction in directions.items():
            assert torch.isfinite(direction).all()
            dz=hp @ direction.T; dz=dz-dz.mean(1,keepdim=True)
            eta=.01/dz.square().mean().sqrt().clamp_min(1e-12)
            updated=matrix-eta*direction
            value=dict(norm=float(direction.norm()),eta=float(eta),splits={})
            for split,(h,yy) in test.items():
                logits=h @ updated.T; before=h @ matrix.T
                predictions[f'{k}_{name}_{split}']=logits.argmax(1).cpu().numpy().astype('uint8')
                value['splits'][split]=dict(cosines={risk:cosine(direction,gg) for risk,gg in refs[split].items()},
                    ce_change=float(F.cross_entropy(logits,yy)-F.cross_entropy(before,yy)),
                    benign_ce_change=float(F.cross_entropy(logits[yy==0],yy[yy==0])-F.cross_entropy(before[yy==0],yy[yy==0])),
                    report=report(yy,logits.argmax(1)))
            entry['directions'][name]=value
        # Paired full-gradient comparison isolates source geometry when the target is held identical by class.
        entry['same_target_gradient_cos_to_real']={}
        for t in teachers:
            gr=directions[f'{t}_real_oracle_same_real_classmean_syn']
            entry['same_target_gradient_cos_to_real'][t]={s:cosine(directions[f'{t}_{s}_same_real_classmean_syn'],gr) for s in hs}
        entry['original']={s:report(yy,(h @ matrix.T).argmax(1)) for s,(h,yy) in test.items()}
        for s,(h,yy) in test.items(): predictions[f'{k}_original_{s}']=(h @ matrix.T).argmax(1).cpu().numpy().astype('uint8')
        out['clients'][k]=entry
        out['seconds']=time.monotonic()-start
        path.write_text(json.dumps(out,indent=1,allow_nan=False)+'\n')
        log('client',k,'gradient checks',error)
        del private,hp,hs,test,refs
    np.savez_compressed(path.with_suffix('.npz'),**predictions)
    out['predictions_sha']=sha(path.with_suffix('.npz'))
    out['seconds']=time.monotonic()-start
    path.write_text(json.dumps(out,indent=1,allow_nan=False)+'\n')
    log('saved',str(path))


if __name__ == '__main__': main()
