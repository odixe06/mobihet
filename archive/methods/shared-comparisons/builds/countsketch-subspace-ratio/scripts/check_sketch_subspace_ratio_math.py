"""Small real-model arithmetic checks before V2 training; no feasibility F1 claim."""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = next((p for p in Path(__file__).resolve().parents if (p/'CONTEXT.md').is_file()), Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))
from mh.dagsnet import build_model, CFG
from mh.v2_math import (pack_vector, pack_patch, unpack, SecureSum, CountSketch,
                        Layout, SeedBasis, ratio_weights)
from mh.channel import Transfer, contact


def rejects(fn):
    try:
        fn()
    except (AssertionError, ValueError):
        return
    raise AssertionError('invalid input accepted')


def main():
    torch.set_num_threads(1); dev = torch.device('cuda')
    x = torch.tensor([.123456, -2.3456, 1e-5])
    p = pack_vector('coeff', x, 3, 4, count=12, half=True)
    d = unpack(p, expected_base=3)
    assert len(p) == 32 + 8 + 6 and d['count'] == 12
    assert torch.equal(d['values'], x.half().float()) and not torch.equal(x, d['values'])
    rejects(lambda: unpack(p[:-1] + bytes([p[-1] ^ 1])))
    rejects(lambda: unpack(p, expected_base=2))
    rejects(lambda: pack_vector('dense', torch.tensor([1e6]), half=True))
    t = Transfer('coeff', len(p), p)
    done, cost = contact([t], 20); assert not done and cost == 20 and t.sent == 0
    done, cost = contact([t], 100); assert done == [t] and cost == len(p) + 24
    agg = SecureSum('coeff', 0, 3, 'cpu')
    for i in range(4):
        agg.add(i, pack_vector('coeff', x * (i+1), count=i+1))
    rejects(agg.finish)
    rejects(lambda: agg.add(0, pack_vector('coeff', x, count=1)))
    agg.add(4, pack_vector('coeff', x * 5, count=5))
    total, count = agg.finish(); assert count == 15 and torch.allclose(total / count, x)

    sketch = CountSketch(4096, 3, 512, 1701, dev)
    g = torch.Generator(device=dev).manual_seed(10)
    a = torch.randn(4096, device=dev, generator=g); b = torch.randn(4096, device=dev, generator=g)
    assert torch.allclose(sketch.encode(a+b), sketch.encode(a)+sketch.encode(b), atol=5e-6, rtol=2e-5)
    sparse = torch.zeros(4096, device=dev); sparse[[4,17,55]] = torch.tensor([.25,-.7,2.1], device=dev)
    residual = sketch.encode(sparse)
    packet, committed = sketch.patch(residual, 3, 0)
    assert torch.allclose(committed, sparse.half().float(), atol=0, rtol=0)
    left = residual - sketch.encode(committed)
    assert torch.allclose(left + sketch.encode(committed), residual, atol=1e-6, rtol=1e-6)
    assert len(packet) == 32 + 3*6

    torch.manual_seed(42)
    model = build_model(dict(CFG, n_features=66, num_classes=16), latent=True).to(dev).eval()
    layout = Layout(model); assert layout.size == 395024
    basis = SeedBasis(model, 8, 1901, dev)
    fixture = ROOT / 'papers/pilot/fixture'
    xx = torch.from_numpy(np.load(fixture / 'train_0_X.npy')[:32].astype('float32')).to(dev)
    yy = torch.from_numpy(np.load(fixture / 'train_0_y.npy')[:32].astype('int64')).to(dev)
    coeff = torch.zeros(8, device=dev, requires_grad=True)
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(basis.forward(model, coeff, xx)[0], yy)
    loss.backward()
    grad = coeff.grad.clone(); model.zero_grad(set_to_none=True)
    F.cross_entropy(model(xx)[0], yy).backward()
    full = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    assert torch.allclose(grad, basis.z.T @ full, atol=2e-5, rtol=1e-3)
    j = int(grad.abs().argmax()); eps = .03
    with torch.no_grad():
        plus = coeff.detach().clone(); minus = plus.clone(); plus[j] += eps; minus[j] -= eps
        fd = (F.cross_entropy(basis.forward(model, plus, xx)[0], yy) -
              F.cross_entropy(basis.forward(model, minus, xx)[0], yy))/(2*eps)
    assert abs(float(fd - grad[j])) < max(2e-4, abs(float(grad[j]))*.05), (fd, grad[j])
    c1 = torch.randn(8, device=dev, generator=g); c2 = torch.randn(8, device=dev, generator=g)
    assert torch.allclose(basis.vector((c1+c2)/2), (basis.vector(c1)+basis.vector(c2))/2, atol=1e-7, rtol=1e-5)
    clone = build_model(dict(CFG, n_features=66, num_classes=16), latent=True).to(dev).eval()
    layout.load(clone, basis.vector(c1)); clone.load_state_dict(clone.state_dict(), strict=True)
    assert torch.allclose(clone(xx)[0], basis.forward(clone, c1, xx)[0], atol=1e-6, rtol=1e-5)

    labels = torch.arange(16, device=dev).repeat_interleave(8)
    logits = torch.randn(128,16, device=dev, generator=g)
    weights, raw = ratio_weights(logits, labels)
    shuffled, _ = ratio_weights(logits, labels, shuffle=g)
    for y in range(16):
        assert torch.isclose(weights[labels==y].mean(), torch.tensor(1., device=dev))
        assert torch.allclose(weights[labels==y].sort().values, shuffled[labels==y].sort().values)
    assert torch.isfinite(weights).all() and float(raw.max()) <= 10
    print('PASS V2: actual codec/tamper/frame, cohort/duplicate, sketch linearity/decoded residual,')
    print('real DAGSNet chain rule/finite difference/reconstruction, class weight normalization/shuffle.')


if __name__ == '__main__':
    main()
