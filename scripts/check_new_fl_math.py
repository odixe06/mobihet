"""Small independent arithmetic checks; all inputs here are artificial, never client records."""
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mh.new_methods import moments, kernel_sums, frequencies, phi, gaussian_embedding, fit_kernel_bank, fp16
from mh.channel import Transfer, contact, wire

torch.set_num_threads(1); torch.manual_seed(71)
x = torch.randn(1024, 66); y = torch.arange(1024) % 16; w = frequencies('cpu')
for getter in (lambda a, b: moments(a, b), lambda a, b: (kernel_sums(a, b, w),)):
    full = getter(x, y); left = getter(x[:311], y[:311]); right = getter(x[311:], y[311:])
    for a, b, c in zip(full, left, right):
        # phi's float32 GEMM can vary with batch shape; sum itself is float64.
        torch.testing.assert_close(a, b + c, rtol=1e-6, atol=2e-6)
mu = torch.zeros(16, 66); sd = torch.ones_like(mu)
target = gaussian_embedding(mu, sd, w)
empirical = phi(torch.randn(65536, 66), w).mean(0)
assert (empirical - target[0]).abs().max() < .003
bank, trace = fit_kernel_bank(mu, sd, target, w, torch.Generator().manual_seed(42), 20, size=32)
assert trace[-1]['fit'] < trace[0]['fit'] and torch.isfinite(bank).all()
q = fp16(bank); assert torch.equal(q, q.half().float())
t = Transfer(('synthetic', 0, 1), 528)
done, used = contact([t], 300); assert not done and t.sent == 0 and used == 300
done, used = contact([t], wire(528)); assert done == [t] and t.sent == 528 and used == wire(528)
print('PASS: additive moments/kernel sums; Gaussian Fourier expectation; optimization; fp16; cut-frame accounting')
