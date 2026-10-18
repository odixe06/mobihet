"""FLOPs per sample of DAGSNet (one forward; one train step = forward + backward), counted by torch FlopCounterMode on CPU.

    conda run -n nckh python scripts/count_dagsnet_flops.py
"""
import torch
from torch.utils.flop_counter import FlopCounterMode
from mh import pilot as P

torch.manual_seed(42); model = P.make('dagsnet')
params = sum(p.numel() for p in model.parameters())
x = torch.randn(512, P.D)
model.eval()
with FlopCounterMode(display=False) as counter, torch.no_grad():
    model(x)
forward = counter.get_total_flops() / len(x)
model.train()
with FlopCounterMode(display=False) as counter:
    out = model(x); (out[0] if isinstance(out, tuple) else out).float().sum().backward()
train = counter.get_total_flops() / len(x)
print('params', params, 'features', P.D, 'classes', P.C, 'forward FLOPs/sample', round(forward),
      'train FLOPs/sample', round(train), 'train/forward', round(train / forward, 2))
