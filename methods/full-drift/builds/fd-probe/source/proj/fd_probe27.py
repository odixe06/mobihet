"""Support oracle: replace inversion with real pooled input, identical synthetic class schedule and head-training steps.

Unreachable under R3. Diagnosis only: if this fixes transfer, private capacity/optimizer alone cannot explain the gap.
Gaussian matched controls remain unchanged. No official test, no real rows in an algorithm payload.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

import fd_probe24 as probe
from fd_probe23 import sha


def main():
    fx = Path(sys.argv[sys.argv.index('--fixture') + 1])
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    manifest = json.loads((fx / 'manifest.json').read_text())
    x = np.concatenate([np.load(fx / f'train_{k}_X.npy') for k in range(20)])
    y = np.concatenate([np.load(fx / f'train_{k}_y.npy') for k in range(20)])
    index = [np.where(y == c)[0] for c in range(16)]

    def real_bank(model, initial, targets, projection, input_scale, feature_scale, zero, steps):
        gen = torch.Generator(device=initial.device).manual_seed(24)
        labels = torch.multinomial(torch.ones(16, device=initial.device) / 16, len(initial), replacement=True, generator=gen)
        labels_cpu = labels.cpu().numpy(); rows = np.empty((len(initial), 66), dtype='float32')
        random = np.random.default_rng(27)
        for c in range(16):
            slots = np.where(labels_cpu == c)[0]
            chosen = random.choice(index[c], len(slots), replace=True)
            assert (y[chosen] == c).all()
            rows[slots] = x[chosen]
        result = torch.from_numpy(rows).to(initial.device); result[:, zero] = 0
        return result

    probe.invert = real_bank
    probe.main()
    result = json.loads(out.read_text())
    result.update(support_override='inversion fields are pooled-real oracle with identical class schedule; NOT deployable synthetic',
                  oracle_source_sha=sha(__file__), scope=result['scope'] + '; real-input oracle diagnostic, R3-ineligible')
    out.write_text(json.dumps(result, indent=1) + '\n')


if __name__ == '__main__': main()
