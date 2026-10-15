"""Actual loss gradients, wire corruption, checkpoint state and next-round resume."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[5]
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import scripts
sys.path.insert(0, str(BASE/'source/proj'))
from tail_math import C, B, EDGES, histogram, snapshot, decode, weights
from scripts.check_v3_driver import equal


def main():
    torch.set_num_threads(1)
    loss = torch.tensor([.1, .4, .9, 2.], requires_grad=True)
    zeta = torch.tensor([.4]*4)
    objective = (.75*loss+.25*(zeta+(loss-zeta).relu()/.25)).mean()
    objective.backward()
    assert torch.equal(loss.grad*4, torch.tensor([.75, .75, 1.75, 1.75]))
    y = np.tile(np.arange(C), 128)
    ce = np.tile(np.geomspace(1e-4, 4, 128), C).astype('float32')
    local = [histogram(ce, y) for _ in range(5)]
    identity = hashlib.sha256(EDGES.tobytes()).digest()[:16]
    z, mask, prior, ups, down = snapshot(local, 3, identity)
    assert mask.all() and np.isclose(prior.sum(), 1)
    assert all(len(x) == 2272 for x in ups) and len(down) == 162
    assert set(weights(np.array([.1, 4.]), np.array([0, 0]), z, mask)[0]) <= {.75, 1.75}
    for wire in (down[:-1]+bytes([down[-1]^1]), down):
        try:
            decode(wire, 2, 4, identity)
        except AssertionError:
            pass
        else:
            raise AssertionError('bad version or CRC accepted')
    overflow = [histogram(np.full(len(y), 9.), y) for _ in range(5)]
    assert not snapshot(overflow, 3, identity)[1].any()
    driver = BASE/'source/proj/tail_driver.py'
    tag = sys.argv[1] if len(sys.argv)>1 else 'initial'
    reference = BASE/f'artifacts/smoke-reference-{tag}'
    resumed = BASE/f'artifacts/smoke-resumed-{tag}'
    assert not reference.exists() and not resumed.exists(), 'use existing proof or fresh smoke paths'
    for arm in ('ct_fl', 'ct_classmass', 'ct_dml', 'ct_gaussian'):
        command = [sys.executable, str(driver), '--arm', arm, '--smoke', '--out']
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        subprocess.run(command+[str(reference)], check=True, cwd=ROOT, env=env)
        subprocess.run(command+[str(resumed), '--stop_after', '2'], check=True, cwd=ROOT, env=env)
        subprocess.run(command+[str(resumed), '--resume'], check=True, cwd=ROOT, env=env)
        for r in range(1, 4):
            for kind in ('weights', 'resume'):
                a = torch.load(reference/f'{arm}_s42/{kind}/round_{r:03d}.pt', map_location='cpu', weights_only=True)
                b = torch.load(resumed/f'{arm}_s42/{kind}/round_{r:03d}.pt', map_location='cpu', weights_only=True)
                assert equal(a, b), (arm, r, kind)
        for folder in (reference, resumed):
            run = folder/f'{arm}_s42'
            record = json.loads((run/'result.json').read_text())
            assert record['complete'] and len(record['history']) == 3
            for row in record['history']:
                state = torch.load(run/f'resume/round_{row["round"]:03d}.pt', map_location='cpu', weights_only=True)
                for k, item in enumerate(row['clients']):
                    p = state['algorithm']['mass'][k]/item['applied']+1/item['rows']
                    assert torch.allclose((p/p.sum()).log().float(), state['algorithm']['log_pi_effective'][k], atol=1e-6)
    (BASE/'probes/local-preflight.json').write_text(json.dumps(dict(
        gradient=True, wire=True, overflow_fallback=True, exact_resume_arms=4,
        effective_mass=True, scope='bounded fixture on local GPU; not T4 cost/utility/whole-data evidence'), indent=2)+'\n')
    print('ClassTail gradients/wire/4-arm exact resume/mass PASS')


if __name__ == '__main__':
    main()
