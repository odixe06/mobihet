"""Check real smoke artifacts, exact resume, and rejection of altered stored metrics."""
import json, subprocess, sys, tempfile
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'scripts'))
from verify_new_fl import verify

for d in sorted((ROOT / 'papers/new-fl/smoke').glob('*_s42')):
    verify(d)
subprocess.run([sys.executable, '-m', 'mh.new_fl', '--arm', 'smoke_kme', '--method', 'kme', '--smoke',
                '--out', 'papers/new-fl/resume_reference'], cwd=ROOT, check=True)
a = ROOT / 'papers/new-fl/smoke/smoke_kme_s42'
b = ROOT / 'papers/new-fl/resume_reference/smoke_kme_s42'
verify(b)
for r in range(1, 4):
    x = torch.load(a / 'weights' / f'round_{r:03d}.pt', weights_only=True, map_location='cpu')
    y = torch.load(b / 'weights' / f'round_{r:03d}.pt', weights_only=True, map_location='cpu')
    assert all(torch.equal(sa[k], sb[k]) for sa, sb in zip(x['models'], y['models']) for k in sa), ('resume mismatch', r)
with tempfile.TemporaryDirectory(prefix='mobihet-audit-') as tmp:
    d = Path(tmp)
    for f in a.iterdir():
        if f.name != 'result.json':
            (d / f.name).symlink_to(f.resolve(), target_is_directory=f.is_dir())
    result = json.loads((a / 'result.json').read_text()); result['history'][-1]['loss'] = 999.
    (d / 'result.json').write_text(json.dumps(result))
    try:
        verify(d)
    except AssertionError:
        pass
    else:
        raise AssertionError('tampered result accepted')
print('PASS: all six real-data smoke arms, strict reload, bit-exact resume, tampered artifact rejected')
