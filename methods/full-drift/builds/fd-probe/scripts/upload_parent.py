"""Round 17 --parent: upload an audited run's committed epoch (config, marker, weights) flat as a private Kaggle dataset; verify
names + sizes (the notebook and fd.py check SHA).

    python methods/full-drift/builds/fd-probe/scripts/upload_parent.py <owner/slug> <run dir> <round> <scratch work dir>
"""
import json, shutil, subprocess, sys, time
from pathlib import Path
H = '.agents/skills/kaggle-training-notebook/scripts/kaggle_as.py'
dataset, run, r, work = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
owner, slug = dataset.split('/')
kg = lambda *a: subprocess.run([sys.executable, H, owner, '--', 'kaggle', *a], capture_output=True, text=True)
if work.exists(): shutil.rmtree(work)
work.mkdir(parents=True)
files = {'config.json': run / 'config.json', f'round_{r:03d}.json': run / f'complete/round_{r:03d}.json', f'round_{r:03d}.pt': run / f'weights/round_{r:03d}.pt'}
for name, src in files.items(): shutil.copyfile(src, work / name)
(work / 'dataset-metadata.json').write_text(json.dumps(dict(title=slug, id=dataset, licenses=[dict(name='CC0-1.0')])))
status = kg('datasets', 'status', dataset)
if 'ready' not in status.stdout:
    c = kg('datasets', 'create', '-p', str(work), '-q'); print(c.stdout[-400:], c.stderr[-400:], flush=True)
    for _ in range(90):
        status = kg('datasets', 'status', dataset)
        if 'ready' in status.stdout: break
        time.sleep(20)
assert 'ready' in status.stdout, status.stdout + status.stderr
listing = kg('datasets', 'files', dataset, '--csv').stdout
seen = {l.split(',')[0]: l.split(',')[1] for l in listing.splitlines() if l and not l.startswith('name,')}
print('listing', seen, {n: (work / n).stat().st_size for n in files}, flush=True)
assert set(files) <= set(seen), 'missing files'
print('VERIFIED', dataset, flush=True)
