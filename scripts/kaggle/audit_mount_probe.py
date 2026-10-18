"""Audit exact CPU production-import proof before authorizing the staged GPU continuation."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
folder = Path(sys.argv[1]).resolve()
meta = json.loads((folder/'kernel-metadata.json').read_text())
owner = meta['id'].split('/')[0]
probe = (folder.parent.parent/'probes/mount-probe'/folder.name)
if not probe.exists():
    probe = ROOT/'.compat/papers'/sys.argv[2]/'mount-probe'/folder.name
cpu = json.loads((probe/'kernel-metadata.json').read_text())['id']
helper = ROOT/'.agents/skills/kaggle-training-notebook/scripts'
prefix = [sys.executable,str(helper/'kaggle_as.py'),owner,'--']
status = subprocess.run(prefix+['kaggle','kernels','status',cpu],check=True,capture_output=True,text=True)
if 'KernelWorkerStatus.COMPLETE' not in status.stdout:
    print('CPU probe pending or failed:',status.stdout.strip())
    sys.exit(3)
dest = probe/'output'
subprocess.run(prefix+[sys.executable,str(helper/'pull_kernel_output.py'),cpu,'-p',str(dest),'-j','2',
    '--pattern',r'^source/(MOUNT_VERIFIED|resume-import)\.json$'],check=True)
proof = json.loads((dest/'source/MOUNT_VERIFIED.json').read_text())
plan = json.loads((folder/'resume-plan.json').read_text())
assert proof['cpu_only'] and len(proof['imports']) == len(plan)
for item in proof['imports']:
    expected = plan[item.get('arm',item.get('run'))]
    assert item['last_round'] == expected['last_round'] and item['fingerprint'] == expected['fingerprint']
    assert item.get('markers_published_last',item.get('markers_last'))
(folder/'CPU_MOUNT_VERIFIED.json').write_text(json.dumps(dict(kernel=cpu,version=1,**proof),indent=2)+'\n')
print(meta['id'],'CPU exact import/state/SHA gate PASS')
