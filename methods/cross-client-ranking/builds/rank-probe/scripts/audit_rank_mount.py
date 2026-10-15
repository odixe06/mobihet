"""Verify actual CPU mount, every train/source hash and 20 strict warm models before GPU."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[5];BASE=Path(__file__).resolve().parents[1]
seed=int(sys.argv[1]);rid=f'ccr-probe-s{seed}-20261015-v1';folder=BASE/'notebooks'/rid
probe=BASE/'probes/mount-probe'/rid
meta=json.loads((folder/'kernel-metadata.json').read_text());cpu=json.loads((probe/'kernel-metadata.json').read_text())
assert meta['dataset_sources']==cpu['dataset_sources'] and not cpu['enable_gpu']
helper=ROOT/'.agents/skills/kaggle-training-notebook/scripts';prefix=[sys.executable,str(helper/'kaggle_as.py'),meta['id'].split('/')[0],'--']
status=subprocess.run(prefix+['kaggle','kernels','status',cpu['id']],check=True,capture_output=True,text=True)
if 'KernelWorkerStatus.COMPLETE' not in status.stdout:
    print(status.stdout.strip());sys.exit(3)
dest=probe/'output'
subprocess.run(prefix+[sys.executable,str(helper/'pull_kernel_output.py'),cpu['id'],'-p',str(dest),'-j','2',
    '--pattern',r'^source/MOUNT_VERIFIED\.json$'],check=True)
proof=json.loads((dest/'source/MOUNT_VERIFIED.json').read_text())
inputs=json.loads((folder/'input_manifest.json').read_text())
assert proof['cpu_only'] and proof['round']==2 and proof['models']==20
assert proof['weights_sha']==inputs['warm_weights.pt'] and proof['inputs_verified']==len(inputs)
(folder/'CPU_MOUNT_VERIFIED.json').write_text(json.dumps(dict(kernel=cpu['id'],version=1,**proof),indent=2)+'\n')
print(meta['id'],'CPU mount/SHA/20 strict warm models PASS')
