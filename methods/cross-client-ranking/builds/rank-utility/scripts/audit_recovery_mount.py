"""Require mounted committed r3 and compatible strict artifact audit before recovery GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[3]
seed=int(sys.argv[1]);rid=f'cr-u-s{seed}-20261015-v2'
folder=BASE/'notebooks'/rid;probe=BASE/'probes/mount-probe'/rid
meta=json.loads((folder/'kernel-metadata.json').read_text());cpu=json.loads((probe/'kernel-metadata.json').read_text())
assert meta['dataset_sources']==cpu['dataset_sources'] and not cpu['enable_gpu']
helper=ROOT/'.agents/skills/kaggle-training-notebook/scripts';prefix=[sys.executable,str(helper/'kaggle_as.py'),'minhtriethihi','--']
status=subprocess.run(prefix+['kaggle','kernels','status',cpu['id']],check=True,capture_output=True,text=True)
if 'KernelWorkerStatus.COMPLETE' not in status.stdout:print(status.stdout.strip());sys.exit(3)
subprocess.run(prefix+[sys.executable,str(helper/'pull_kernel_output.py'),cpu['id'],'-p',str(probe/'output'),'-j','2',
    '--pattern',r'^source/RECOVERY_MOUNT_VERIFIED\.json$'],check=True)
proof=json.loads((probe/'output/source/RECOVERY_MOUNT_VERIFIED.json').read_text())
assert proof['compat_sha']==hashlib.sha256((BASE/'scripts/rank_bytes_compat.py').read_bytes()).hexdigest()
assert len(proof['audits'])==2 and all(a['logical_rounds']==3 and a['committed_rounds']==1 and a['metric_checks']==40 for a in proof['audits'])
assert proof['files']==len(json.loads((BASE/'artifacts/input'/rid/'resume_manifest.json').read_text()))
(folder/'CPU_MOUNT_VERIFIED.json').write_text(json.dumps(dict(kernel=cpu['id'],version=1,**proof),indent=2)+'\n')
print(meta['id'],'CPU exact r3/bytes/state/wire/predictions audit PASS')
