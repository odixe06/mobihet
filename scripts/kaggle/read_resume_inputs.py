"""Owner-scoped read of continuation inputs; print filenames/counts, never signed URLs."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
HELPER=ROOT/'.agents/skills/kaggle-training-notebook/scripts'
sys.path.insert(0,str(HELPER))
from pull_kernel_output import listing


def main():
    folder=Path(sys.argv[1])
    meta=json.loads((folder/'kernel-metadata.json').read_text())
    plan=json.loads((folder/'resume-plan.json').read_text())
    rows=[]
    for slug in meta['dataset_sources']:
        result=subprocess.run(['kaggle','datasets','files',slug],capture_output=True,text=True,timeout=180)
        assert result.returncode==0 and 'name' in result.stdout.lower(), ('dataset inaccessible',slug)
        rows.append(dict(input=slug,kind='dataset',readable=True))
    for slug in meta['kernel_sources']:
        _,files,_=listing(slug)
        for arm,value in plan.items():
            name=arm if '_s' in arm else f'{arm}_s42'
            r=value['last_round']
            for suffix in ('config.json',f'weights/round_{r:03d}.pt',f'resume/round_{r:03d}.pt',f'complete/round_{r:03d}.json'):
                assert f'runs/{name}/{suffix}' in files, ('parent artifact inaccessible',slug,name,suffix)
        rows.append(dict(input=slug,kind='kernel',readable=True,listed_files=len(files)))
    (folder/'input-access.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(meta['id'],'owner-scoped input read PASS',len(rows),flush=True)


if __name__=='__main__':main()
