"""Push frozen, preflighted N20 once per exact notebook; ambiguous pushes require inspection."""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/n-study'
SKILL=ROOT/'.agents/skills/kaggle-training-notebook/scripts'


def main():
    raw=(OUT/'batch.json').read_bytes();batch=json.loads(raw);assert len(batch)==8
    assert json.loads((OUT/'smoke-audit.json').read_text())['r1_four_cells_bitexact']
    spec=importlib.util.spec_from_file_location('ka',SKILL/'kaggle_account.py');ka=importlib.util.module_from_spec(spec);spec.loader.exec_module(ka)
    assert ka.active_user()=='dinhan25'
    for row in batch:
        slug=row['slug'];folder=ROOT/'notebooks'/slug;nb=folder/(slug+'.ipynb')
        assert hashlib.sha256(nb.read_bytes()).hexdigest()==row['notebook_sha256']
        meta=json.loads((folder/'kernel-metadata.json').read_text());assert meta['is_private'] is True
        receipt=OUT/(slug+'.push.json');intent=OUT/(slug+'.push-intent.json')
        if receipt.exists():
            old=json.loads(receipt.read_text());assert old['notebook_sha256']==row['notebook_sha256']
            print('ALREADY_PUSHED',slug,flush=True);continue
        assert not intent.exists(),f'Inspect ambiguous previous push before retry: {slug}'
        q=ka.account_quota(row['account']);assert q['gpu_left']>=6.9 and q['gpu_reserved']==0,q
        cmd=[sys.executable,str(SKILL/'kaggle_as.py'),row['account'],'--',sys.executable,str(ROOT/'scripts/kaggle_cli_bounded.py')]
        access=subprocess.run(['timeout','--foreground','--kill-after=5s','90s',*cmd,'datasets','files','-d','odixe0502/veremi-nextgen-v4-scaled-per-vehicle'],capture_output=True,text=True)
        assert access.returncode==0,f'Dataset access failed {row["account"]}: rc{access.returncode}'
        intent.write_text(json.dumps({'at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'ref':f'{row["account"]}/{slug}/1','notebook_sha256':row['notebook_sha256'],'quota':q},indent=2)+'\n')
        result=subprocess.run(['timeout','--foreground','--kill-after=10s','240s',*cmd,'kernels','push','-p',str(folder)],capture_output=True,text=True)
        log=OUT/(slug+'.push.log');log.write_text(result.stdout+result.stderr);os.chmod(log,0o600)
        assert result.returncode==0 and re.search(r'Kernel version 1 successfully pushed',result.stdout),f'Push ambiguous/failed {slug}: inspect private log, do not retry blindly'
        receipt.write_text(json.dumps({'at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'ref':f'{row["account"]}/{slug}/1','returncode':0,'notebook_sha256':row['notebook_sha256']},indent=2)+'\n')
        print('PUSHED',row['arm'],row['account'],slug,'version1',flush=True)
        assert (OUT/'batch.json').read_bytes()==raw
    assert ka.active_user()=='dinhan25'
    (OUT/'launch-complete.json').write_text(json.dumps({'batch_sha256':hashlib.sha256(raw).hexdigest(),'runs':8,'global_login':'dinhan25','at':time.strftime('%Y-%m-%dT%H:%M:%S%z')},indent=2)+'\n')
    print('N20_ALL_8_PUSHED',flush=True)


if __name__=='__main__':main()
