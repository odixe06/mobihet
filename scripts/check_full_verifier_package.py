"""Check generated resume verifier imports without the checkout's ambient scripts path."""
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path
import nbformat

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'papers/full-validation'


def main():
    folders=sorted((BASE/'notebook').glob('*cont01*'))
    assert len(folders)==4
    work=BASE/'smoke/verifier-package-20261013'
    assert not work.exists(),'fresh package check required'
    support=[]
    for folder in folders:
        meta=json.loads((folder/'kernel-metadata.json').read_text())
        notebook=nbformat.read(folder/meta['code_file'],as_version=4)
        assignments={node.targets[0].id:ast.literal_eval(node.value)
            for node in ast.parse(notebook.cells[2].source).body if isinstance(node,ast.Assign)
            and isinstance(node.targets[0],ast.Name) and node.targets[0].id in ('CODE','PROJ','VERIFIER_SUPPORT')}
        support.append(assignments['VERIFIER_SUPPORT'])
    assert len(set(support))==1
    source=BASE/'input/catbaochau'
    for name in json.loads((source/'source_manifest.json').read_text()):
        target=work/'mh'/name if name in assignments['CODE'] else work/'papers/full-validation/proj'/name if name in assignments['PROJ'] else work/'knowledge'/name
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
    (work/'scripts').mkdir(exist_ok=True)
    (work/'scripts/check_v3_driver.py').write_text(support[0])
    code="import sys;from pathlib import Path;p=Path(sys.argv[1]);sys.path.insert(0,str(p/'papers/full-validation/proj'));from full_verify import verify;assert callable(verify);import check_v3_driver;assert Path(check_v3_driver.__file__).resolve()==p/'scripts/check_v3_driver.py';print('PASS isolated frozen verifier package; four generated notebooks')"
    subprocess.run([sys.executable,'-I','-c',code,str(work.resolve())],cwd=work,check=True)


if __name__=='__main__':main()
