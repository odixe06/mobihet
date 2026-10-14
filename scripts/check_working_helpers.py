"""Import current parent-side helpers and check canonical path reads; never launch training."""
import importlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.project_paths import current_path


def main():
    names=['scripts.kaggle.gen_full_resume_batch','scripts.kaggle.gen_full_validation',
        'scripts.kaggle.gen_full_continuation','scripts.kaggle.gen_hybrid_full',
        'scripts.kaggle.gen_hybrid_dataset_resume','scripts.kaggle.gen_boundary_utility',
        'scripts.kaggle.boundary_logging','scripts.kaggle.boundary_resume','scripts.kaggle.full_resume',
        'scripts.kaggle.collect_boundary_calibration','scripts.check_v3_driver']
    for name in names:importlib.import_module(name)
    paths=['scripts/kaggle/full_resume.py','scripts/check_v3_driver.py','scripts/kaggle/boundary_logging.py',
        'scripts/kaggle/boundary_resume.py','scripts/kaggle/collect_boundary_calibration.py',
        'docs/research/full-h1-h2.md','docs/research/full-hybrid.md']
    for path in paths:assert (ROOT/current_path(path)).is_file(),path
    proof=dict(imports=names,pass_check=True,
        limits='Parent helper imports/path reads only; no train, new notebook/tag or remote push.')
    (ROOT/'docs/operations/workspace-helper-imports-2026-10-14.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('Current helper imports/path reads PASS:',len(names),'modules')


if __name__=='__main__':main()
