"""Pilot fixture for screening H1-H5 (docs/research/pilot-plan.md): 20 clients, alpha=0.5, 2.5% of each client's
non-holdout rows (partition size ratio kept, >=64 rows/class reserved), plus the archive holdouts `old`
(temporal last 2% per class x scenario) and `group` (unseen receiver groups), 65,536 rows each. Test20% untouched.

    python scripts/run_local_guarded.py python scripts/build_pilot_fixture.py      (env nckh)
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from mh.prepare import prepare

meta = json.loads((ROOT / 'knowledge/meta.json').read_text())
prepare(ROOT.parent / 'dataset/fl_client/alpha05/20_client', ROOT / 'papers/pilot/fixture', meta['feature_cols'],
        meta['class_names'], eval_cap=65536, clients=tuple(range(20)), train_frac=0.025)
