"""Install scoped routing/timeouts before saved-account authentication and its child command."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.argv.insert(1, str(ROOT / ".agents/skills/kaggle-training-notebook/scripts/kaggle_as.py"))
runpy.run_path(str(ROOT / "scripts/run_bounded_python.py"), run_name="__main__")
