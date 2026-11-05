"""V20 authorized whole-phase quota -> immutable private notebooks -> launch/watch/pull/audit."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert not (ROOT / "runs/v-study/batch.json").exists(), "Inspect immutable batch/receipts before retry"
    for script in ("inventory_v20.py", "prepare_v20.py", "run_v20_batch.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    print("V20_PIPELINE_DONE", flush=True)


if __name__ == "__main__": main()
