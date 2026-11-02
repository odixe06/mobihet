"""Complete the authorized four-repeat phase without disturbing active replay/validation."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    state = json.loads(Path('/home/odixe/.codex/long-jobs/jobs/repeat20-new-account-inventory-20261102/state.json').read_text())
    assert state["status"] == "succeeded" and state["marker_seen"]
    assert not (ROOT / "runs/r-repeat/batch.json").exists(), "Inspect existing frozen phase before retry"
    for script in ("prepare_repeat20.py", "run_repeat20_batch.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    print("REPEAT20_PIPELINE_DONE", flush=True)


if __name__ == "__main__": main()
