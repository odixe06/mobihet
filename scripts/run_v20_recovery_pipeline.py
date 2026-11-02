"""Verified CE guard preflight -> fresh quota/phase lock -> two recovery runs and artifact audit."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    state = json.loads(Path('/home/odixe/.codex/long-jobs/jobs/v20-ce-recovery-preflight-20261102/state.json').read_text())
    assert state["status"] == "succeeded" and state["marker_seen"]
    for script in ("prepare_v20_recovery.py", "run_v20_recovery_batch.py"):
        subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    print("V20_CE_RECOVERY_PIPELINE_DONE", flush=True)


if __name__ == "__main__": main()
