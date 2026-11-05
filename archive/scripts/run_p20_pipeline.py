"""Wait for frozen local gates without model calls, prepare once, then authorized32run launch/collect."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p-study"
PREFLIGHT = Path("/home/odixe/.codex/long-jobs/jobs/p20-local-preflight-bankguard-20261101/state.json")


def main():
    while True:
        state = json.loads(PREFLIGHT.read_text())
        if state["status"] in {"succeeded", "failed", "cancelled", "interrupted"}: break
        time.sleep(2)
    assert state["status"] == "succeeded", "Final local preflight failed; do not launch"
    smoke = json.loads((OUT / "smoke-audit.json").read_text())
    assert smoke["source_unchanged_during_preflight"] and smoke["proxy_carrier_3round_bitexact"]
    tests = json.loads((OUT / "tests-current.json").read_text())
    assert tests["tests"] >= 83 and tests["failures"] == tests["errors"] == 0
    for name, digest in tests["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    if not (OUT / "batch.json").exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/inventory_p20.py")], check=True, cwd=ROOT)
        subprocess.run([sys.executable, str(ROOT / "scripts/prepare_p20.py")], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_p20_batch.py")], check=True, cwd=ROOT)
    print("P20_PIPELINE_VERIFIED_DONE", flush=True)


if __name__ == "__main__": main()
