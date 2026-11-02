"""Authorized O20 one-shot launch, startup,20min worker watch, collection and replay audit."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/o-study"


def phase(name):
    target = OUT / "progress.json"; tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps({"phase": name, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}) + "\n")
    tmp.replace(target)


phase("launch")
subprocess.run([sys.executable, str(ROOT / "scripts/run_bounded_python.py"), str(ROOT / "scripts/launch_o20.py")], check=True, cwd=ROOT)
phase("startup_watch_collect_audit")
entity = json.loads((OUT / "wandb-identity.json").read_text())["entity"]
subprocess.run([sys.executable, str(ROOT / "scripts/watch_collect_screening.py"), "--batch", str(OUT / "batch.json"),
    "--out", str(OUT / "collection"), "--root", str(ROOT / "artifacts/o20"), "--entity", entity,
    "--auditor", str(ROOT / "scripts/audit_o20.py")], check=True, cwd=ROOT)
phase("complete")
print("O20_LAUNCH_COLLECT_AUDIT_DONE", flush=True)
