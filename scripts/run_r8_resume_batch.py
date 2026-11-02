"""R8 resume watch, pull, and 20-round artifact audit."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/r-study/recovery-r8"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
sys.path.insert(0, str(ROOT))
from scripts.watch_kaggle_batch import query, atomic_json
from scripts import audit_screening

INTERVAL_SECONDS = 1200


def collect(spec):
    dest = ROOT / "artifacts/r20" / spec["run_dir"]
    dest.mkdir(parents=True, exist_ok=True)
    pattern = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|train_cms_r\d{3}\.npz|diagnostics_rank[01]\.jsonl|external_cms_rank[01]_r\d{3}\.npz|teacher_probe_rank[01]_r\d{3}\.json)$|/models/r\d{3}\.pt$|/recovery/rank[01]\.jsonl$)'
    collection = OUT / "collection"
    collection.mkdir(exist_ok=True)
    log_file = collection / f"{spec['slug']}.collect.log"
    with log_file.open("w") as log:
        p = subprocess.run([sys.executable, str(SKILL / "kaggle_as.py"), spec["account"], "--", sys.executable,
            str(ROOT / "scripts/run_bounded_python.py"), str(SKILL / "pull_kernel_output.py"),
            f'{spec["account"]}/{spec["slug"]}/{spec["version"]}', "-p", str(dest), "-j", "8", "--pattern", pattern],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return {"arm": spec["arm"], "returncode": p.returncode, "root": str(dest)}


def audit_r8(dest, spec):
    print("Auditing R8 20 rounds using scripts/audit_r20.py...", flush=True)
    audit_batch = [spec | {"slug": spec["run_dir"]}]
    (OUT / "audit-spec.json").write_text(json.dumps(audit_batch, indent=2))
    p = subprocess.run([sys.executable, str(ROOT / "scripts/audit_r20.py"),
        "--batch", str(OUT / "audit-spec.json"),
        "--root", str(ROOT / "artifacts/r20"),
        "--out", str(OUT / "audit")], check=True, cwd=ROOT, capture_output=True, text=True)
    print(p.stdout.strip(), flush=True)
    return json.loads((OUT / "audit/audit.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect-only", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    if not a.collect_only:
        subprocess.run([sys.executable, str(ROOT / "scripts/launch_r8_resume.py")], check=True, cwd=ROOT)

    batch = json.loads((OUT / "batch.json").read_text())
    assert len(batch) == 1
    spec = batch[0]
    cmd = [sys.executable, str(SKILL / "kaggle_as.py"), spec["account"], "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]

    cycle = 0
    ref = f"{spec['account']}/{spec['slug']}/1"

    while True:
        cycle += 1
        st = query(spec)
        status = st["status"]
        print(f"Cycle {cycle}: {ref} -> {status}", flush=True)
        atomic_json(OUT / "status.json", {
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "cycle": cycle,
            "ref": ref,
            "status": status,
            "terminal": st["terminal"]
        })

        if status == "COMPLETE":
            print(f"Kernel {ref} COMPLETE, collecting output...", flush=True)
            cr = collect(spec)
            assert cr["returncode"] == 0, f"Collection failed for {ref}"
            audit_res = audit_r8(cr["root"], spec)
            atomic_json(OUT / "closure.json", {
                "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "arm": spec["arm"],
                "status": "PASS_BOUNDED",
                "endpoint": 20,
                "ref": ref
            })
            print("R8_RESUME_PIPELINE_DONE", flush=True)
            return

        if st["terminal"]:
            raise RuntimeError(f"Kernel {ref} terminated with failure status: {status}")

        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
