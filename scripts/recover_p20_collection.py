"""Collect the already-pushed P20 versions; never launch or modify a Kaggle kernel."""
import concurrent.futures
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_p20_batch import collect
from scripts.watch_kaggle_batch import atomic_json, query

OUT = ROOT / "runs/p-study"
RECOVERY = OUT / "recovery-v1"


def main():
    RECOVERY.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = (OUT / "batch.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == json.loads((ROOT / "docs/research/p20-launch.json").read_text())["batch_sha256"]
    batch = json.loads(raw)
    previous = json.loads((OUT / "status.json").read_text())
    assert previous["batch_sha256"] == digest
    known = {s["arm"]: s for s in previous["runs"]}
    ledger_path = RECOVERY / "collection.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
    # The failed pipeline already pulled P21. Reuse only with its actual evidence present.
    old = json.loads((OUT / "progress.json").read_text())
    for spec in batch:
        arm, slug = spec["arm"], spec["slug"]
        dest = ROOT / "artifacts/p20" / slug
        required = [dest / "provenance/source_manifest.json", dest / slug / "config.json", dest / (slug + ".log")]
        if arm not in ledger and arm in old["collected"] and known[arm]["status"] == "ERROR" and all(p.is_file() for p in required):
            ledger[arm] = {"status": "ERROR", "pulled": True, "audit": "not_complete", "reused_pipeline_pull": True}
    atomic_json(ledger_path, ledger)
    # Preserve the original 20-minute status cadence, including the first recovery cycle.
    checked = max(datetime.fromisoformat(s["checked_at"]).timestamp() for s in previous["runs"])
    delay = max(0, checked + 1200 - time.time())
    print("P20_RECOVERY_COLLECT_ONLY", len(batch), "versions; initial_wait_seconds", round(delay), flush=True)
    if delay: time.sleep(delay)
    cycle = 0
    while True:
        cycle += 1
        started = time.monotonic()
        assert (OUT / "batch.json").read_bytes() == raw
        pending = [s for s in batch if not known[s["arm"]]["terminal"]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(query, pending): known[row["arm"]] = row
        snapshot = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cycle": cycle,
                    "batch_sha256": digest, "interval_seconds": 1200, "runs": [known[s["arm"]] for s in batch]}
        atomic_json(RECOVERY / "status.json", snapshot)
        with (RECOVERY / "history.jsonl").open("a") as f: f.write(json.dumps(snapshot) + "\n")
        terminal = [s for s in batch if known[s["arm"]]["terminal"] and s["arm"] not in ledger]
        assert shutil.disk_usage(ROOT / "artifacts/p20").free > 40 * 2**30, "Need40GiB free before remaining P20 pulls"
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(collect, terminal))
        atomic_json(RECOVERY / f"cycle{cycle}-pull.json", results)
        for spec, result in zip(terminal, results):
            arm = spec["arm"]
            if result["returncode"] != 0:
                print("P20_RECOVERY_PULL_RETRY", arm, flush=True)
                continue
            status = known[arm]["status"]
            row = {"status": status, "pulled": True, "audit": "not_complete"}
            if status == "COMPLETE":
                folder = RECOVERY / arm
                folder.mkdir(exist_ok=True)
                path = folder / "batch.json"
                path.write_text(json.dumps([spec]) + "\n")
                with (folder / "audit.log").open("w") as log:
                    result = subprocess.run([sys.executable, str(ROOT / "scripts/audit_p20.py"), "--batch", str(path),
                        "--root", str(ROOT / "artifacts/p20"), "--out", str(folder)], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                row.update(audit="PASS" if result.returncode == 0 else "FAILED", audit_returncode=result.returncode)
            ledger[arm] = row
            atomic_json(ledger_path, ledger)
            print("P20_RECOVERY_COLLECTED", arm, status, row["audit"], flush=True)
        print("P20_RECOVERY_PROGRESS", len(ledger), "/", len(batch), flush=True)
        if len(ledger) == len(batch):
            failed = {k: v for k, v in ledger.items() if v["audit"] != "PASS"}
            atomic_json(RECOVERY / "terminal-audit.json", {"batch_sha256": digest, "runs": ledger, "attention": failed})
            print("P20_RECOVERY_ALL_TERMINAL_COLLECTED", flush=True)
            if failed: raise RuntimeError("Terminal evidence collected; failed or incomplete runs require inspection")
            print("P20_RECOVERY_ALL_AUDITED", flush=True)
            return
        time.sleep(max(1, 1200 - (time.monotonic() - started)))


if __name__ == "__main__": main()
