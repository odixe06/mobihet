"""Read-only batch completion watcher. Sleep locally; never invoke a model or launch a kernel."""
import argparse
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}


def status_from_cli(returncode, output):
    match = re.search(r'has status "KernelWorkerStatus\.([A-Z_]+)"', output)
    return match.group(1) if returncode == 0 and match else "QUERY_ERROR"


def query(spec):
    ref = f"{spec['account']}/{spec['slug']}/{spec.get('version', 1)}"
    command = ["timeout", "--foreground", "--kill-after=5s", "45s", sys.executable,
               str(ROOT / ".agents/skills/kaggle-training-notebook/scripts/kaggle_as.py"), spec["account"],
               "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py"), "kernels", "status", ref]
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        status = status_from_cli(result.returncode, result.stdout)
        return {"arm": spec["arm"], "ref": ref, "status": status,
                "terminal": status in TERMINAL, "query_returncode": result.returncode,
                "checked_at": datetime.datetime.now().astimezone().isoformat()}
    except OSError as exc:
        return {"arm": spec["arm"], "ref": ref, "status": "QUERY_ERROR", "terminal": False,
                "error_type": type(exc).__name__, "checked_at": datetime.datetime.now().astimezone().isoformat()}


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as f:
        json.dump(value, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--interval", type=int, default=1200)
    ap.add_argument("--stop-on-error", action="store_true", help="Wake the owner on a terminal kernel error; leave other kernels running")
    args = ap.parse_args()
    assert args.interval >= 60
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out / "watch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    batch_bytes = args.batch.read_bytes()
    batch_hash = hashlib.sha256(batch_bytes).hexdigest()
    batch = json.loads(batch_bytes)
    assert batch and len({r["arm"] for r in batch}) == len(batch)
    latest = args.out / "latest.json"
    known = {}
    if latest.exists():
        previous = json.loads(latest.read_text())
        assert previous["batch_sha256"] == batch_hash
        known = {r["arm"]: r for r in previous["runs"]}
    cycle = 0
    print(f"Watching {len(batch)} exact Kaggle sessions; interval={args.interval}s; status-only; no model calls.", flush=True)
    while True:
        cycle += 1
        started = time.monotonic()
        assert hashlib.sha256(args.batch.read_bytes()).hexdigest() == batch_hash, "Batch manifest changed"
        pending = [r for r in batch if not known.get(r["arm"], {}).get("terminal", False)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for row in pool.map(query, pending):
                known[row["arm"]] = row
        rows = [known[r["arm"]] for r in batch]
        done = sum(r["terminal"] for r in rows)
        snapshot = {"at": datetime.datetime.now().astimezone().isoformat(), "cycle": cycle,
                    "batch_sha256": batch_hash, "interval_seconds": args.interval,
                    "terminal_count": done, "total": len(rows), "runs": rows}
        atomic_json(latest, snapshot)
        with (args.out / "history.jsonl").open("a") as f:
            f.write(json.dumps(snapshot) + "\n")
            f.flush()
            os.fsync(f.fileno())
        counts = {s: sum(r["status"] == s for r in rows) for s in sorted({r["status"] for r in rows})}
        print(json.dumps({"at": snapshot["at"], "terminal": done, "total": len(rows), "statuses": counts}), flush=True)
        if args.stop_on_error and any(r["status"] in {"ERROR", "CANCEL_ACKNOWLEDGED"} for r in rows):
            atomic_json(args.out / "attention.json", snapshot)
            print("TERMINAL_KERNEL_ERROR_REQUIRES_INSPECTION", flush=True)
            return 2
        if done == len(rows):
            atomic_json(args.out / "terminal.json", snapshot)
            print("All exact sessions terminal; COMPLETE/ERROR are operational states, not artifact audit.", flush=True)
            print("Inspect terminal.json before reporting training success. Do not relaunch or change frozen runs.", flush=True)
            print("ALL_KERNELS_TERMINAL", flush=True)
            return
        time.sleep(max(1, args.interval - (time.monotonic() - started)))


if __name__ == "__main__":
    sys.exit(main())
