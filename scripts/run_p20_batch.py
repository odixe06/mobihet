"""Authorized P20 launch and20min local worker loop; collect/audit early terminals, zero model polling."""
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p-study"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
sys.path.insert(0, str(ROOT))
from scripts.watch_kaggle_batch import query, atomic_json


def collect(spec):
    dest = ROOT / "artifacts/p20" / spec["slug"]; dest.mkdir(parents=True, exist_ok=True)
    logdir = OUT / "collection"; logdir.mkdir(exist_ok=True)
    pattern = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|train_cms_r\d{3}\.npz|diagnostics_rank[01]\.jsonl|external_cms_rank[01]_r\d{3}\.npz|teacher_probe_rank[01]_r\d{3}\.json)$|/models/r\d{3}\.pt$)'
    cmd = [sys.executable, str(SKILL / "kaggle_as.py"), spec["account"], "--", sys.executable,
           str(ROOT / "scripts/run_bounded_python.py"), str(SKILL / "pull_kernel_output.py"),
           f'{spec["account"]}/{spec["slug"]}/{spec["version"]}', "-p", str(dest), "-j", "8", "--pattern", pattern]
    with (logdir / (spec["slug"] + ".collect.log")).open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return {"arm": spec["arm"], "returncode": result.returncode, "root": str(dest)}


def startup(batch, entity, cycle):
    pending = []
    for spec in batch:
        ready = OUT / "startup" / (spec["arm"] + ".json")
        if not ready.exists(): pending.append(spec)
    chunks = [pending[i::4] for i in range(4)]
    def check(item):
        i, chunk = item
        if not chunk: return []
        folder = OUT / "startup" / f"cycle{cycle}-chunk{i}"; folder.mkdir(parents=True, exist_ok=True)
        path = folder / "batch.json"; path.write_text(json.dumps(chunk) + "\n")
        subprocess.run([sys.executable, str(ROOT / "scripts/check_screening_startup.py"), "--batch", str(path),
                        "--out", str(folder), "--entity", entity], check=True, cwd=ROOT)
        records = json.loads((folder / "startup.json").read_text())["runs"]
        for record in records:
            if record["ready"]: atomic_json(OUT / "startup" / (record["arm"] + ".json"), record)
        return records
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(check, enumerate(chunks)))
    ready = len(list((OUT / "startup").glob("P*.json")))
    print("P20_STARTUP", ready, "/", len(batch), flush=True)
    atomic_json(OUT / "startup/progress.json", {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ready": ready, "total": len(batch)})


def main():
    OUT.mkdir(exist_ok=True)
    lock = (OUT / "worker.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    subprocess.run([sys.executable, str(ROOT / "scripts/launch_p20.py")], check=True, cwd=ROOT)
    raw = (OUT / "batch.json").read_bytes(); batch = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    entity = json.loads((OUT / "wandb-identity.json").read_text())["entity"]
    collected, audited, known, cycle = set(), set(), {}, 0
    collection = OUT / "collection"; collection.mkdir(exist_ok=True)
    while True:
        cycle += 1; started = time.monotonic()
        assert (OUT / "batch.json").read_bytes() == raw
        active = [spec for spec in batch if not known.get(spec["arm"], {}).get("terminal")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for status in pool.map(query, active): known[status["arm"]] = status
        snapshot = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cycle": cycle, "batch_sha256": digest,
                    "interval_seconds": 1200, "runs": [known[s["arm"]] for s in batch]}
        atomic_json(OUT / "status.json", snapshot)
        with (OUT / "status-history.jsonl").open("a") as f: f.write(json.dumps(snapshot) + "\n")
        terminal = [s for s in batch if known[s["arm"]]["terminal"] and s["arm"] not in collected]
        if terminal:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(collect, terminal))
            atomic_json(collection / f"cycle{cycle}-pull.json", results)
            for spec, result in zip(terminal, results):
                assert result["returncode"] == 0, f'Pull incomplete {spec["arm"]}; preserve outputs'
                collected.add(spec["arm"])
                if known[spec["arm"]]["status"] != "COMPLETE": continue
                # Audit serially: do not multiply memory use by download concurrency.
                armout = collection / spec["arm"]; armout.mkdir(exist_ok=True)
                path = armout / "batch.json"; path.write_text(json.dumps([spec]) + "\n")
                subprocess.run([sys.executable, str(ROOT / "scripts/audit_p20.py"), "--batch", str(path),
                    "--root", str(ROOT / "artifacts/p20"), "--out", str(armout)], check=True, cwd=ROOT)
                audited.add(spec["arm"])
                print("P20_EARLY_AUDITED", spec["arm"], len(audited), "/32", flush=True)
        atomic_json(OUT / "progress.json", {"at": snapshot["at"], "collected": sorted(collected), "audited": sorted(audited), "total": 32})
        errors = [s for s in snapshot["runs"] if s["status"] in {"ERROR", "CANCEL_ACKNOWLEDGED"}]
        if errors:
            atomic_json(OUT / "attention.json", {"errors": errors, "all_statuses": snapshot, "note": "Other kernels remain untouched"})
            raise RuntimeError("P20 terminal kernel error collected; inspect artifacts, leave other kernels running")
        startup(batch, entity, cycle)
        if len(audited) == len(batch):
            atomic_json(OUT / "complete.json", {"batch_sha256": digest, "audited": sorted(audited), "rounds": 20})
            print("P20_LAUNCH_COLLECT_AUDIT_DONE", flush=True)
            return
        print("P20_WAIT", len(audited), "/32 audited; next status cycle1200s", flush=True)
        time.sleep(max(1, 1200 - (time.monotonic() - started)))


if __name__ == "__main__": main()
