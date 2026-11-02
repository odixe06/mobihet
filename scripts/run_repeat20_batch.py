"""REPEAT20 one-shot launch, local20min watch, early terminal pull/audit; collect-only recovery after failure."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/r-repeat"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
sys.path.insert(0, str(ROOT))
from scripts.watch_kaggle_batch import query, atomic_json


def collect(spec):
    dest = ROOT / "artifacts/r20" / spec["slug"]; dest.mkdir(parents=True, exist_ok=True)
    pattern = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|train_cms_r\d{3}\.npz|diagnostics_rank[01]\.jsonl|external_cms_rank[01]_r\d{3}\.npz|teacher_probe_rank[01]_r\d{3}\.json)$|/models/r\d{3}\.pt$|/recovery/rank[01]\.jsonl$)'
    with (OUT / "collection" / (spec["slug"] + ".collect.log")).open("w") as log:
        p = subprocess.run([sys.executable, str(SKILL / "kaggle_as.py"), spec["account"], "--", sys.executable,
            str(ROOT / "scripts/run_bounded_python.py"), str(SKILL / "pull_kernel_output.py"),
            f'{spec["account"]}/{spec["slug"]}/{spec["version"]}', "-p", str(dest), "-j", "8", "--pattern", pattern],
            stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    return {"arm": spec["arm"], "returncode": p.returncode, "root": str(dest)}


def notify_attention(names):
    """Wake owner once on a new failure while this worker keeps healthy kernels supervised."""
    for arm in sorted(set(names)):
        receipt = OUT / "collection" / f"{arm}.attention-notify.json"
        if receipt.exists(): continue
        name = f"repeat20-attention-{arm.lower()}-20261102"
        log = OUT / "collection" / f"{arm}.attention.log"
        atomic_json(receipt, {"job": name, "intent": True, "arm": arm, "attention": str(OUT / "attention.json")})
        controller = Path("/home/odixe/.codex/skills/codex-long-jobs/scripts/codex-long-jobs")
        cmd = ["bash", str(controller), "start", "--name", name, "--log", str(log), "--success-pattern",
            "^REPEAT20_ATTENTION_RECORDED$", "--delivery", "auto", "--viewer", "none", "--", sys.executable, "-c",
            "from pathlib import Path; print(Path('runs/r-repeat/attention.json').read_text()); print('REPEAT20_ATTENTION_RECORDED')"]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        atomic_json(receipt, {"job": name, "arm": arm, "returncode": p.returncode, "controller": p.stdout,
            "scope": "Failure notification only; main worker still owns live watch/collect"})


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--collect-only", action="store_true"); a = ap.parse_args()
    lock = (OUT / "worker.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not a.collect_only: subprocess.run([sys.executable, str(ROOT / "scripts/launch_repeat20.py")], check=True, cwd=ROOT)
    raw = (OUT / "batch.json").read_bytes(); batch = json.loads(raw)
    assert len(batch) == 4 and (OUT / "launch-complete.json").exists(), "Partial push: inspect receipts, never launch blindly"
    progress_path = OUT / "progress.json"
    previous = json.loads(progress_path.read_text()) if progress_path.exists() else {"runs": {}}
    known, results = {}, previous["runs"]
    for spec in batch:
        if results.get(spec["arm"], {}).get("terminal"): known[spec["arm"]] = results[spec["arm"]]["state"]
    collection = OUT / "collection"; collection.mkdir(exist_ok=True)
    cycle = 0
    while True:
        cycle += 1; started = time.monotonic()
        assert (OUT / "batch.json").read_bytes() == raw
        active = [r for r in batch if not known.get(r["arm"], {}).get("terminal")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for state in pool.map(query, active): known[state["arm"]] = state
        snapshot = {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cycle": cycle, "interval_seconds": 1200,
                    "batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": [known[r["arm"]] for r in batch]}
        atomic_json(OUT / "status.json", snapshot)
        with (OUT / "status-history.jsonl").open("a") as f: f.write(json.dumps(snapshot) + "\n")
        terminal = [r for r in batch if known[r["arm"]]["terminal"] and not results.get(r["arm"], {}).get("pulled")]
        new_failures = []
        if terminal:
            assert shutil.disk_usage(ROOT).free > max(3, 2 * len(terminal)) * 2**30
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: pulled = list(pool.map(collect, terminal))
            for spec, result in zip(terminal, pulled):
                name = spec["arm"]; state = known[name]
                result.update(state=state, terminal=True, pulled=result["returncode"] == 0, audited=False)
                results[name] = result
                if result["returncode"] != 0 or state["status"] != "COMPLETE": new_failures.append(name); continue
        for spec in batch:
            name = spec["arm"]
            result = results.get(name, {})
            if not result.get("pulled") or result.get("audited") or known[name]["status"] != "COMPLETE": continue
            armout = collection / name; armout.mkdir(exist_ok=True)
            path = armout / "batch.json"; atomic_json(path, [spec])
            p = subprocess.run([sys.executable, str(ROOT / "scripts/audit_r20.py"), "--batch", str(path),
                "--root", str(ROOT / "artifacts/r20"), "--out", str(armout)], cwd=ROOT)
            if p.returncode:
                result["audit_error"] = True
                new_failures.append(name)
            else:
                result["audited"] = True
                result.pop("audit_error", None)
                result["report_sha256"] = {n: hashlib.sha256((armout / n).read_bytes()).hexdigest()
                    for n in ["audit.json", "diagnostic-audit.json"]}
                print("REPEAT20_EARLY_AUDITED", name, flush=True)
        atomic_json(progress_path, {"batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": results})
        if new_failures:
            atomic_json(OUT / "attention.json", {"new_failures": new_failures, "snapshot": snapshot,
                "note": "Remote healthy kernels left running; inspect failure, use collect-only worker for remainder"})
            print("REPEAT20_ATTENTION", new_failures, "healthy kernels remain supervised", flush=True)
            notify_attention(new_failures)
        audited = sum(r.get("audited", False) for r in results.values())
        print("REPEAT20_PROGRESS", audited, "/4 audited", flush=True)
        if all(s["terminal"] for s in known.values()):
            atomic_json(OUT / "closure.json", {"batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": results,
                "full20_pass": audited, "all4_classified": True})
            assert audited == 4, "All terminal classified; failed arms still need separate verified recovery"
            print("REPEAT20_LAUNCH_COLLECT_AUDIT_DONE", flush=True)
            return
        startup = OUT / "startup/startup.json"
        if not startup.exists() or not json.loads(startup.read_text())["all_ready"]:
            subprocess.run([sys.executable, str(ROOT / "scripts/check_screening_startup.py"), "--batch", str(OUT / "batch.json"),
                "--out", str(OUT / "startup"), "--entity", "21522798-uit"], check=True, cwd=ROOT)
        time.sleep(max(1, 1200 - (time.monotonic() - started)))


if __name__ == "__main__": main()
