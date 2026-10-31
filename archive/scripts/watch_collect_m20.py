"""Authorized deterministic M20 status wait -> selective pull -> artifact audit. No model calls."""
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/m-study"
SKILL = ".agents/skills/kaggle-training-notebook/scripts"
PATTERN = r"(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|train_cms_r\d{3}\.npz|external_cms_rank[01]_r\d{3}\.npz|diagnostics_rank[01]\.jsonl|models/r\d{3}\.pt)$)"


def pull(spec):
    ref = f"{spec['account']}/{spec['slug']}/{spec['version']}"
    dest = ROOT / "runs/pulls" / spec["slug"]
    inner = [sys.executable, "scripts/run_bounded_python.py", SKILL + "/pull_kernel_output.py", ref,
             "-p", str(dest), "-j", "8", "--pattern", PATTERN, "--passes", "3", "--retries", "3", "--timeout", "90"]
    cmd = [sys.executable, "scripts/run_bounded_python.py", SKILL + "/kaggle_as.py", spec["account"], "--", *inner]
    log = OUT / (spec["slug"] + ".collect.log")
    with log.open("w") as f:
        os.chmod(log, 0o600)
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        if rc:
            token = Path.home() / ".kaggle/accounts" / (spec["account"] + ".mcp-token")
            assert token.is_file()
            f.write("\nRetry same exact output using saved API-token FILE PATH; credential value suppressed.\n")
            f.flush()
            rc = subprocess.run(inner, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                                env=dict(os.environ, KAGGLE_API_TOKEN=str(token))).returncode
    d = dest / spec["slug"]
    expected = ["config.json", "rounds.jsonl", "diagnostics_rank0.jsonl", "diagnostics_rank1.jsonl"]
    expected += [f"cms_r{r:03}.npz" for r in range(1, 21)]
    expected += [f"models/r{r:03}.pt" for r in range(1, 21)]
    expected += [f"train_cms_r{r:03}.npz" for r in (1, 2, 5, 10, 15, 20)]
    expected += [f"external_cms_rank{rank}_r{r:03}.npz" for rank in (0, 1) for r in (1, 10, 20)]
    missing = [name for name in expected if not (d / name).is_file() or (d / name).stat().st_size == 0]
    if not (dest / "provenance/source_manifest.json").is_file():
        missing.append("provenance/source_manifest.json")
    files = [p for p in dest.rglob("*") if p.is_file()]
    record = {"arm": spec["arm"], "ref": ref, "returncode": rc, "expected_artifacts": len(expected),
              "missing": missing, "files": len(files), "bytes": sum(p.stat().st_size for p in files),
              "at": datetime.datetime.now().astimezone().isoformat()}
    (OUT / (spec["slug"] + ".collected.json")).write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    return record


def main():
    os.chdir(ROOT)
    lock = (OUT / "watch-collect.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = OUT / "batch.json"
    batch_bytes = manifest.read_bytes()
    batch = json.loads(batch_bytes)
    assert len(batch) == 16
    assert all(json.loads((OUT / (s["slug"] + ".push.json")).read_text())["ambiguous"] is False for s in batch)
    watched = subprocess.run([sys.executable, "scripts/watch_kaggle_batch.py", "--batch", str(manifest),
                              "--out", str(OUT / "watch-20m"), "--interval", "1200", "--stop-on-error"])
    if watched.returncode == 2:
        attention = json.loads((OUT / "watch-20m/attention.json").read_text())
        assert attention["batch_sha256"] == hashlib.sha256(batch_bytes).hexdigest()
        errors = {r["arm"] for r in attention["runs"] if r["status"] in {"ERROR", "CANCEL_ACKNOWLEDGED"}}
        records = [pull(s) for s in batch if s["arm"] in errors]
        (OUT / "error-collection.json").write_text(json.dumps(records, indent=2))
        raise RuntimeError("Terminal kernel error pulled; other kernels left running; inspect attention.json/error-collection.json")
    assert watched.returncode == 0, "Watcher failed before terminal collection"
    terminal = json.loads((OUT / "watch-20m/terminal.json").read_text())
    assert terminal["batch_sha256"] == hashlib.sha256(batch_bytes).hexdigest()
    assert manifest.read_bytes() == batch_bytes and terminal["terminal_count"] == 16
    assert shutil.disk_usage(ROOT).free > 20 * 2**30, "Need20GiB free before selective M20 collection"
    # Error kernels are pulled too; the inventory/audit will preserve evidence and report incomplete runs.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        inventory = list(pool.map(pull, batch))
    (OUT / "collection.json").write_text(json.dumps(inventory, indent=2))
    assert all(r["returncode"] == 0 and not r["missing"] for r in inventory), "Incomplete pull/run; inspect collection.json"
    assert all(r["status"] == "COMPLETE" for r in terminal["runs"]), "Terminal error kernels collected; no success claim"
    subprocess.run([sys.executable, "scripts/audit_m20.py"], check=True,
                   env=dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2"))
    print("M20_PULL_AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
