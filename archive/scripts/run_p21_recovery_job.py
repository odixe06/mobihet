"""One-shot P21 launch, deterministic20min monitoring, terminal pull and full20round audit."""
import concurrent.futures
import csv
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p-study/recovery-p21"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
sys.path.insert(0, str(ROOT))
from scripts.run_p20_t4_diagnostic import owned_refs
from scripts.watch_kaggle_batch import atomic_json, query, status_from_cli


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    lock = (OUT / "worker.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    raw = (OUT / "batch.json").read_bytes(); spec, = json.loads(raw)
    assert spec["arm"] == "P21R1" and spec["endpoint"] == 20
    parent = next(s for s in json.loads((ROOT / "runs/p-study/batch.json").read_text()) if s["arm"] == "P21")
    assert spec["hp"] == parent["hp"] and spec["base_sha"] == parent["base_sha"]
    for name, h in spec["overlay_sha256"].items(): assert sha(ROOT / name) == h
    assert sha(OUT / "preflight.json") == spec["preflight_sha256"]
    module = importlib.util.spec_from_file_location("ka", SKILL / "kaggle_account.py")
    ka = importlib.util.module_from_spec(module); module.loader.exec_module(ka)
    assert ka.active_user() == "dinhan25"
    q = ka.account_quota(spec["account"]); atomic_json(OUT / "launch-quota.json", q)
    assert q["gpu_left"] >= spec["cap_hours"] * 1.15, "Quota below full7h +15% reserve"
    cmd = [sys.executable, str(SKILL / "kaggle_as.py"), spec["account"], "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]
    refs = owned_refs(cmd, spec["account"])
    assert f'{spec["account"]}/{spec["slug"]}' not in refs, "Recovery slug already exists; inspect before retry"
    atomic_json(OUT / "owned-refs.json", refs)
    listed = subprocess.run(["timeout", "90s", *cmd, "kernels", "list", "--mine", "--sort-by", "dateRun",
                             "--page-size", "10", "--csv"], capture_output=True, text=True)
    assert listed.returncode == 0
    recent = [r["ref"] for r in csv.DictReader(io.StringIO(listed.stdout))]
    def status(ref):
        p = subprocess.run(["timeout", "45s", *cmd, "kernels", "status", ref], capture_output=True, text=True)
        return {"ref": ref, "status": status_from_cli(p.returncode, p.stdout)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: inventory = list(pool.map(status, recent))
    atomic_json(OUT / "recent-statuses.json", {"runs": inventory, "scope": "Latest10 owned runs, not all-session liveness proof"})
    assert all(r["status"] in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"} for r in inventory), "Active/unknown recent session; inspect slot use"
    access = subprocess.run(["timeout", "90s", *cmd, "datasets", "files", "-d", "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"], capture_output=True, text=True)
    assert access.returncode == 0, "Dataset access unavailable"
    folder = ROOT / "notebooks" / spec["slug"]
    assert sha(folder / (spec["slug"] + ".ipynb")) == spec["notebook_sha256"]
    assert json.loads((folder / "kernel-metadata.json").read_text())["is_private"] is True
    ref = f'{spec["account"]}/{spec["slug"]}/1'
    assert not (OUT / "push.json").exists()
    with (OUT / "push-intent.json").open("x") as f:
        json.dump({"ref": ref, "notebook_sha256": spec["notebook_sha256"], "quota": q}, f, indent=2)
    p = subprocess.run(["timeout", "--foreground", "--kill-after=10s", "240s", *cmd, "kernels", "push", "-p", str(folder)], capture_output=True, text=True)
    (OUT / "push.log").write_text(p.stdout + p.stderr); (OUT / "push.log").chmod(0o600)
    assert p.returncode == 0 and re.search(r"Kernel version 1 successfully pushed", p.stdout), "Push failed/ambiguous; inspect before retry"
    atomic_json(OUT / "push.json", {"ref": ref, "notebook_sha256": spec["notebook_sha256"]})
    assert ka.active_user() == "dinhan25"
    print("P21_RECOVERY_PUSHED", ref, flush=True)
    cycle = 0
    while True:
        started = time.monotonic(); cycle += 1
        assert (OUT / "batch.json").read_bytes() == raw
        state = query(spec)
        atomic_json(OUT / "status.json", {"cycle": cycle, "interval_seconds": 1200, **state})
        with (OUT / "status-history.jsonl").open("a") as f: f.write(json.dumps(state) + "\n")
        print("P21_RECOVERY_STATUS", state["status"], flush=True)
        if state["terminal"]: break
        if not (OUT / "startup/startup.json").exists() or not json.loads((OUT / "startup/startup.json").read_text())["all_ready"]:
            subprocess.run([sys.executable, str(ROOT / "scripts/check_screening_startup.py"), "--batch", str(OUT / "batch.json"),
                "--out", str(OUT / "startup"), "--entity", "21522798-uit"], check=True, cwd=ROOT)
        time.sleep(max(1, 1200 - (time.monotonic() - started)))
    assert shutil.disk_usage(ROOT).free > 3 * 2**30
    dest = ROOT / "artifacts/p20" / spec["slug"]; dest.mkdir(parents=True, exist_ok=True)
    pattern = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|train_cms_r\d{3}\.npz|diagnostics_rank[01]\.jsonl|external_cms_rank[01]_r\d{3}\.npz|teacher_probe_rank[01]_r\d{3}\.json)$|/models/r\d{3}\.pt$|/recovery/rank[01]\.jsonl$)'
    with (OUT / "collect.log").open("w") as log:
        p = subprocess.run([sys.executable, str(SKILL / "kaggle_as.py"), spec["account"], "--", sys.executable,
            str(ROOT / "scripts/run_bounded_python.py"), str(SKILL / "pull_kernel_output.py"), ref,
            "-p", str(dest), "-j", "8", "--pattern", pattern], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
    assert p.returncode == 0, "Terminal pull incomplete; keep artifacts and resume only collector"
    atomic_json(OUT / "collection.json", {"ref": ref, "status": state["status"], "root": str(dest), "pulled": True})
    assert state["status"] == "COMPLETE", "P21 recovery terminal error collected; inspect numerical metadata before any new retry"
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_p20.py"), "--batch", str(OUT / "batch.json"),
        "--root", str(ROOT / "artifacts/p20"), "--out", str(OUT / "audit")], check=True, cwd=ROOT)
    for rank in (0, 1):
        records = [json.loads(line) for line in (dest / spec["slug"] / "recovery" / f"rank{rank}.jsonl").read_text().splitlines()]
        assert records[0] == {"rank": rank, "event": "compile_guard", "compiled_ranks": 2, "world": 2}
        assert len(records) == 21 and all(r["event"] == "round_finite" for r in records[1:])
        assert [r["round"] for r in records[1:]] == list(range(1, 21))
        assert all(r["clients"] == (2672 if rank == 0 else 2671) and r["groups"] > 0 for r in records[1:])
    atomic_json(OUT / "complete.json", {"ref": ref, "rounds": 20, "status": "PASS_BOUNDED", "logical_arm": "P21",
        "recovery_arm": "P21R1", "batch_sha256": sha(OUT / "batch.json"), "preflight_sha256": spec["preflight_sha256"],
        "reports": {name: sha(OUT / "audit" / name) for name in ["audit.json", "diagnostic-audit.json"]}})
    print("P21_RECOVERY20_PULL_AUDIT_DONE", flush=True)


if __name__ == "__main__": main()
