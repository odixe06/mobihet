"""All-owner gates before mutation, one-shot V20 pushes, immutable notebook/source receipts."""
import concurrent.futures
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.watch_kaggle_batch import status_from_cli, atomic_json

OUT = ROOT / "runs/v-study"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def owned_refs(cmd, owner):
    refs = []
    for page in range(1, 101):
        p = subprocess.run(["timeout", "90s", *cmd, "kernels", "list", "--mine", "--page-size", "100", "--page", str(page), "--csv"], capture_output=True, text=True)
        assert p.returncode == 0, (owner, "owned listing unavailable")
        rows = list(csv.DictReader(io.StringIO(p.stdout)))
        if not rows: return refs
        refs.extend(row["ref"] for row in rows if row.get("ref", "").startswith(owner + "/"))
    raise RuntimeError("Owned listing did not finish pagination")


def main():
    raw = (OUT / "batch.json").read_bytes(); batch = json.loads(raw)
    assert len(batch) == 8 and len({r["slug"] for r in batch}) == 8
    smoke = json.loads((OUT / "preflight.json").read_text())
    import xml.etree.ElementTree as ET
    suites = ET.parse(OUT / "audit-tests.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 5
    assert all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == int(s.attrib.get("skipped", 0)) == 0 for s in suites)
    for name, h in smoke["source_sha256"].items(): assert sha(ROOT / name) == h, name
    module = importlib.util.spec_from_file_location("ka", SKILL / "kaggle_account.py")
    ka = importlib.util.module_from_spec(module); module.loader.exec_module(ka)
    global_before = ka.active_user()
    busy = {r["account"] for r in json.loads((ROOT / "runs/r-study/batch.json").read_text())}
    groups = {}
    for row in batch:
        assert row["account"] not in busy, "V must not consume R owners or reservations"
        groups.setdefault(row["account"], []).append(row)
        folder = ROOT / "notebooks" / row["slug"]
        assert sha(folder / (row["slug"] + ".ipynb")) == row["notebook_sha256"]
        assert sha(folder / "kernel-metadata.json") == row["metadata_sha256"]
        meta = json.loads((folder / "kernel-metadata.json").read_text())
        assert meta["is_private"] is True and meta["id"] == f'{row["account"]}/{row["slug"]}'
        if row.get("replay_input"):
            assert row["replay_input"]["private"] and row["replay_input"]["download_sha_verified"]
            assert row["replay_input"]["ref"] in meta["dataset_sources"]
        assert not (OUT / (row["slug"] + ".push-intent.json")).exists(), "Existing push intent; inspect before retry"
    def gate(item):
        owner, runs = item
        q = ka.account_quota(owner)
        assert q["gpu_left"] >= sum(r["cap_hours"] for r in runs) * 1.15, (owner, "full owner allocation quota changed")
        cmd = [sys.executable, str(SKILL / "kaggle_as.py"), owner, "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]
        refs = owned_refs(cmd, owner)
        assert all(f'{owner}/{r["slug"]}' not in refs for r in runs), (owner, "slug already exists")
        p = subprocess.run(["timeout", "90s", *cmd, "kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", "10", "--csv"], capture_output=True, text=True)
        assert p.returncode == 0
        status = []
        for row in csv.DictReader(io.StringIO(p.stdout)):
            ref = row.get("ref", "")
            if not ref.startswith(owner + "/"): continue
            p = subprocess.run(["timeout", "45s", *cmd, "kernels", "status", ref], capture_output=True, text=True)
            status.append({"ref": ref, "status": status_from_cli(p.returncode, p.stdout)})
        assert all(r["status"] in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"} for r in status), (owner, "active or unknown recent session")
        access = subprocess.run(["timeout", "90s", *cmd, "datasets", "files", "-d", "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"], capture_output=True, text=True)
        assert access.returncode == 0, (owner, "dataset access unavailable")
        return {"quota": q, "planned_runs": len(runs), "planned_cap_hours": sum(r["cap_hours"] for r in runs),
                "owned_refs": refs, "recent_status": status}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: gates = list(pool.map(gate, groups.items()))
    atomic_json(OUT / "launch-gates.json", {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "accounts": gates,
        "scope": "Authenticated quota, all-page owned slug inventory, latest10 operational slots and dataset access; not all-session proof"})
    def push(item):
        owner, runs = item
        cmd = [sys.executable, str(SKILL / "kaggle_as.py"), owner, "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]
        for row in runs:
            assert (OUT / "batch.json").read_bytes() == raw
            slug = row["slug"]; folder = ROOT / "notebooks" / slug
            assert sha(folder / (slug + ".ipynb")) == row["notebook_sha256"]
            ref = f"{owner}/{slug}/1"
            with (OUT / (slug + ".push-intent.json")).open("x") as f:
                json.dump({"ref": ref, "notebook_sha256": row["notebook_sha256"], "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, f)
            p = subprocess.run(["timeout", "--foreground", "--kill-after=10s", "240s", *cmd, "kernels", "push", "-p", str(folder)], capture_output=True, text=True)
            log = OUT / (slug + ".push.log"); log.write_text(p.stdout + p.stderr); log.chmod(0o600)
            assert p.returncode == 0 and re.search(r"Kernel version 1 successfully pushed", p.stdout), f"Push failed/ambiguous {slug}; inspect before retry"
            atomic_json(OUT / (slug + ".push.json"), {"ref": ref, "notebook_sha256": row["notebook_sha256"], "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            print("V20_PUSHED", row["arm"], ref, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(push, groups.items()))
    atomic_json(OUT / "launch-complete.json", {"batch_sha256": sha(OUT / "batch.json"), "runs": 8,
        "global_cli_before": global_before, "global_cli_after": ka.active_user(),
        "auth_scope": "Each owner quota is server-introspected; pushes use process-scoped kaggle_as; global login not modified"})
    print("V20_ALL_8_PUSHED", flush=True)


if __name__ == "__main__": main()
