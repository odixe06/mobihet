"""Read-only R20 quota and recent slots; no prior batch excluded or changed."""
import concurrent.futures
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/r-study"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
sys.path.insert(0, str(ROOT))
from scripts.watch_kaggle_batch import status_from_cli


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("ka", SKILL / "kaggle_account.py")
    ka = importlib.util.module_from_spec(spec); spec.loader.exec_module(ka)
    assert ka.active_user() == "dinhan25"
    def inventory(owner):
        q = ka.account_quota(owner)
        cmd = [sys.executable, str(SKILL / "kaggle_as.py"), owner, "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]
        p = subprocess.run(["timeout", "90s", *cmd, "kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", "10", "--csv"], capture_output=True, text=True)
        assert p.returncode == 0, (owner, "inventory unavailable")
        rows = list(csv.DictReader(io.StringIO(p.stdout)))
        foreign = [row.get("ref") for row in rows if not row.get("ref", "").startswith(owner + "/")]
        rows = [row for row in rows if row.get("ref", "").startswith(owner + "/")]
        statuses = []
        for row in rows:
            ref = row["ref"]; assert ref.startswith(owner + "/")
            r = subprocess.run(["timeout", "45s", *cmd, "kernels", "status", ref], capture_output=True, text=True)
            statuses.append({"ref": ref, "status": status_from_cli(r.returncode, r.stdout)})
        q.update(non_owner_refs_filtered=foreign, recent_sessions=statuses, active_recent=[r["ref"] for r in statuses if r["status"] in {"QUEUED", "RUNNING"}],
                 unknown_recent=[r["ref"] for r in statuses if r["status"] == "QUERY_ERROR"])
        print("R20_INVENTORY", owner, round(q["gpu_left"], 2), len(q["active_recent"]), len(q["unknown_recent"]), flush=True)
        return q
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(inventory, ka.known_users()))
    (OUT / "quota.json").write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "accounts": rows,
        "scope": "Latest10 owned runs, account-scoped identity/quota; platform slot gate repeated before push; prior batches terminal"}, indent=2) + "\n")
    assert ka.active_user() == "dinhan25"
    print("R20_INVENTORY_DONE", flush=True)


if __name__ == "__main__": main()
