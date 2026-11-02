"""Read-only account-scoped quota and recent session inventory before the authorized P20 batch."""
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
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
sys.path.insert(0, str(ROOT))
from scripts.watch_kaggle_batch import status_from_cli


def main():
    module = importlib.util.spec_from_file_location("ka", SKILL / "kaggle_account.py")
    ka = importlib.util.module_from_spec(module); module.loader.exec_module(ka)
    def account(user):
        q = ka.account_quota(user)
        cmd = [sys.executable, str(SKILL / "kaggle_as.py"), user, "--", sys.executable,
               str(ROOT / "scripts/kaggle_cli_bounded.py"), "kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", "10", "--csv"]
        p = subprocess.run(["timeout", "90s", *cmd], capture_output=True, text=True)
        assert p.returncode == 0, (user, p.returncode)
        rows = list(csv.DictReader(io.StringIO(p.stdout)))
        refs = [row["ref"] for row in rows if row.get("ref", "").startswith(user + "/")]
        statuses = []
        for ref in refs:
            # Inventory asks latest operational status, not historicalversion1 of a reused slug.
            status_cmd = [sys.executable, str(SKILL / "kaggle_as.py"), user, "--", sys.executable,
                          str(ROOT / "scripts/kaggle_cli_bounded.py"), "kernels", "status", ref]
            result = subprocess.run(["timeout", "45s", *status_cmd], capture_output=True, text=True)
            statuses.append({"ref": ref, "version_scope": "latest for inventory only", "status": status_from_cli(result.returncode, result.stdout)})
        q["recent_sessions"] = statuses
        q["active_recent"] = [s["ref"] for s in statuses if s["status"] in {"QUEUED", "RUNNING"}]
        q["unknown_recent"] = [s["ref"] for s in statuses if s["status"] == "QUERY_ERROR"]
        print("INVENTORY", user, round(q["gpu_left"], 2), "active", len(q["active_recent"]), "unknown", len(q["unknown_recent"]), flush=True)
        return q
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        accounts = list(pool.map(account, ka.known_users()))
    assert ka.active_user() == "dinhan25"
    out = ROOT / "runs/p-study/quota.json"
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "accounts": accounts,
        "scope": "Last10 most recently run notebooks per account; quota is not session liveness proof; platform slot limit checked again on push."}, indent=2) + "\n")
    print("P20_INVENTORY_DONE", flush=True)


if __name__ == "__main__": main()
