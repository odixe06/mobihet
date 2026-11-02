"""Read-only inventory of newly authorized owners; leave active R/V files alone."""
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
OUT = ROOT / "runs/r-repeat"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
sys.path.insert(0, str(ROOT))
from scripts.watch_kaggle_batch import status_from_cli, atomic_json


def main():
    spec = importlib.util.spec_from_file_location("ka", SKILL / "kaggle_account.py")
    ka = importlib.util.module_from_spec(spec); spec.loader.exec_module(ka)
    owners = json.loads((OUT / "authorized-owners.json").read_text())["owners"]
    busy = {r["account"] for branch in ("r-study", "v-study")
            for r in json.loads((ROOT / "runs" / branch / "batch.json").read_text())}
    assert len(owners) >= 4 and not set(owners) & busy and set(owners) <= set(ka.known_users())
    before = ka.active_user()

    def inventory(owner):
        q = ka.account_quota(owner)
        cmd = [sys.executable, str(SKILL / "kaggle_as.py"), owner, "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]
        p = subprocess.run(["timeout", "90s", *cmd, "kernels", "list", "--mine", "--sort-by", "dateRun", "--page-size", "10", "--csv"], capture_output=True, text=True)
        assert p.returncode == 0, (owner, "inventory unavailable")
        states = []
        for row in csv.DictReader(io.StringIO(p.stdout)):
            ref = row.get("ref", "")
            if not ref.startswith(owner + "/"): continue
            p = subprocess.run(["timeout", "45s", *cmd, "kernels", "status", ref], capture_output=True, text=True)
            states.append({"ref": ref, "status": status_from_cli(p.returncode, p.stdout)})
        q.update(recent_sessions=states,
                 active_recent=[r["ref"] for r in states if r["status"] in {"QUEUED", "RUNNING"}],
                 unknown_recent=[r["ref"] for r in states if r["status"] == "QUERY_ERROR"])
        print("REPEAT20_INVENTORY", owner, round(q["gpu_left"], 2), len(q["active_recent"]), len(q["unknown_recent"]), flush=True)
        return q

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: rows = list(pool.map(inventory, owners))
    atomic_json(OUT / "quota.json", {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "accounts": rows,
        "global_cli_before": before, "global_cli_after": ka.active_user(),
        "scope": "Owner02Nov authorized newly saved accounts; scoped server identity/quota, latest10 owned slots; active R/V excluded"})
    print("REPEAT20_INVENTORY_DONE", flush=True)


if __name__ == "__main__": main()
