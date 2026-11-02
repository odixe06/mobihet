"""One-shot owner-scoped P20 launch. Exact hashes/intent receipts; no automatic ambiguous retry."""
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p-study"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"


def main():
    raw = (OUT / "batch.json").read_bytes(); batch = json.loads(raw)
    assert len(batch) == 32 and len({r["slug"] for r in batch}) == 32
    smoke = json.loads((OUT / "smoke-audit.json").read_text())
    assert smoke["source_unchanged_during_preflight"] and smoke["proxy_carrier_3round_bitexact"]
    for name, digest in batch[0]["overlay_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    module = importlib.util.spec_from_file_location("ka", SKILL / "kaggle_account.py")
    ka = importlib.util.module_from_spec(module); module.loader.exec_module(ka)
    assert ka.active_user() == "dinhan25"
    groups = {}
    for row in batch: groups.setdefault(row["account"], []).append(row)
    assert len(groups) == 16 and all(len(rows) == 2 for rows in groups.values())
    # Query every account before any external mutation, budget both planned sessions plus15%.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        quota = list(pool.map(ka.account_quota, groups))
    assert all(q["gpu_left"] >= 16.1 and q["gpu_reserved"] == 0 for q in quota), "Live quota changed; inspect before push"
    (OUT / "launch-quota.json").write_text(json.dumps(quota, indent=2) + "\n")

    def push_account(item):
        account, rows = item
        for row in rows:
            slug = row["slug"]; folder = ROOT / "notebooks" / slug
            nb = folder / (slug + ".ipynb")
            assert hashlib.sha256(nb.read_bytes()).hexdigest() == row["notebook_sha256"]
            assert json.loads((folder / "kernel-metadata.json").read_text())["is_private"] is True
            receipt, intent = OUT / (slug + ".push.json"), OUT / (slug + ".push-intent.json")
            if receipt.exists():
                assert json.loads(receipt.read_text())["notebook_sha256"] == row["notebook_sha256"]
                continue
            assert not intent.exists(), f"Ambiguous earlier push: {slug}; inspect exact remote version"
            q = ka.account_quota(account)
            assert q["gpu_left"] >= 8.05, (account, "quota insufficient for7h+15%")
            cmd = [sys.executable, str(SKILL / "kaggle_as.py"), account, "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]
            access = subprocess.run(["timeout", "90s", *cmd, "datasets", "files", "-d", "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"], capture_output=True, text=True)
            assert access.returncode == 0, (account, "dataset access", access.returncode)
            intent.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ref": f"{account}/{slug}/1",
                "notebook_sha256": row["notebook_sha256"], "quota": q}, indent=2) + "\n")
            result = subprocess.run(["timeout", "--foreground", "--kill-after=10s", "240s", *cmd, "kernels", "push", "-p", str(folder)], capture_output=True, text=True)
            log = OUT / (slug + ".push.log"); log.write_text(result.stdout + result.stderr); log.chmod(0o600)
            assert result.returncode == 0 and re.search(r"Kernel version 1 successfully pushed", result.stdout), f"Push failed/ambiguous {slug}; inspect private log, never blind retry"
            receipt.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ref": f"{account}/{slug}/1", "returncode": 0,
                "notebook_sha256": row["notebook_sha256"]}, indent=2) + "\n")
            assert (OUT / "batch.json").read_bytes() == raw
            print("PUSHED", row["arm"], f"{account}/{slug}/1", flush=True)
    # Independent accounts in parallel; serial pushes per account, at most2 slots/account.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(push_account, groups.items()))
    assert ka.active_user() == "dinhan25"
    (OUT / "launch-complete.json").write_text(json.dumps({"batch_sha256": hashlib.sha256(raw).hexdigest(), "runs": 32,
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "global_login": "dinhan25"}, indent=2) + "\n")
    print("P20_ALL_32_PUSHED", flush=True)


if __name__ == "__main__": main()
