"""Push only frozen authorized M20 notebooks; no retries after uncertain acceptance."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/m-study"


def push(spec, quota):
    q = quota[spec["account"]]
    assert q["gpu_left"] >= 6.9 and q["gpu_reserved"] == 0
    notebook = ROOT / "notebooks" / spec["slug"] / (spec["slug"] + ".ipynb")
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == spec["notebook_sha256"]
    for name, digest in spec["overlay_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    marker = OUT / (spec["slug"] + ".push.json")
    assert not marker.exists(), "Inspect original version/status before retrying any push"
    result = {"arm": spec["arm"], "account": spec["account"], "slug": spec["slug"],
              "at": datetime.datetime.now().astimezone().isoformat()}
    command = [sys.executable, ".agents/skills/kaggle-training-notebook/scripts/kaggle_as.py", spec["account"],
               "--", sys.executable, "scripts/kaggle_cli_bounded.py", "kernels", "push", "-p", str(notebook.parent)]
    log = marker.with_suffix(".log")
    with log.open("x") as f:
        os.chmod(log, 0o600)
        try:
            p = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=150)
            result["returncode"] = p.returncode
        except subprocess.TimeoutExpired:
            result["returncode"] = None
    text = log.read_text()
    match = re.search(r"Kernel version (\d+) successfully pushed", text)
    version = int(match.group(1)) if match else None
    result.update(version=version, ambiguous=result["returncode"] != 0 or version != spec["version"])
    marker.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return not result["ambiguous"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="+")
    args = ap.parse_args()
    assert "47 passed" in (OUT / "tests.log").read_text()
    assert len(json.loads((OUT / "equivalence.json").read_text())) == 3
    smoke = json.loads((OUT / "smoke-audit.json").read_text())
    assert smoke["full_private_state_strict_rebuild"] == "PASS" and smoke["skipped_frac"] == [0, 0]
    q = json.loads((OUT / "quota.json").read_text())
    assert q["active"] == "dinhan25"
    assert (datetime.datetime.now().astimezone() - datetime.datetime.fromisoformat(q["at"])).total_seconds() < 900
    batch = json.loads((OUT / "batch.json").read_text())
    selected = [r for r in batch if r["arm"] in args.arms]
    assert len(selected) == len(set(args.arms))
    quota = {r["user"]: r for r in q["accounts"]}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        ok = list(pool.map(lambda s: push(s, quota), selected))
    assert all(ok), "Uncertain push: inspect exact version before any new action; no automatic retry"


if __name__ == "__main__":
    main()
