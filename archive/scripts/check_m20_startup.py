"""One bounded read of exact M20 sessions; no launch, retry or polling loop."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/m-study"


def check(spec):
    base = [sys.executable, ".agents/skills/kaggle-training-notebook/scripts/kaggle_as.py", spec["account"],
            "--", sys.executable, "scripts/kaggle_cli_bounded.py", "kernels"]
    ref = f"{spec['account']}/{spec['slug']}/{spec['version']}"
    status = subprocess.run(["timeout", "--kill-after=5s", "35s", *base, "status", ref],
                            cwd=ROOT, capture_output=True, text=True)
    (OUT / (spec["slug"] + ".status.log")).write_text(status.stdout + status.stderr)
    path = OUT / (spec["slug"] + ".live.log")
    with path.open("w") as f:
        p = subprocess.run(["timeout", "--kill-after=5s", "35s", *base, "logs", ref, "--follow"],
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           env=dict(os.environ, PYTHONUNBUFFERED="1"))
    text = path.read_text()
    return {"arm": spec["arm"], "ref": ref, "status": status.stdout.strip(), "status_rc": status.returncode,
            "logs_rc": p.returncode, "two_T4_confirmed": "['Tesla T4', 'Tesla T4']" in text,
            "manifest": re.findall(r"source manifest ([0-9a-f]{64})", text),
            "wandb_authenticated": "W&B authenticated" in text,
            "compile_gate_ranks": re.findall(r"compile gate rank(\d+): PASS", text),
            "round_lines": re.findall(r"r\d\d lr=.*", text),
            "error": any(t in text for t in ("Traceback (most recent call last)", "CUDA out of memory"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="*")
    args = ap.parse_args()
    batch = json.loads((OUT / "batch.json").read_text())
    selected = [s for s in batch if (not args.arms or s["arm"] in args.arms)
                and (OUT / (s["slug"] + ".push.json")).exists()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(check, selected))
    path = OUT / "startup.json"
    prior = {r["arm"]: r for r in json.loads(path.read_text())["runs"]} if path.exists() else {}
    prior.update({r["arm"]: r for r in rows})
    path.write_text(json.dumps({"observed_at": datetime.datetime.now().astimezone().isoformat(),
                                "runs": list(prior.values())}, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
