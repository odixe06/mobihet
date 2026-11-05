"""Deferred F50 arm: core seed 43, same locked recipe and base source as F2; only seed and owner differ.

Owner 05/11 provided a new account. Isolated receipts in runs/f-study-s43; run_f50 watch/collect/audit reused unchanged.

    python scripts/run_f50_s43.py [--collect-only]
"""
import argparse
import fcntl
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import f50_common, run_f50 as F
from scripts.watch_kaggle_batch import atomic_json

OWNER, CELL, SEED, ARM = "outlookcheck", "core", 43, "F3"
OUT, PARENT = ROOT / "runs/f-study-s43", ROOT / "runs/f-study"


def prepare(build):
    path = OUT / "batch.json"
    if path.exists(): return json.loads(path.read_text())
    W = F.W
    smoke = json.loads((PARENT / "preflight.json").read_text())
    assert smoke["source_unchanged"] and all(W.sha(ROOT / n) == h for n, h in smoke["source_sha256"].items())
    f2 = next(r for r in json.loads((PARENT / "batch.json").read_text()) if r["arm"] == "F2")
    hp = f50_common.hp_for(CELL, SEED)
    assert [h for h in hp if not h.startswith("seed=")] == [h for h in f2["hp"] if not h.startswith("seed=")] and f"seed={SEED}" in hp
    assert W.sha(ROOT / "methods/mobihet.py") == f2["source_sha256"]
    q = W.ka().account_quota(OWNER); assert q["gpu_left"] >= F.NEED, q
    atomic_json(OUT / "quota.json", q)
    slug = f"mh-{ARM.lower()}-{CELL}-s{SEED}-r50"
    nb, meta = F.notebook(OWNER, slug, slug, hp, f2["base_sha"], build, F.CAP)
    batch = [dict(f2, arm=ARM, account=OWNER, slug=slug, hp=hp, notebook_sha256=nb, metadata_sha256=meta)]
    atomic_json(path, batch)
    atomic_json(ROOT / "docs/research/f50-s43-launch.json", {"batch_sha256": W.sha(path), "batch": batch, "builder": build,
        "authorization": "Owner 05/11: new account for deferred core s43", "scope": "F2 recipe and base SHA, seed 43 only"})
    return batch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--collect-only", action="store_true"); a = ap.parse_args()
    F.OUT, F.TAG = OUT, "F50S43"; F.configure(); OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "worker.lock").open("a"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    build = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    if not a.collect_only: F.W.launch(prepare(build))
    assert (OUT / "launch-complete.json").exists(), "Partial push: inspect receipts, never launch blindly"
    F.watch(json.loads((OUT / "batch.json").read_text()), build)
    print("F50S43_PIPELINE_DONE", flush=True)


if __name__ == "__main__": main()
