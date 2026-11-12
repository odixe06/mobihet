"""Isolated sequential shadow/C/E/H phases, one seed per Kaggle notebook."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b18_partition import build, read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--world", type=int, default=2)
    ap.add_argument("--until", type=int, default=20)
    ap.add_argument("--max-hours", type=float, default=5.5)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    assert a.seed in (42, 43) and a.world in (1, 2)
    out.mkdir(exist_ok=True, parents=True)
    assert not (out / "SEED_DONE.json").exists(), "Do not restart a completed seed"
    started = time.monotonic()
    os.environ.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TORCHINDUCTOR_COMPILE_THREADS="1")
    if not (out / "partition.json").exists():
        build(out)
    _, meta = read(out)
    for phase in ("shadow", "C", "E", "H"):
        phase_out = out / phase
        assert not (phase_out / "rounds.jsonl").exists(), "Inspect partial phase; no blind restart"
        left = a.max_hours - (time.monotonic() - started) / 3600 - .1
        assert left > .3, "Insufficient remaining budget for next phase"
        cmd = [sys.executable]
        if a.world == 2:
            cmd += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2"]
        cmd += ["scripts/b18_phase.py", "--root", str(out), "--phase", phase, "--seed", str(a.seed),
                "--until", str(a.until), "--max-hours", str(left)]
        print("B18_PHASE", phase, "seed", a.seed, "hours_left", round(left, 2), flush=True)
        with (out / f"{phase}.log").open("x") as log:
            subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        rows = [json.loads(s) for s in (phase_out / "rounds.jsonl").read_text().splitlines()]
        assert [r["round"] for r in rows] == list(range(1, a.until + 1)), "Incomplete phase, retain resume state"
        assert all(r["clients_without_applied"] == 0 and r["skipped_frac"] <= .01 for r in rows)
        for rank in range(a.world):
            assert f"compile gate rank{rank}: PASS" in (out / f"{phase}.log").read_text()
        print("B18_PHASE_DONE", phase, flush=True)
    subprocess.run([sys.executable, "scripts/audit_b18.py", "--root", str(out), "--rounds", str(a.until),
                    "--world", str(a.world)], cwd=ROOT, check=True)
    (out / "SEED_DONE.json").write_text(json.dumps({"seed": a.seed, "rounds": a.until,
        "partition_sha256": meta["logical_sha256"], "world": a.world,
        "elapsed_seconds": time.monotonic() - started, "audit_pass": True}, indent=2) + "\n")
    print("B18_SEED_AUDITED", a.seed, flush=True)


if __name__ == "__main__":
    main()
