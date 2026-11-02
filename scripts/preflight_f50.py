"""F50 preflight: the 50-round driver path (per-round full state, cross-session resume) on the real v4 fixture.

For core and bank (X20 hp, fixture overrides only): one uninterrupted run to round 4 against a run stopped at round 2,
resumed to round 3, then resumed again from the round-2 state (a crash after round 3's rows, before its state save) to
round 4. History, CMs, server snapshots and the final full state must be bit-identical; *.jsonl holds each round once.
Local sm86, world 1, 2 clients/173 train rows: not T4/2-rank/full5343 proof.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts import f50_common

OUT = ROOT / "runs/f-study"
FIXTURE = ["group_rows=128", "diagnostic_ids=[613,2029]", "diagnostic_rounds=[1,2,3]", "external_diagnostics=false"]
TIMING = ("t_train", "t_eval", "t_diagnostics", "scoring_seconds_local")


def drive(out, until, hp):
    cmd = [sys.executable, "scripts/run_q20.py", "--method", "mobihet_wstudy", "--out", str(out), "--width", "1",
           "--rounds", "50", "--until", str(until), "--save-every", "1", "--compile", "--clients", "613,2029",
           "--cache", str(ROOT / "runs/n-study/smoke-cache"), "--set", *hp, *FIXTURE]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (out.parent / f"{out.name}-u{until}.log").write_text(p.stdout + p.stderr)
    assert p.returncode == 0, f"{out} until {until}: see log"
    assert "compile gate rank0: PASS" in p.stdout
    return p.stdout


def history(d):
    rows = [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines()]
    return [{k: v for k, v in r.items() if not k.startswith(TIMING)} for r in rows]


def same_tree(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys(); return all(same_tree(a[k], b[k]) for k in a)
    if torch.is_tensor(a): return torch.equal(a, b)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(same_tree(x, y) for x, y in zip(a, b))
    return a == b


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in f50_common.SOURCES}
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_wstudy.py", "tests/test_r20.py", "tests/test_mobihet_next.py",
        "tests/test_units.py", "--junitxml=" + str(OUT / "tests.xml"),
        "--deselect", "tests/test_r20.py::test_frozen_history_independent_source_and_tamper"],   # Q6/Q7 checkpoints DVC-only
        cwd=ROOT, check=True)
    parent = {r["cell"]: r for r in json.loads((ROOT / "runs/x-study/batch.json").read_text()) if "seed=42" in r["hp"]}
    reports = []
    for cell in f50_common.CELLS:
        hp = parent[cell]["hp"]
        assert f50_common.hp_for(cell, 42) == hp, cell          # F50 = X20 recipe; only the round count changes
        base = OUT / f"smoke-{cell}"; shutil.rmtree(base, ignore_errors=True); base.mkdir()
        a, b = base / "straight", base / "resumed"
        drive(a, 4, hp)
        drive(b, 2, hp)
        keep = base / "state-r2"; keep.mkdir()
        shutil.copy2(b / "state.pt", keep / "state.pt")
        log = drive(b, 3, hp)
        assert "resumed rank0 at round 3" in log and not (b / "state_rank1.pt").exists()
        assert torch.load(b / "state.pt", weights_only=True)["round"] == 3
        shutil.copy2(keep / "state.pt", b / "state.pt")       # crash after round 3's rows, before its state save
        log = drive(b, 4, hp)
        assert "resumed rank0 at round 3" in log and "dropped 1 rows past round 2" in log
        for d in (a, b):
            for f in [*d.glob("*.jsonl"), *d.glob("recovery/*.jsonl")]:
                rounds = [json.loads(x)["round"] for x in f.read_text().splitlines() if "round" in json.loads(x)]
                assert rounds == sorted(set(rounds)), f
        assert history(a) == history(b) and [r["round"] for r in history(a)] == [1, 2, 3, 4]
        for r in range(1, 5):
            with np.load(a / f"cms_r{r:03d}.npz") as x, np.load(b / f"cms_r{r:03d}.npz") as y:
                assert x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files), r
            assert same_tree(torch.load(a / f"models/r{r:03d}.pt", weights_only=True),
                             torch.load(b / f"models/r{r:03d}.pt", weights_only=True)), r
        sa, sb = (torch.load(d / "state.pt", weights_only=True) for d in (a, b))
        assert sa["round"] == sb["round"] == 4 and same_tree(sa, sb), "final full state differs"
        assert [p.name for p in a.glob("state*")] == ["state.pt"]   # one copy on disk, no .tmp left
        reports.append({"cell": cell, "rounds": 4, "resume_points": [2, 2], "history_cms_snapshots_state_bitexact": True,
                        "jsonl_rounds_unique": True, "train_loss_r4": history(a)[-1]["train_loss"]})
        print("F50_RESUME_PASS", cell, flush=True)
    assert before == {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in f50_common.SOURCES}
    (OUT / "preflight.json").write_text(json.dumps({"source_sha256": before, "source_unchanged": True, "reports": reports,
        "scope": "Real v4 W1 local sm86 world1 2 clients, 4 rounds, resume at r2 twice incl. post-row crash; not T4/2-rank/full5343"},
        indent=2) + "\n")
    print("F50_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
