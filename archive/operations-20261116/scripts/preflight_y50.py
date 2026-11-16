"""Y50 preflight: each cell on the real v4 fixture, straight to r4 vs stopped at r2, resumed to r3, resumed again from the r2
state (a crash after r3's rows) to r4. History, CMs, snapshots and final state must be bit-identical.
Local sm86, world 1, 2 clients: not T4/2-rank/full5343 proof.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from scripts import y50_common as Y
from scripts.preflight_f50 import FIXTURE, history, same_tree

OUT = ROOT / "runs/y-study"
SOURCES = ["methods/mobihet.py", "methods/mobihet_prior.py", "methods/mobihet_wstudy.py", "core/local.py", "scripts/y50_common.py"]


def drive(out, until, cell, hp):
    cmd = [sys.executable, "scripts/run_q20.py", "--method", Y.CELLS[cell]["method"], "--out", str(out), "--width", "1", "--rounds", "50",
           "--until", str(until), "--save-every", "1", "--compile", "--clients", "613,2029",
           "--cache", str(ROOT / "runs/n-study/smoke-cache"), "--set", *hp, *FIXTURE]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    (out.parent / f"{out.name}-u{until}.log").write_text(p.stdout + p.stderr)
    assert p.returncode == 0, f"{out} until {until}: see log"
    assert "compile gate rank0: PASS" in p.stdout
    return p.stdout


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    before = {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in SOURCES}
    subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_y50.py", "tests/test_wstudy.py", "tests/test_lossless_cdf.py",
                    "--junitxml=" + str(OUT / "tests.xml")], cwd=ROOT, check=True)
    reports = []
    for cell in Y.CELLS:
        hp = Y.hp_for(cell, 42)
        base = OUT / f"smoke-{cell}"; shutil.rmtree(base, ignore_errors=True); base.mkdir()
        a, b = base / "straight", base / "resumed"
        drive(a, 4, cell, hp)
        drive(b, 2, cell, hp)
        keep = base / "state-r2"; keep.mkdir()
        shutil.copy2(b / "state.pt", keep / "state.pt")
        assert "resumed rank0 at round 3" in drive(b, 3, cell, hp)
        shutil.copy2(keep / "state.pt", b / "state.pt")
        log = drive(b, 4, cell, hp)
        assert "resumed rank0 at round 3" in log and "dropped 1 rows past round 2" in log
        assert history(a) == history(b) and [r["round"] for r in history(a)] == [1, 2, 3, 4]
        for r in range(1, 5):
            with np.load(a / f"cms_r{r:03d}.npz") as x, np.load(b / f"cms_r{r:03d}.npz") as y:
                assert x.files == y.files and all(np.array_equal(x[k], y[k]) for k in x.files), r
            assert same_tree(torch.load(a / f"models/r{r:03d}.pt", weights_only=True),
                             torch.load(b / f"models/r{r:03d}.pt", weights_only=True)), r
        sa, sb = (torch.load(d / "state.pt", weights_only=True) for d in (a, b))
        assert sa["round"] == sb["round"] == 4 and same_tree(sa, sb), "final full state differs"
        rows = history(a)
        assert all(r["bytes_up"] == r["bytes_down"] == (0 if cell == "noq" else 544) for r in rows), cell
        assert (sa["method_state"]["Q"] is None) == (cell == "noq"), cell
        reports.append({"cell": cell, "rounds": 4, "history_cms_snapshots_state_bitexact": True, "train_loss_r4": rows[-1]["train_loss"],
                        "bytes_round": rows[-1]["bytes_up"], "bootstrap": json.loads((a / "config.json").read_text())["payload_bytes"]["bootstrap"]})
        print("Y50_RESUME_PASS", cell, reports[-1]["train_loss_r4"], reports[-1]["bootstrap"], flush=True)
    assert before == {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in SOURCES}
    (OUT / "preflight.json").write_text(json.dumps({"source_sha256": before, "source_unchanged": True, "reports": reports,
        "scope": "Real v4 W1 local sm86 world1 2 clients, 4 rounds, resume at r2 twice; not T4/2-rank/full5343"}, indent=2) + "\n")
    print("Y50_PREFLIGHT_PASS", flush=True)


if __name__ == "__main__": main()
