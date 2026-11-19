"""FB-L sim batch, DAGSNet (18/11/2026): FB N4/N2 + real rows chosen by stored CE; fold train -> fold val, 200 vehicles, cosine20,
DAGSNet W1, seed 42. Reuses scripts/run_sim_budget.py (notebook generation, slots, launch, collect); only arms and folders differ.

    python scripts/run_sim_fbl.py cost | quota | launch | collect

Rule locked before push: docs/research/screening-fbl.md (gate across DAGSNet and FeatureBERT: scripts/gate_fbl.py).
19/11/2026 (owner): every arm runs the compact engine (compact=true: executed client compute ~1,00 x FedAvg instead of ~2,07 x,
same function, docs/research/compute-harness.md); paired bases N4c / N2c rerun here on that engine. S0 = the audited Lite2 sim arm.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sim_budget as B  # noqa: E402

N4 = ([[64, 64], [16, 16], [16, 16]], [64, 37, 37])
N2 = ([[64, 32], [16, 16], [16, 16]], [64, 40, 40])
# arm: (tag, base, extra hp) -- tag also names the FeatureBERT arm (scripts/run_sim_fbl.py in fl_veremi_bert)
ARMS = {"N4c": ("n4c-37-16", N4, []),
        "N2c": ("n2c-40-16", N2, []),
        "L4": ("l4-top", N4, ["real_select=top"]),
        "P4": ("p4-prop", N4, ["real_select=prop"]),
        "Q4": ("q4-top-q18", N4, ["real_select=top", "real_select_q=18"]),
        "L2": ("l2-top", N2, ["real_select=top"]),
        "P2": ("p2-prop", N2, ["real_select=prop"]),
        "Q2": ("q2-top-q20", N2, ["real_select=top", "real_select_q=20"])}
CONTROLS = ("N4c", "N2c")

B.ARMS = {a: (tag, synth, real, extra + ["compact=true"], "control" if a in CONTROLS else "candidate")
          for a, (tag, (synth, real), extra) in ARMS.items()}
B.OUT, B.ART = ROOT / "runs/sim-kaggle-fbl", ROOT / "artifacts/sim-kaggle-fbl"
B.W.OUT, B.W.ART, B.W.TAG = B.OUT, B.ART, "FBL"


def quota(tries=4):
    """Flaky network (docs/operations/lessons.md): repeat the roster read and keep, per account, its latest clean read
    (quota known, listing ok, no QUERY_ERROR status) taken within this call; an account never read clean gets no slot."""
    import json
    import time
    receipt, clean = B.OUT / "quota.json", {}
    for _ in range(tries):
        receipt.unlink(missing_ok=True)
        sys.argv[1:] = ["quota"]
        B.main()
        for a in json.loads(receipt.read_text())["accounts"]:
            if not a.get("inventory_error") and a.get("active_recent") is not None and not a.get("unknown_recent"):
                clean[a["user"]] = a
        print("clean reads", sorted(clean), flush=True)
    out = json.loads(receipt.read_text())
    out["accounts"] = [clean.get(a["user"], dict(a, inventory_error=a.get("inventory_error") or "no clean read"))
                       for a in out["accounts"]]
    out["merged_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    B.atomic_json(receipt, out)


if __name__ == "__main__":
    assert sys.argv[1] in ("cost", "quota", "launch", "collect"), "gate: scripts/gate_fbl.py (both models)"
    if sys.argv[1] == "quota":
        quota()
    else:
        B.main()
