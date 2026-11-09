"""Y50 batch: three Core candidates of docs/research/pipeline-review.md, locked before launch (screening-y50.md).

Each cell changes one thing of the audited F2/F3 Core recipe; F2/F3 are the paired controls, so no new control is run.
"""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from core.metrics import ten
from scripts import f50_common
from scripts.analyse_m20 import conditional_accuracy
from scripts.research_metrics import client_f1

CORE = {"synth_step_floor": 16, "w_boot": "compact"}
CELLS = {   # set: hp replaced in the Core list; w: W knobs the F50 audit compares; method: overlay when not mobihet_wstudy
    "oracle": {"w": dict(CORE), "set": {"synth": "oracle"}, "method": "mobihet_wstudy"},
    "prior": {"w": dict(CORE), "set": {}, "method": "mobihet_prior"},
    "noq": {"w": dict(CORE), "set": {"no_kd": "true"}, "method": "mobihet_wstudy"},
    "nofloor": {"w": {"w_boot": "compact"}, "set": {}, "method": "mobihet_wstudy"},
}
SCREEN = 20
# Arm, cell, seed, owner, first-session end round, first-session cap (h). Owner 06/11: start now on the second outlookcheck slot and
# on short quotas; a run that hits its cap continues on the same owner after the 07/11 reset. outlookcheck: Y0 11 h + Y6 8 h < 19.79 h.
WAVES = {1: [("Y0", "oracle", 42, "outlookcheck", 50, 11.0), ("Y1", "prior", 42, "heykewtiie", SCREEN, 5.3), ("Y2", "prior", 43, "keocaosu", SCREEN, 5.3)],
         2: [("Y3", "noq", 42, "novahihi", 50, 5.6), ("Y4", "noq", 43, "nguyetquedo", 50, 5.4),
             ("Y5", "nofloor", 42, "accanniez", 50, 4.6), ("Y6", "nofloor", 43, "outlookcheck", 50, 8.0)]}
CONTROL = {42: ROOT / "artifacts/f50/mh-f2-core-s42-r50/mh-f2-core-s42-r50", 43: ROOT / "artifacts/f50/mh-f3-core-s43-r50/mh-f3-core-s43-r50"}
NOISE = {"f1_macro": .0005, "accuracy": .0005, "other15": .0005, "benign": .0005, "p10": .006, "low256": .003, "timeDelay": .0035}
PRIOR_REJECT = {"timeDelay": -.02, "absent": -.01}
ORACLE_WORTH = {"f1_macro": .01, "timeDelay": .05}
GUARD = ("p10", "low256")


def hp_for(cell, seed):
    spec = CELLS[cell]
    hp = [h for h in f50_common.hp_for("core", seed) if h.split("=")[0] not in spec["set"]]
    hp = [h for h in hp if h.split("=")[0] not in CORE or h.split("=")[0] in spec["w"]]
    return hp + [f"{k}={v}" for k, v in spec["set"].items()]


def read(run_dir, r):
    """Own-test numbers of round r from the run's CM; the same definitions as scripts/analyse_f50.py."""
    counts = np.load(ROOT / "evidence/research-data/train_counts.npy"); small = counts.sum(1) < 256
    cm = np.load(Path(run_dir) / f"cms_r{r:03d}.npz")["private"]
    f, p, c = client_f1(cm), ten(cm.sum(0)), conditional_accuracy(cm[small], counts[small])
    per = np.asarray(p["f1_per_class"], dtype=float)
    return {"accuracy": float(p["accuracy"]), "f1_macro": float(p["f1_macro"]), "client_mean": float(f.mean()),
            "p10": float(np.percentile(f, 10)), "low256": float(f[small].mean()), "timeDelay": float(per[13]), "benign": float(per[0]),
            "other15": float(np.delete(per, 13).mean()), "absent": float(c["absent"]["accuracy"]), "rare1_16": float(c["rare1_16"]["accuracy"])}


def delta(run_dir, seed, r):
    a, b = read(run_dir, r), read(CONTROL[seed], r)
    return {k: a[k] - b[k] for k in a}


def prior_rule(deltas):
    """deltas: one dict per seed. PASS needs both gains on both seeds, no rejection trigger, no guard loss beyond noise."""
    gain = all(d[k] >= NOISE[k] for d in deltas for k in ("other15", "benign"))
    rejected = [k for d in deltas for k, v in PRIOR_REJECT.items() if d[k] <= v]
    guard = [k for d in deltas for k in GUARD if d[k] <= -NOISE[k]]
    return {"pass": gain and not rejected and not guard, "gain": gain, "rejected": rejected, "guard_loss": guard}


def ablation_rule(deltas):
    """Core with one part removed vs Core. removable: no primary metric drops beyond noise on either seed;
    needed: some metric drops beyond noise on both seeds; otherwise unresolved."""
    keys = ("accuracy", "f1_macro", "p10", "low256", "timeDelay")
    both = [k for k in keys if all(d[k] <= -NOISE[k] for d in deltas)]
    any_ = [k for k in keys if any(d[k] <= -NOISE[k] for d in deltas)]
    return {"verdict": "needed" if both else "unresolved" if any_ else "removable", "drops_both_seeds": both, "drops_one_seed": any_}


def oracle_rule(d):
    """oracle - Core, seed 42 only: a position, not a significance claim."""
    if any(d[k] >= v for k, v in ORACLE_WORTH.items()): return "generator_worth_it"
    if d["f1_macro"] < NOISE["f1_macro"] and d["timeDelay"] < NOISE["timeDelay"]: return "stop_generator"
    return "small_ceiling"
