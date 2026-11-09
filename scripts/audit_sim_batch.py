"""Artifact audit of a pulled simulator batch (fold val only), before its locked analysis.

    python scripts/audit_sim_batch.py b9|b10 [round=20]   -> runs/sim-kaggle-<batch>/audit.json, exit 1 if any arm fails

Per arm: pulled.ok; executed provenance base SHA + overlay SHAs == batch.json and executed mobihet_sim.py bytes hash to the
overlay SHA; config hp == locked hp; rounds 1..R each once, finite loss/F1, skipped_frac <= 1 %; cms_r<R> ids == sim panel
(sim_fold_val.pick, seed 7) and per-vehicle/class CM support == fold-val counts; ten metrics finite; KM byte line logged.
Scope: no prediction replay, no checkpoint reconstruction (sim runs save no weights).
"""
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core import metrics  # noqa: E402
from scripts.sim_fold_val import pick  # noqa: E402

B = sys.argv[1]
R = int(sys.argv[2]) if len(sys.argv) > 2 else 20
RUNS, ART = ROOT / f"runs/sim-kaggle-{B}", ROOT / f"artifacts/sim-kaggle-{B}"
cnt = np.load(ROOT / "cache/fold_counts.npy")
panel = np.sort(pick(200, cnt=cnt))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def norm(v):
    return str(v).lower()


def audit(spec):
    slug, out = spec["slug"], {}
    d = ART / slug
    out["pulled"] = (d / "pulled.ok").exists()
    if not out["pulled"]:
        return out
    man = json.loads((d / "provenance/source_manifest.json").read_text())
    out["base_sha"] = man["base_sha"] == spec["base_sha"]
    out["overlay_sha"] = man["overlay"] == spec["overlay_sha256"]
    src = d / "provenance/source/methods/mobihet_sim.py"
    out["executed_overlay_bytes"] = src.exists() and sha(src) == spec["overlay_sha256"]["methods/mobihet_sim.py"]
    run = next(iter(glob.glob(str(d / "**" / "config.json"), recursive=True)), None)
    if not run:
        out["config"] = False
        return out
    run = Path(run).parent
    hp = json.loads((run / "config.json").read_text())["hp"]
    bad = [h for h in spec["hp"] if norm(hp.get(h.split("=", 1)[0])) != norm(h.split("=", 1)[1])]
    out["config"] = not bad
    if bad:
        out["config_mismatch"] = bad
    rows = [json.loads(l) for l in (run / "rounds.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "round" in r and "train_loss" in r]
    rounds = [r["round"] for r in rows]
    out["rounds_exact"] = rounds == list(range(1, len(rounds) + 1)) and len(rounds) >= R
    out["last_round"] = max(rounds) if rounds else 0
    out["finite"] = all(math.isfinite(r["train_loss"]) and math.isfinite(r.get("private/f1_macro", float("nan"))) for r in rows[:R])
    out["max_skipped_frac"] = max((r.get("skipped_frac", 0.0) for r in rows[:R]), default=None)
    out["skipped_ok"] = out["max_skipped_frac"] is not None and out["max_skipped_frac"] <= 0.01
    f = run / f"cms_r{R:03d}.npz"
    out["cm_present"] = f.exists()
    if f.exists():
        z = np.load(f)
        ids, cm = z["ids"], z["private"].astype(np.int64)
        out["cm_ids_panel"] = bool(np.array_equal(np.sort(ids), panel))
        out["cm_support"] = bool(np.array_equal(cm.sum(2), cnt[ids, 1]))
        ten = metrics.summary(cm)
        out["ten_finite"] = all(math.isfinite(ten[k]) for k in metrics.TEN)
        extra = [k for k in z.files if k not in ("ids", "private")]      # b11 views (raw / ph*): same rows, same support
        if extra:
            out["views_support"] = all(np.array_equal(z[k].astype(np.int64).sum(2), cnt[ids, 1]) for k in extra)
    log = next(iter(glob.glob(str(d / "*.log"))), None)
    out["km_bytes_logged"] = bool(log) and "KM-copula K=" in Path(log).read_text()
    return out


batch = json.loads((RUNS / "batch.json").read_text())
res, ok_all = {}, True
for spec in batch:
    a = audit(spec)
    checks = {k: v for k, v in a.items() if isinstance(v, bool)}
    a["pass"] = bool(checks) and all(checks.values())
    ok_all &= a["pass"]
    res[spec["arm"]] = a
    print(spec["arm"], spec["slug"], "PASS" if a["pass"] else "FAIL " + str([k for k, v in checks.items() if not v]),
          "last_round", a.get("last_round"), flush=True)
json.dump({"batch": B, "round": R, "arms": res, "pass": ok_all}, open(RUNS / "audit.json", "w"), indent=1)
print("AUDIT", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
