"""Audit + locked rule of the full-scale lite6 pilot (docs/research/screening-km-lite.md §"Pilot full-scale lite6").

    python scripts/analyse_km_lite_pilot20.py [round=20]  -> runs/km-lite-pilot20/{audit,verdict-r<round>}.json; exit 1 if audit fails

Audit (artifact only): pulled.ok, rounds 1..R exactly once, finite loss, skipped <= 1 %, cms_r001..R, per-vehicle/class CM support
== control KM50 Q42, config hp == KM50 Q42 config hp + cop_kmeans_lite=6, executed overlay sha == batch, ten metrics finite, bytes logged.
Rule: lite6 vs KM50 Q42 at R, non-inferiority macro >= -.0015 + pilot20 guards; absent mean r16..R reported, not gated.
"""
import glob
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.metrics import TEN, summary  # noqa: E402
from scripts import y50_common as Y  # noqa: E402

R = int(sys.argv[1]) if len(sys.argv) > 1 else 20
OUT, ART = ROOT / "runs/km-lite-pilot20", ROOT / "artifacts/km-lite-pilot20"
CTRL = ROOT / "artifacts/km50/mh-k-km16q8-s42-r50/mh-k-km16q8-s42-r50"
CORE = Y.CONTROL[42]
spec, = json.loads((OUT / "batch.json").read_text())
top = ART / spec["slug"]
run = top / spec["slug"]

a = {"pulled": (top / "pulled.ok").exists()}
rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines() if x.strip()] if a["pulled"] else []
a["rounds_exact"] = [r["round"] for r in rows][:R] == list(range(1, R + 1)) and len({r["round"] for r in rows}) == len(rows)
a["finite"] = all(math.isfinite(r["train_loss"]) for r in rows)
a["max_skipped_frac"] = max((r.get("skipped_frac", 0) for r in rows), default=None)
a["skipped_ok"] = a["max_skipped_frac"] is not None and a["max_skipped_frac"] <= .01
a["cm_present"] = all((run / f"cms_r{r:03d}.npz").exists() for r in range(1, R + 1))
if a["cm_present"]:
    with np.load(run / f"cms_r{R:03d}.npz") as z, np.load(CTRL / f"cms_r{R:03d}.npz") as c:
        cm, cc = z["private"].astype(np.int64), c["private"].astype(np.int64)
        a["cm_support"] = bool(cm.shape == cc.shape == (5343, 16, 16) and (cm.sum(2) == cc.sum(2)).all())
        ten = {k: float(v) for k, v in summary(cm).items() if k in TEN}
        a["ten_finite"] = all(math.isfinite(v) for v in ten.values())
cfg = json.loads((run / "config.json").read_text())["hp"]
ref = dict(json.loads((CTRL / "config.json").read_text())["hp"], cop_kmeans_lite=6)
a["config"] = {k: v for k, v in cfg.items() if k in ref} == ref and cfg.get("cop_kmeans_lite") == 6
src = top / "provenance/source/methods/mobihet_sim.py"
a["overlay_sha"] = src.exists() and hashlib.sha256(src.read_bytes()).hexdigest() == spec["overlay_sha256"]["methods/mobihet_sim.py"]
log = glob.glob(str(top / "*.log"))
m = re.findall(r"KM_LITE_BYTES (\{.*?\})(?:\\n|KM_LITE_BYTES|$)", Path(log[0]).read_text()) if log else []
a["bytes_logged"] = bool(m)
a["pass"] = all(a.get(k) is True for k in ("pulled", "rounds_exact", "finite", "skipped_ok", "cm_present", "cm_support",
                                           "ten_finite", "config", "overlay_sha", "bytes_logged"))
json.dump(a, open(OUT / "audit.json", "w"), indent=1)
print("audit", "PASS" if a["pass"] else "FAIL", {k: v for k, v in a.items() if v is not True})
if not a["pass"]:
    sys.exit(1)

arm, ctl, core = Y.read(run, R), Y.read(CTRL, R), Y.read(CORE, R)
d = {k: arm[k] - ctl[k] for k in ("f1_macro", "p10", "low256", "timeDelay", "benign", "absent", "other15")}
gate = {"macro": d["f1_macro"] >= -.0015, "p10": d["p10"] >= -.006, "low256": d["low256"] >= -.003,
        "timeDelay": d["timeDelay"] >= -.0035, "benign": d["benign"] >= -.0005, "absent": d["absent"] >= -.01}
absent_mean = float(np.mean([Y.read(run, r)["absent"] - Y.read(CTRL, r)["absent"] for r in range(max(1, R - 4), R + 1)]))
byte = json.loads(m[0].replace('\\"', '"'))
out = {"round": R, "vs": "catbaochau/mh-k-km16q8-s42-r50/1", "ten": ten, "arm": arm, "control": ctl, "delta": d,
       "gate": {k: bool(v) for k, v in gate.items()}, "pass": bool(all(gate.values())), "absent_mean_last5": absent_mean,
       "delta_vs_core_f2": {k: arm[k] - core[k] for k in ("f1_macro", "p10", "low256", "timeDelay", "benign", "absent")},
       "bytes": byte}
json.dump(out, open(OUT / f"verdict-r{R}.json", "w"), indent=1)
print("lite6 vs KM50 Q42 r%d" % R, {k: round(v, 4) for k, v in d.items()}, "PASS" if out["pass"] else "FAIL",
      [k for k, v in gate.items() if not v])
print("absent mean last5", round(absent_mean, 4), "| vs Core F2", {k: round(v, 4) for k, v in out["delta_vs_core_f2"].items()})
print("bytes up", byte["total_up"], "down", byte["total_down"])
print("ten", {k: round(v, 4) for k, v in ten.items()})
