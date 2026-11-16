"""Audit + locked rule of the codec combo (docs/research/screening-codec.md "Combo r20", locked 10/11 before push).

    python scripts/analyse_codec_combo20.py [round=20]  -> runs/codec-combo20/{audit,verdict-r<round>}.json

Same audit and rule as scripts/analyse_codec_pilot20.py. Controls: KM arms (CBA/CBB) vs KM50 of the same seed; CQ8 vs C0
(in-batch Core, mobihet_sim knobs off); C0 vs F2 core s42 reported only (sanity: the sim path reproduces Core).
Combo verdict: an arm PASSes only when both of its seeds PASS. Bytes: KM_CODEC_BYTES for KM arms, rounds.jsonl for all.
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
OUT, ART = ROOT / "runs/codec-combo20", ROOT / "artifacts/codec-combo20"
KM50 = {42: ROOT / "artifacts/km50/mh-k-km16q8-s42-r50/mh-k-km16q8-s42-r50",
        43: ROOT / "artifacts/km50/mh-k-km16q8-s43-r50/mh-k-km16q8-s43-r50"}
F2 = ROOT / "artifacts/f50/mh-f2-core-s42-r50/mh-f2-core-s42-r50"
OFF = {"diagnostics", "teacher_diagnostics", "external_diagnostics", "diagnostic_ids", "diagnostic_rounds"}
KEYS = ("f1_macro", "accuracy", "p10", "low256", "timeDelay", "benign", "absent", "other15")
GUARD = {"p10": -.010, "low256": -.010, "absent": -.03, "timeDelay": -.005, "benign": -.001}
batch = json.loads((OUT / "batch.json").read_text())
by_arm = {s["arm"]: s for s in batch}
run_dir = lambda s: ART / s["slug"] / s["slug"]


def control(spec):
    if spec["km"]:
        return KM50[spec["seed"]]
    return run_dir(by_arm["C0"]) if spec["knobs"] else F2


audit, verdict = {}, {}
for spec in batch:
    arm, top = spec["arm"], ART / spec["slug"]
    run, ctl = run_dir(spec), control(spec)
    a = {"pulled": (top / "pulled.ok").exists()}
    rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines() if x.strip()] if a["pulled"] else []
    a["rounds_exact"] = [r["round"] for r in rows][:R] == list(range(1, R + 1)) and len({r["round"] for r in rows}) == len(rows)
    a["finite"] = bool(rows) and all(math.isfinite(r["train_loss"]) for r in rows)
    a["max_skipped_frac"] = max((r.get("skipped_frac", 0) for r in rows), default=None)
    a["skipped_ok"] = a["max_skipped_frac"] is not None and a["max_skipped_frac"] <= .01
    a["cm_present"] = a["pulled"] and all((run / f"cms_r{r:03d}.npz").exists() for r in range(1, R + 1))
    ten = None
    if a["cm_present"]:
        with np.load(run / f"cms_r{R:03d}.npz") as z, np.load(KM50[spec["seed"]] / f"cms_r{R:03d}.npz") as c:
            cm, cc = z["private"].astype(np.int64), c["private"].astype(np.int64)
            a["cm_support"] = bool(cm.shape == cc.shape == (5343, 16, 16) and (cm.sum(2) == cc.sum(2)).all())
            ten = {k: float(v) for k, v in summary(cm).items() if k in TEN}
            a["ten_finite"] = all(math.isfinite(v) for v in ten.values())
    log = []
    if a["pulled"]:
        cfg = json.loads((run / "config.json").read_text())["hp"]
        ref = dict(h.split("=", 1) for h in spec["hp"])               # launched hp (control hp + knobs, asserted at prepare)
        same = lambda x, v: x == json.loads(v) if v.startswith("[") else str(x).lower() == v.lower().strip('"')
        a["config"] = all(same(cfg.get(k), v) for k, v in ref.items())
        src = top / "provenance/source/methods/mobihet_sim.py"
        a["overlay_sha"] = src.exists() and hashlib.sha256(src.read_bytes()).hexdigest() == spec["overlay_sha256"]["methods/mobihet_sim.py"]
        log = glob.glob(str(top / "*.log"))
    m = re.findall(r"KM_CODEC_BYTES (\{.*?\})(?:\\n|KM_CODEC_BYTES|$)", Path(log[0]).read_text()) if log else []
    a["bytes_logged"] = bool(m) or not spec["km"]
    a["pass"] = all(a.get(k) is True for k in ("pulled", "rounds_exact", "finite", "skipped_ok", "cm_present", "cm_support",
                                               "ten_finite", "config", "overlay_sha", "bytes_logged"))
    audit[arm] = a
    print(arm, "audit", "PASS" if a["pass"] else "FAIL", {k: v for k, v in a.items() if v is not True}, flush=True)
    if not a["pass"] or (ctl != F2 and ctl.is_relative_to(ART) and not audit.get("C0", {}).get("pass")):
        continue
    per = {r: (Y.read(run, r), Y.read(ctl, r)) for r in range(max(1, R - 4), R + 1)}
    dmean = {k: float(np.mean([x[k] - c[k] for x, c in per.values()])) for k in KEYS}
    dR = {k: per[R][0][k] - per[R][1][k] for k in KEYS}
    gate = {"macro_mean": dmean["f1_macro"] >= -.0015, "macro_R": dR["f1_macro"] >= -.003}
    gate |= {k: dmean[k] >= v for k, v in GUARD.items()}
    rnd = [r for r in rows if r["round"] <= R]
    verdict[arm] = {"seed": spec["seed"], "control": str(ctl.relative_to(ROOT)), "report_only": ctl == F2, "ten": ten,
                    "delta_mean_r16_R": dmean, "delta_R": dR, "gate": {k: bool(v) for k, v in gate.items()},
                    "pass": bool(all(gate.values())), "bootstrap_bytes": json.loads(m[0].replace('\\"', '"')) if m else None,
                    "round_bytes_up_mean": float(np.mean([r["bytes_up"] for r in rnd])),
                    "round_bytes_down_mean": float(np.mean([r["bytes_down"] for r in rnd]))}
    print(f"{arm:6s} vs {ctl.name} Δmean " + " ".join(f"{k} {v:+.4f}" for k, v in dmean.items()),
          "| ΔR macro", f"{dR['f1_macro']:+.4f}", "PASS" if verdict[arm]["pass"] else "FAIL " + str([k for k, v in gate.items() if not v]),
          f"| round {verdict[arm]['round_bytes_up_mean']:.0f}+{verdict[arm]['round_bytes_down_mean']:.0f}", flush=True)
combo = {}
for name, arms in (("CBA", ("CBA42", "CBA43")), ("CBB", ("CBB42", "CBB43")), ("CQ8", ("CQ8",))):
    combo[name] = all(verdict.get(x, {}).get("pass") is True for x in arms) if all(x in verdict for x in arms) else None
print("COMBO", combo, flush=True)
json.dump(audit, open(OUT / "audit.json", "w"), indent=1)
json.dump({"round": R, "rule": "docs/research/screening-codec.md#combo-r20", "arms": verdict, "combo_pass": combo},
          open(OUT / f"verdict-r{R}.json", "w"), indent=1)
