"""Audit + locked rule of the codec study (docs/research/screening-codec.md, locked 09/11 before push).

    python scripts/analyse_codec_pilot20.py [round=20] [batch=codec-pilot20]  -> runs/<batch>/{audit,verdict-r<round>}.json
    (batch g1-pilot20: G1 synthetic rows, same rule; t_train ratio vs control reported as the compute measure)

Audit per arm (artifact only): pulled.ok, rounds 1..R once, finite loss, skipped <= 1 %, cms_r001..R, CM support == control,
config hp == control hp + the arm's knobs, executed overlay sha == batch, ten metrics finite, KM_CODEC_BYTES logged.
Rule (paired vs KM50 of the same seed; mean of r16..R deltas unless stated):
  primary  Δf1_macro mean >= -.0015 and Δf1_macro at R >= -.003;
  relaxed guards (owner 09/11: small-class guards may loosen while system F1 holds): Δp10 >= -.010, Δlow256 >= -.010,
  Δabsent >= -.03, ΔtimeDelay >= -.005, Δbenign >= -.001.
Bytes: bootstrap from KM_CODEC_BYTES (exact per vehicle, unicast + broadcast copy), per round from rounds.jsonl bytes_up/down.
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
BATCH = sys.argv[2] if len(sys.argv) > 2 else "codec-pilot20"
OUT, ART = ROOT / "runs" / BATCH, ROOT / "artifacts" / BATCH
CTRL = {42: ROOT / "artifacts/km50/mh-k-km16q8-s42-r50/mh-k-km16q8-s42-r50",
        43: ROOT / "artifacts/km50/mh-k-km16q8-s43-r50/mh-k-km16q8-s43-r50"}
KEYS = ("f1_macro", "accuracy", "p10", "low256", "timeDelay", "benign", "absent", "other15")
GUARD = {"p10": -.010, "low256": -.010, "absent": -.03, "timeDelay": -.005, "benign": -.001}
batch = json.loads((OUT / "batch.json").read_text())
audit, verdict = {}, {}
for spec in batch:
    arm, top = spec["arm"], ART / spec["slug"]
    run, ctl = top / spec["slug"], CTRL[spec["seed"]]
    a = {"pulled": (top / "pulled.ok").exists()}
    rows = [json.loads(x) for x in (run / "rounds.jsonl").read_text().splitlines() if x.strip()] if a["pulled"] else []
    a["rounds_exact"] = [r["round"] for r in rows][:R] == list(range(1, R + 1)) and len({r["round"] for r in rows}) == len(rows)
    a["finite"] = bool(rows) and all(math.isfinite(r["train_loss"]) for r in rows)
    a["max_skipped_frac"] = max((r.get("skipped_frac", 0) for r in rows), default=None)
    a["skipped_ok"] = a["max_skipped_frac"] is not None and a["max_skipped_frac"] <= .01
    a["cm_present"] = a["pulled"] and all((run / f"cms_r{r:03d}.npz").exists() for r in range(1, R + 1))
    ten = None
    if a["cm_present"]:
        with np.load(run / f"cms_r{R:03d}.npz") as z, np.load(ctl / f"cms_r{R:03d}.npz") as c:
            cm, cc = z["private"].astype(np.int64), c["private"].astype(np.int64)
            a["cm_support"] = bool(cm.shape == cc.shape == (5343, 16, 16) and (cm.sum(2) == cc.sum(2)).all())
            ten = {k: float(v) for k, v in summary(cm).items() if k in TEN}
            a["ten_finite"] = all(math.isfinite(v) for v in ten.values())
    if a["pulled"]:
        cfg = json.loads((run / "config.json").read_text())["hp"]
        ref = json.loads((ctl / "config.json").read_text())["hp"]
        knobs = dict(h.split("=", 1) for h in spec["hp"][len(spec["hp"]) - len(spec["knobs"]):]) if spec.get("knobs") else {}
        a["config"] = {k: v for k, v in cfg.items() if k in ref and k not in knobs} == {k: v for k, v in ref.items() if k not in knobs} \
            and all(str(cfg.get(k)).lower() == str(v).lower() for k, v in knobs.items())
        src = top / "provenance/source/methods/mobihet_sim.py"
        a["overlay_sha"] = src.exists() and hashlib.sha256(src.read_bytes()).hexdigest() == spec["overlay_sha256"]["methods/mobihet_sim.py"]
        log = glob.glob(str(top / "*.log"))
        m = re.findall(r"KM_CODEC_BYTES (\{.*?\})(?:\\n|KM_CODEC_BYTES|$)", Path(log[0]).read_text()) if log else []
        a["bytes_logged"] = bool(m)
    a["pass"] = all(a.get(k) is True for k in ("pulled", "rounds_exact", "finite", "skipped_ok", "cm_present", "cm_support",
                                               "ten_finite", "config", "overlay_sha", "bytes_logged"))
    audit[arm] = a
    print(arm, "audit", "PASS" if a["pass"] else "FAIL", {k: v for k, v in a.items() if v is not True}, flush=True)
    if not a["pass"]:
        continue
    per = {r: (Y.read(run, r), Y.read(ctl, r)) for r in range(max(1, R - 4), R + 1)}
    dmean = {k: float(np.mean([x[k] - c[k] for x, c in per.values()])) for k in KEYS}
    dR = {k: per[R][0][k] - per[R][1][k] for k in KEYS}
    gate = {"macro_mean": dmean["f1_macro"] >= -.0015, "macro_R": dR["f1_macro"] >= -.003}
    gate |= {k: dmean[k] >= v for k, v in GUARD.items()}
    byte = json.loads(m[0].replace('\\"', '"'))
    rnd = [r for r in rows if r["round"] <= R]
    verdict[arm] = {"seed": spec["seed"], "ten": ten, "delta_mean_r16_R": dmean, "delta_R": dR,
                    "gate": {k: bool(v) for k, v in gate.items()}, "pass": bool(all(gate.values())), "bootstrap_bytes": byte,
                    "round_bytes_up_mean": float(np.mean([r["bytes_up"] for r in rnd])),
                    "round_bytes_down_mean": float(np.mean([r["bytes_down"] for r in rnd])),
                    "t_train_ratio": float(np.mean([r["t_train"] for r in rnd[1:]]) / np.mean(
                        [json.loads(x)["t_train"] for x in (ctl / "rounds.jsonl").read_text().splitlines() if x.strip()][1:R]))}
    print(f"{arm:7s} Δmean " + " ".join(f"{k} {v:+.4f}" for k, v in dmean.items()), "| ΔR macro", f"{dR['f1_macro']:+.4f}",
          "PASS" if verdict[arm]["pass"] else "FAIL " + str([k for k, v in gate.items() if not v]),
          f"| boot up {byte['up_mean']:.0f} down {byte['down_unicast_mean']:.0f} bcast {byte['broadcast_copy']} "
          f"| round {verdict[arm]['round_bytes_up_mean']:.0f}+{verdict[arm]['round_bytes_down_mean']:.0f}"
          f" | t_train x{verdict[arm]['t_train_ratio']:.3f}", flush=True)
json.dump(audit, open(OUT / "audit.json", "w"), indent=1)
json.dump({"round": R, "rule": "docs/research/screening-codec.md", "arms": verdict}, open(OUT / f"verdict-r{R}.json", "w"), indent=1)
