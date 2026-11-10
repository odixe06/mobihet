"""Audit + locked rule of codec combo r50 (docs/research/screening-codec.md "Combo r50", locked 10/11 before push).

    python scripts/analyse_codec_combo50.py [--no-query]   -> runs/codec-combo50/{audit,verdict-r50,progress-live}.json

Pulled arms (artifacts/codec-combo50/<kernel>/pulled.ok): audit as combo r20 (exact rounds, finite, skip <= 1 %, CM support
== control, hp == launched hp, overlay sha, KM bytes logged) plus r50 checks: LR per round == control (cosine50 kept);
resumed arms also need both ranks resumed at r21 and rows/CMs r1-r20 identical to the audited r20 pilot.
Rule: Δ mean r46-r50 macro >= -.0015 and Δ macro at r50 >= -.003; guards on the r46-r50 mean. Arm PASS = both seeds PASS.
A partial arm (< 50 committed rounds) gets no verdict, only its last committed round vs control at the same round.
Not-pulled arms: live Kaggle status (exact owner/slug/version) and an elapsed-time progress estimate from the control's
per-round times; Kaggle exposes no output/log while RUNNING, so this is an estimate, not evidence.
"""
import datetime
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

E = 50
OUT, ART, PILOT = ROOT / "runs/codec-combo50", ROOT / "artifacts/codec-combo50", ROOT / "artifacts/codec-combo20"
KM50 = {42: ROOT / "artifacts/km50/mh-k-km16q8-s42-r50/mh-k-km16q8-s42-r50",
        43: ROOT / "artifacts/km50/mh-k-km16q8-s43-r50/mh-k-km16q8-s43-r50"}
CORE = {42: ROOT / "artifacts/f50/mh-f2-core-s42-r50/mh-f2-core-s42-r50",
        43: ROOT / "artifacts/f50/mh-f3-core-s43-r50/mh-f3-core-s43-r50"}
KEYS = ("f1_macro", "accuracy", "p10", "low256", "timeDelay", "benign", "absent", "other15")
GUARD = {"p10": -.010, "low256": -.010, "absent": -.03, "timeDelay": -.005, "benign": -.001}
ARMS = {"CBB": ("CBB42", "CBB43"), "CQ8": ("CQ8", "CQ8s43")}
ROW_KEYS = ("round", "lr", "train_loss", "skipped_frac", "bytes_up", "bytes_down", "teacher_produced_sha256")

batch = json.loads((OUT / "batch.json").read_text())
control = lambda s: (KM50 if s["km"] else CORE)[s["seed"]]
run_dir = lambda s: ART / s["slug"] / s.get("run_slug", s["slug"])
rows_of = lambda d: [json.loads(x) for x in (d / "rounds.jsonl").read_text().splitlines() if x.strip()]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def live_status(spec):
    if "--no-query" in sys.argv:
        return {"status": "NOT_QUERIED"}
    from scripts.watch_kaggle_batch import query
    return query(spec)


def progress(spec, st):
    """Elapsed since push vs the control's cumulative per-round time (an estimate only)."""
    push = json.loads((OUT / f"{spec['slug']}.push.json").read_text())
    t0 = datetime.datetime.strptime(push["at"], "%Y-%m-%dT%H:%M:%S%z")
    el = (datetime.datetime.now().astimezone() - t0).total_seconds() / 3600
    start = spec.get("resume_round", 0)
    per = [(r["t_train"] + r["t_eval"]) / 3600 for r in rows_of(control(spec))[start:E]]
    cum = np.cumsum(per)
    est = start + int(np.searchsorted(cum, el, side="right"))
    return {"ref": f"{spec['account']}/{spec['slug']}/{spec['version']}", "status": st.get("status"), "pushed_at": push["at"],
            "elapsed_h": round(el, 2), "cap_h": spec["cap_hours"], "need_h_est": round(float(cum[-1]), 2),
            "round_est": min(est, E), "eta_h_est": round(max(float(cum[-1]) - el, 0), 2),
            "fits_cap_est": bool(cum[-1] <= spec["cap_hours"]), "basis": f"{control(spec).name} t_train+t_eval, no setup"}


def resume_ok(spec, run, rows, log_text):
    """Both ranks resumed at r21; rows and CMs r1-r20 equal the audited r20 pilot."""
    old = PILOT / spec["run_slug"] / spec["run_slug"]
    prev = rows_of(old)
    same_rows = len(prev) == 20 and all({k: a.get(k) for k in ROW_KEYS} == {k: b.get(k) for k in ROW_KEYS}
                                        for a, b in zip(rows[:20], prev))
    same_cms = all(sha(run / f"cms_r{r:03d}.npz") == sha(old / f"cms_r{r:03d}.npz") for r in range(1, 21))
    ranks = all(f"resumed rank{k} at round 21" in log_text for k in (0, 1))
    return {"resume_ranks_r21": ranks, "resume_rows_r1_20": same_rows, "resume_cms_r1_20": same_cms}


audit, verdict, live, partial = {}, {}, {}, {}
for spec in batch:
    arm, top, run, ctl = spec["arm"], ART / spec["slug"], run_dir(spec), control(spec)
    a = {"pulled": (top / "pulled.ok").exists()}
    if not a["pulled"]:
        live[arm] = progress(spec, live_status(spec))
        print(arm, "not pulled:", live[arm], flush=True)
        continue
    rows = rows_of(run)
    done = [r["round"] for r in rows]
    a["rounds"] = len(rows)
    a["rounds_exact"] = done == list(range(1, len(rows) + 1))
    a["complete"] = len(rows) == E
    a["finite"] = bool(rows) and all(math.isfinite(r["train_loss"]) for r in rows)
    a["max_skipped_frac"] = max((r.get("skipped_frac", 0) for r in rows), default=None)
    a["skipped_ok"] = a["max_skipped_frac"] is not None and a["max_skipped_frac"] <= .01
    a["cm_present"] = bool(rows) and all((run / f"cms_r{r:03d}.npz").exists() for r in done)
    ctl_rows = {r["round"]: r for r in rows_of(ctl)}
    a["lr_eq_control"] = all(math.isclose(r["lr"], ctl_rows[r["round"]]["lr"], rel_tol=1e-6) for r in rows)
    last, ten = (done[-1] if done else None), None
    if a["cm_present"] and last:
        with np.load(run / f"cms_r{last:03d}.npz") as z, np.load(ctl / f"cms_r{last:03d}.npz") as c:
            cm, cc = z["private"].astype(np.int64), c["private"].astype(np.int64)
            a["cm_support"] = bool(cm.shape == cc.shape == (5343, 16, 16) and (cm.sum(2) == cc.sum(2)).all())
            ten = {k: float(v) for k, v in summary(cm).items() if k in TEN}
            a["ten_finite"] = all(math.isfinite(v) for v in ten.values())
    cfg = json.loads((run / "config.json").read_text())["hp"]
    ref = dict(h.split("=", 1) for h in spec["hp"])
    same = lambda x, v: x == json.loads(v) if v.startswith("[") else str(x).lower() == v.lower().strip('"')
    a["config"] = all(same(cfg.get(k), v) for k, v in ref.items()) and cfg.get("rounds") == E
    src = top / "provenance/source/methods/mobihet_sim.py"
    a["overlay_sha"] = src.exists() and sha(src) == spec["overlay_sha256"]["methods/mobihet_sim.py"]
    log = glob.glob(str(top / "*.log"))
    text = Path(log[0]).read_text() if log else ""
    m = re.findall(r"KM_CODEC_BYTES (\{.*?\})(?:\\n|KM_CODEC_BYTES|$)", text)
    a["bytes_logged"] = bool(m) or not spec["km"]
    checks = ["pulled", "rounds_exact", "finite", "skipped_ok", "cm_present", "cm_support", "ten_finite", "config",
              "overlay_sha", "bytes_logged", "lr_eq_control"]
    if spec.get("resume_from"):
        a |= resume_ok(spec, run, rows, text)
        checks += ["resume_ranks_r21", "resume_rows_r1_20", "resume_cms_r1_20"]
    a["pass"] = all(a.get(k) is True for k in checks)
    audit[arm] = a
    print(arm, f"r{last}", "audit", "PASS" if a["pass"] else "FAIL",
          {k: v for k, v in a.items() if v is not True and k not in ("rounds", "max_skipped_frac")}, flush=True)
    if not a["pass"]:
        continue
    byte = {"bootstrap_bytes": json.loads(m[0].replace('\\"', '"')) if m else None,
            "round_bytes_up_mean": float(np.mean([r["bytes_up"] for r in rows])),
            "round_bytes_down_mean": float(np.mean([r["bytes_down"] for r in rows])),
            "rounds_bytes_total": float(sum(r["bytes_up"] + r["bytes_down"] for r in rows))}
    if not a["complete"]:                       # partial: report only, no verdict (budget stop is not a utility FAIL)
        x, c = Y.read(run, last), Y.read(ctl, last)
        partial[arm] = {"round": last, "control": str(ctl.relative_to(ROOT)), "ten": ten,
                        "delta_at_round": {k: x[k] - c[k] for k in KEYS}} | byte
        print(f"{arm:6s} PARTIAL r{last} vs {ctl.name} same round Δmacro {x['f1_macro'] - c['f1_macro']:+.4f}", flush=True)
        continue
    per = {r: (Y.read(run, r), Y.read(ctl, r)) for r in range(E - 4, E + 1)}
    dmean = {k: float(np.mean([x[k] - c[k] for x, c in per.values()])) for k in KEYS}
    dE = {k: per[E][0][k] - per[E][1][k] for k in KEYS}
    gate = {"macro_mean": dmean["f1_macro"] >= -.0015, "macro_R": dE["f1_macro"] >= -.003}
    gate |= {k: dmean[k] >= v for k, v in GUARD.items()}
    verdict[arm] = {"seed": spec["seed"], "control": str(ctl.relative_to(ROOT)), "ten": ten,
                    "own_r50": per[E][0], "control_r50": per[E][1], "delta_mean_r46_r50": dmean, "delta_r50": dE,
                    "gate": {k: bool(v) for k, v in gate.items()}, "pass": bool(all(gate.values()))} | byte
    print(f"{arm:6s} vs {ctl.name} Δmean " + " ".join(f"{k} {v:+.4f}" for k, v in dmean.items()),
          "| Δr50 macro", f"{dE['f1_macro']:+.4f}",
          "PASS" if verdict[arm]["pass"] else "FAIL " + str([k for k, v in gate.items() if not v]),
          f"| round {byte['round_bytes_up_mean']:.0f}+{byte['round_bytes_down_mean']:.0f} B", flush=True)

combo = {n: (all(verdict[x]["pass"] for x in xs) if all(x in verdict for x in xs) else None) for n, xs in ARMS.items()}
print("COMBO", combo, "(None = waiting for a seed)", flush=True)

# Owner table: Core+QQ8 vs KM+lite6+QQ8, mean over the seeds with a verdict (10 metrics r50, guards, bytes).
table = {}
for n, xs in ARMS.items():
    got = [verdict[x] for x in xs if x in verdict]
    if not got:
        continue
    mean = lambda f: float(np.mean([f(v) for v in got]))
    boot = [v["bootstrap_bytes"]["total_unicast_mean"] for v in got if v["bootstrap_bytes"]]
    table[n] = {"seeds": [v["seed"] for v in got], **{k: mean(lambda v: v["ten"][k]) for k in TEN},
                **{k: mean(lambda v: v["own_r50"][k]) for k in ("p10", "low256", "absent", "timeDelay", "benign")},
                "round_bytes_mean": mean(lambda v: v["round_bytes_up_mean"] + v["round_bytes_down_mean"]),
                "rounds50_bytes_total": mean(lambda v: v["rounds_bytes_total"]),
                "bootstrap_unicast_mean": float(np.mean(boot)) if boot else None}
if table:
    cols = list(TEN) + ["p10", "low256", "absent", "timeDelay", "benign"]
    print("\n| arm | seeds | " + " | ".join(cols) + " | B/round | 50r B (no boot) | bootstrap B |")
    print("|---|---|" + "---:|" * (len(cols) + 3))
    for n, t in table.items():
        b = f"{t['bootstrap_unicast_mean']:.0f}" if t["bootstrap_unicast_mean"] else "n/a (Core: screening-codec)"
        print(f"| {n} | {t['seeds']} | " + " | ".join(f"{t[c]:.4f}" for c in cols)
              + f" | {t['round_bytes_mean']:.0f} | {t['rounds50_bytes_total']:.0f} | {b} |")

json.dump(audit, open(OUT / "audit.json", "w"), indent=1)
json.dump({"round": E, "rule": "docs/research/screening-codec.md#combo-r50--luật-khóa-10112026-trước-push",
           "arms": verdict, "partial": partial, "combo_pass": combo, "table": table},
          open(OUT / f"verdict-r{E}.json", "w"), indent=1)
json.dump({"at": datetime.datetime.now().astimezone().isoformat(), "arms": live}, open(OUT / "progress-live.json", "w"), indent=1)
