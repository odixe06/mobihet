"""B20 full-scale ladder (docs/research/screening-compute-b20.md + addendum): SD r50 seeds 42/43, SM pilot r20 cosine50 s42.

    python scripts/run_b20_full.py quota     # owner-scoped live quota -> runs/b20-full/quota.json
    python scripts/run_b20_full.py launch    # lock runs/b20-full/batch.json (CBB42 base/hp + knob), push
    python scripts/run_b20_full.py collect   # one shot: status; pull terminal kernels into artifacts/b20-full/<slug>
    python scripts/run_b20_full.py audit     # exact rounds/hp/overlay/CM support/10 metrics vs CBB42/43 + locked verdicts

Same base SHA, data builder, overlay mobihet_sim.py and hp as CBB42/CBB43 r50 (runs/codec-combo50/batch.json) except seed and
the arm knob; method mobihet_compute (+ mobihet_refine) as in the audited sim B20. W&B off as CBB r50.
"""
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

OUT, ART = ROOT / "runs/b20-full", ROOT / "artifacts/b20-full"
BASE = "be3b7d5e34a094bab6ea8e6d2f6f921fd24f4a61"                 # CBB42/43 base
OVERLAYS = ["methods/mobihet_sim.py", "methods/mobihet_refine.py", "methods/mobihet_compute.py"]
TOP3 = .9048                                                      # FedAvg 3-seed r50 macro, rank 4 (addendum §4)
ARMS = [  # arm, seed, knobs, until, cap hours
    ("SD42", 42, ["synth_grad=full", "synth_rows_late=32", "synth_late_from=25"], 50, 11.5),
    ("SD43", 43, ["synth_grad=full", "synth_rows_late=32", "synth_late_from=25"], 50, 11.5),
    ("SM42P", 42, ["synth_grad=mix16"], 20, 5.5),
]
REF = {42: "mh-k-cbb-s42-r50-c2/mh-k-cbb-s42-p20", 43: "mh-k-cbb-s43-r50-c2/mh-k-cbb-s43-p20"}
W.OUT, W.ART, W.TAG = OUT, ART, "B20F"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'


def rows(quota):
    ref = {r["arm"]: r for r in json.loads((ROOT / "runs/codec-combo50/batch.json").read_text())}
    strip = lambda hp: [x for x in hp if not x.startswith("seed=")]
    assert strip(ref["CBB42"]["hp"]) == strip(ref["CBB43"]["hp"])
    assert W.sha(ROOT / "methods/mobihet_sim.py") == ref["CBB42"]["overlay_sha256"]["methods/mobihet_sim.py"]
    builder = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    slots = []
    for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"])):
        if q.get("inventory_error") or q.get("unknown_recent") or q.get("gpu_reserved", 0):
            continue
        slots += [(q["user"], q.get("gpu_left", 0))] * max(0, 2 - len(q.get("active_recent", [])))
    batch, used = [], {}
    for arm, seed, knobs, until, cap in ARMS:
        owner = next(u for u, left in slots if left - used.get(u, 0) >= cap * 1.15 and
                     sum(1 for b in batch if b["account"] == u) < sum(1 for s in slots if s[0] == u))
        used[owner] = used.get(owner, 0) + cap * 1.15
        slug = f"mh-k-b20-{arm.lower()}-r{until}"
        h = [x for x in ref["CBB42"]["hp"] if not x.startswith("seed=")] + [f"seed={seed}", *knobs]
        args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_compute", "--out", f"/kaggle/working/{slug}",
                           "--width", "1", "--rounds", "50", "--until", str(until), "--save-every", "5", "--compile",
                           "--max-hours", "BUDGET", "--cache", "", "--set", *h])
        subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", BASE,
                        "--run", slug, "--max-hours", str(cap), "--args", args, "--npy-kernel", builder["ref"],
                        "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *OVERLAYS],
                       cwd=ROOT, check=True, capture_output=True)
        folder = ROOT / "notebooks" / slug
        nb, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
        doc = json.loads(nb.read_text())
        for c in doc["cells"]:
            if "BEGIN INLINE WANDB CREDENTIAL" in "".join(c["source"]):
                c["source"] = ['print("W&B off: B20 full-scale (as CBB r50)")\n']
            if c["cell_type"] == "code":
                compile("".join(c["source"]), str(nb), "exec")
        nb.write_text(json.dumps(doc, indent=1) + "\n")
        m = json.loads(meta.read_text())
        assert m["is_private"] and m["id"] == f"{owner}/{slug}" and m["kernel_sources"] == [builder["ref"]]
        batch.append(dict(arm=arm, seed=seed, method="mobihet_compute", account=owner, slug=slug, version=1, hp=h,
                          base_sha=BASE, overlay_sha256={p: W.sha(ROOT / p) for p in OVERLAYS}, notebook_sha256=W.sha(nb),
                          metadata_sha256=W.sha(meta), cap_hours=cap, endpoint=until, lr_schedule_rounds=50,
                          control=f"CBB{seed} ({REF[seed]})", npy_kernel=builder["ref"],
                          npy_manifest_sha256=builder["manifest_sha256"]))
    return batch


def slope(y):
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


def audit():
    """Per arm: rounds 1..endpoint exact, finite, skip <= 1 %, hp == launched, overlay SHAs executed, CM support == CBB
    same seed, 10 metrics from CM == log, LR == CBB; then the locked verdict (addendum) and 10-metric endpoint tables."""
    from core.metrics import TEN, summary
    out = {}
    for spec in json.loads((OUT / "batch.json").read_text()):
        f = next(iter(ART.glob(f"{spec['slug']}/**/rounds.jsonl")), None)
        if f is None:
            out[spec["arm"]] = {"pulled": False}
            continue
        d, dr = f.parent, ROOT / "artifacts/codec-combo50" / REF[spec["seed"]]
        R = spec["endpoint"]
        rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        ref = [json.loads(x) for x in (dr / "rounds.jsonl").read_text().splitlines() if x.strip()]
        cfg = json.loads((d / "config.json").read_text())
        chk = {"rounds_exact": [r["round"] for r in rs] == list(range(1, R + 1)), "n_params": cfg.get("n_params") == 395408,
               "finite_loss": all(np.isfinite(r["train_loss"]) for r in rs),
               "skip_le_1pct": all(r["skipped_frac"] <= .01 for r in rs),
               "clients_applied": all(r.get("clients_without_applied", 0) == 0 for r in rs),
               "lr_eq_cbb": all(abs(a["lr"] - b["lr"]) < 1e-12 for a, b in zip(rs, ref))}
        hp = {}
        for kv in spec["hp"]:
            k, v = kv.split("=", 1)
            try:
                v = json.loads(v)
            except json.JSONDecodeError:
                pass
            hp[k] = v
        chk["hp_eq_launched"] = all(cfg["hp"].get(k) == v for k, v in hp.items())
        man = next(iter(ART.glob(f"{spec['slug']}/**/source_manifest.json")), None)
        chk["overlay_sha"] = bool(man) and json.loads(man.read_text()).get("overlay") == spec["overlay_sha256"]
        macro, cbb, ok_cm, ok_m = [], [], True, True
        for r in rs:
            with np.load(d / f"cms_r{r['round']:03d}.npz") as z, np.load(dr / f"cms_r{r['round']:03d}.npz") as y:
                ok_cm &= bool(np.array_equal(z["private"].sum(2), y["private"].sum(2)) and np.array_equal(z["ids"], y["ids"]))
                ten, tr = summary(z["private"]), summary(y["private"])
                ok_m &= all(abs(ten[k] - r["private/" + k]) < 1e-9 for k in TEN)
                macro.append(ten["f1_macro"]); cbb.append(tr["f1_macro"])
        chk.update(cm_support_eq_cbb=ok_cm, ten_metrics_from_cm=ok_m)
        chk["pass"] = all(bool(v) for v in chk.values())
        delta = np.array(macro) - np.array(cbb)
        res = {"checks": chk, "endpoint": ten, "cbb_endpoint": tr, "delta_end": float(delta[-1]),
               "delta_mean_last5": float(delta[-5:].mean()), "delta_slope_last_half": slope(delta[R // 2:]),
               "t_train_hours": sum(r["t_train"] for r in rs) / 3600,
               "cbb_t_train_hours": sum(r["t_train"] for r in ref[:R]) / 3600}
        if R == 50:   # addendum §4: stay in the top 3
            res["verdict"] = "PASS" if ten["f1_macro"] > TOP3 else "FAIL"
        else:         # pilot r20: non-inferiority, or trajectory rule (addendum §3)
            ni = res["delta_mean_last5"] >= -.0015 and res["delta_end"] >= -.003
            traj = res["delta_slope_last_half"] > 0 and res["delta_end"] >= -.005
            res["verdict"] = "PASS" if ni else "PASS_TRAJECTORY" if traj else "FAIL"
        out[spec["arm"]] = res
        print(spec["arm"], "audit", "PASS" if chk["pass"] else "FAIL", {k: v for k, v in chk.items() if not v},
              "macro %.4f vs CBB %.4f" % (ten["f1_macro"], tr["f1_macro"]), res["verdict"], flush=True)
    atomic_json(OUT / "audit.json", out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "quota":
        q = W.inventory()
        for a in q["accounts"]:
            print(a["user"], a.get("gpu_left"), a.get("active_recent"), a.get("inventory_error", ""))
    elif cmd == "launch":
        if not (OUT / "batch.json").exists():
            atomic_json(OUT / "batch.json", rows(json.loads((OUT / "quota.json").read_text())))
        W.launch(json.loads((OUT / "batch.json").read_text()))
    elif cmd == "collect":
        batch = json.loads((OUT / "batch.json").read_text())
        states = [query(s) for s in batch]
        atomic_json(OUT / "status.json", states)
        for spec, st in zip(batch, states):
            print(spec["arm"], spec["slug"], st.get("status"), flush=True)
            if st.get("terminal") and not (ART / spec["slug"] / "pulled.ok").exists() and W.collect(spec) == 0:
                (ART / spec["slug"] / "pulled.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        print("ALL_TERMINAL" if all(s.get("terminal") for s in states) else "WAITING", flush=True)
    elif cmd == "audit":
        audit()


if __name__ == "__main__":
    main()
