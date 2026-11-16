"""B21 full-scale ladder (docs/research/screening-dose-b21.md): pilot r20 cosine50 s42 for T4/T3/T4F.

    python scripts/run_b21_full.py quota     # owner-scoped live quota -> runs/b21-full/quota.json
    python scripts/run_b21_full.py launch    # lock runs/b21-full/batch.json (CBB42 base/hp + tier knobs), push
    python scripts/run_b21_full.py collect   # one shot: status; pull terminal kernels into artifacts/b21-full/<slug>
    python scripts/run_b21_full.py audit     # run_b20_full.audit checks vs CBB same seed + locked B21 verdict
    python scripts/run_b21_full.py <cmd> t1  # same commands for the T1 r50 batch (runs/b21-t1; session 1 stops on its budget)

Same base SHA, data builder, overlays and CBB hp as run_b20_full.py; only the tier knobs, slug prefix and verdict differ.
Verdict (design, locked 13/11): delta macro mean last 5 rounds >= -.0015 and delta endpoint >= -.003 vs CBB same seed and
round; r50 additionally macro > .9048. No trajectory clause.
"""
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_b20_full as B  # noqa: E402
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

OUT, ART = ROOT / "runs/b21-full", ROOT / "artifacts/b21-full"
tier = lambda rows: ["synth_tier_min=[4096,16384]", "synth_tier_rows=" + rows, "synth_late_from=25"]
T4 = tier("[[64,64],[48,32],[32,16]]")
ARMS = [  # arm, seed, knobs, until, cap hours; cheapest FLOPs first (quota order)
    ("T4P", 42, T4, 20, 5.5),
    ("T3P", 42, tier("[[64,64],[64,32],[48,16]]"), 20, 5.5),
    ("T4FP", 42, T4 + ["synth_step_floor=48"], 20, 6.0),
]
if sys.argv[2:] == ["t1"]:   # T1 r50 seeds 42/43: first session on the hours left before the weekly reset, then resume same owner
    OUT, ART = ROOT / "runs/b21-t1", ROOT / "artifacts/b21-t1"
    ARMS = [("T142", 42, tier("[[64,64],[64,32],[64,32]]"), 50, 5.5), ("T143", 43, tier("[[64,64],[64,32],[64,32]]"), 50, 4.4)]
if sys.argv[2:] == ["s43"]:  # seed43 r50 for pilot-PASS arms T3/T4F (ladder step 2); session 1 on hours before reset, resume later
    OUT, ART = ROOT / "runs/b21-s43", ROOT / "artifacts/b21-s43"
    XFER = {"outlookcheck", "minhtran0601", "dinhan25", "annhtnt"}  # reserved by run_b21_transfer.py
    ARMS = [("T343", 43, tier("[[64,64],[64,32],[48,16]]"), 50, None), ("T4F43", 43, T4 + ["synth_step_floor=48"], 50, None)]
B.OUT, B.ART = OUT, ART
W.OUT, W.ART, W.TAG = OUT, ART, "B21F"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'


def rows(quota):
    ref = {r["arm"]: r for r in json.loads((ROOT / "runs/codec-combo50/batch.json").read_text())}
    assert W.sha(ROOT / "methods/mobihet_sim.py") == ref["CBB42"]["overlay_sha256"]["methods/mobihet_sim.py"]
    sim = json.loads((ROOT / "runs/sim-kaggle-b21/batch.json").read_text())[0]["overlay_sha256"]
    assert all(W.sha(ROOT / p) == sim[p] for p in B.OVERLAYS), "overlay differs from audited sim b21"
    builder = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    slots = []
    for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"])):
        if q.get("inventory_error") or q.get("unknown_recent") or q.get("gpu_reserved", 0) or q.get("active_recent"):
            continue                              # an active session keeps draining that owner's remaining hours
        slots += [(q["user"], q.get("gpu_left", 0))] * 2
    batch, used = [], {}
    for arm, seed, knobs, until, cap in ARMS:
        owner = next(u for u, left in slots if left - used.get(u, 0) >= cap * 1.15 and
                     sum(1 for b in batch if b["account"] == u) < sum(1 for s in slots if s[0] == u))
        used[owner] = used.get(owner, 0) + cap * 1.15
        slug = f"mh-k-b21-{arm.lower()}-r{until}"
        h = [x for x in ref["CBB42"]["hp"] if not x.startswith("seed=") and x.split("=")[0] not in {k.split("=")[0] for k in knobs}]
        h += [f"seed={seed}", *knobs]
        args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_compute", "--out", f"/kaggle/working/{slug}",
                           "--width", "1", "--rounds", "50", "--until", str(until), "--save-every", "5", "--compile",
                           "--max-hours", "BUDGET", "--cache", "", "--set", *h])
        subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", B.BASE,
                        "--run", slug, "--max-hours", str(cap), "--args", args, "--npy-kernel", builder["ref"],
                        "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *B.OVERLAYS],
                       cwd=ROOT, check=True, capture_output=True)
        folder = ROOT / "notebooks" / slug
        nb, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
        doc = json.loads(nb.read_text())
        for c in doc["cells"]:
            if "BEGIN INLINE WANDB CREDENTIAL" in "".join(c["source"]):
                c["source"] = ['print("W&B off: B21 full-scale (as CBB r50)")\n']
            if c["cell_type"] == "code":
                compile("".join(c["source"]), str(nb), "exec")
        nb.write_text(json.dumps(doc, indent=1) + "\n")
        m = json.loads(meta.read_text())
        assert m["is_private"] and m["id"] == f"{owner}/{slug}" and m["kernel_sources"] == [builder["ref"]]
        batch.append(dict(arm=arm, seed=seed, method="mobihet_compute", account=owner, slug=slug, version=1, hp=h,
                          base_sha=B.BASE, overlay_sha256={p: W.sha(ROOT / p) for p in B.OVERLAYS}, notebook_sha256=W.sha(nb),
                          metadata_sha256=W.sha(meta), cap_hours=cap, endpoint=until, lr_schedule_rounds=50,
                          control=f"CBB{seed} ({B.REF[seed]})", npy_kernel=builder["ref"],
                          npy_manifest_sha256=builder["manifest_sha256"]))
    return batch


def audit():
    B.audit()                                     # artifact checks; its verdict field follows the B20 rule
    out = json.loads((OUT / "audit.json").read_text())
    ends = {s["arm"]: s["endpoint"] for s in json.loads((OUT / "batch.json").read_text())}
    for arm, res in out.items():
        if "checks" not in res:
            continue
        ok = res["delta_mean_last5"] >= -.0015 and res["delta_end"] >= -.003
        if ends[arm] == 50:
            ok = ok and res["endpoint"]["f1_macro"] > B.TOP3
        res["verdict"], res["rule"] = "PASS" if ok and res["checks"]["pass"] else "FAIL", "b21"
        print(arm, "B21", res["verdict"], "d_mean_last5 %+.4f d_end %+.4f macro %.4f" %
              (res["delta_mean_last5"], res["delta_end"], res["endpoint"]["f1_macro"]), flush=True)
    atomic_json(OUT / "audit.json", out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "quota":
        q = W.inventory()
        for a in q["accounts"]:
            print(a["user"], a.get("gpu_left"), a.get("active_recent"), a.get("inventory_error", ""))
    elif cmd == "launch":
        if sys.argv[2:] == ["s43"] and not (OUT / "batch.json").exists():
            q = json.loads((OUT / "quota.json").read_text())
            q["accounts"] = [a for a in q["accounts"] if a["user"] not in XFER]
            free = sorted((a["gpu_left"] for a in q["accounts"] if not (a.get("inventory_error") or a.get("unknown_recent")
                           or a.get("gpu_reserved", 0) or a.get("active_recent"))), reverse=True)
            for i, (arm, seed, knobs, until, _) in enumerate(ARMS):
                ARMS[i] = (arm, seed, knobs, until, min(11., int((free[i] / 1.15 - .1) * 100) / 100))
            assert all(a[4] >= 1.5 for a in ARMS), ARMS
            atomic_json(OUT / "batch.json", rows(q))
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
        done = lambda r: all(s.get("terminal") for s, b in zip(states, batch) if b["endpoint"] == r)
        print("ALL_TERMINAL" if done(20) and done(50) else "PILOTS_TERMINAL" if done(20) else "WAITING", flush=True)
    elif cmd == "audit":
        audit()


if __name__ == "__main__":
    main()
