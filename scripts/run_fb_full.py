"""FB full-scale pilot on DAGSNet (docs/research/screening-budget.md): sim-PASS arms N4 (selected) and N2 (owner 17/11: try every
PASS arm when quota allows), seed 42, 5 343 vehicles, r1-20 of the cosine-50 schedule. Reuses the Lite2 full-scale ladder
(scripts/run_lite2_full.py): same T3 control, collector and audit; only the arm knobs, method and the real_frac check differ.

    python scripts/run_fb_full.py quota            # scoped live quota -> runs/fb-full/quota.json
    python scripts/run_fb_full.py launch N4 [N2]   # lock batch.json (Lite2 pilot hp + sim arm knobs), push once per slug
    python scripts/run_fb_full.py collect          # one shot: status; pull terminal kernels into artifacts/fb-full/<slug>
    python scripts/run_fb_full.py audit            # Lite2 audit vs T3 seed 42 + measured real_frac == estimator -> audit.json

Pilot verdict (locked before push): delta Macro-F1 mean r16-20 >= -.006 and r20 >= -.006 vs T3 seed 42, bootstrap <= 382 000 B
per vehicle, real_frac every round == full-scale estimator.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_lite2_full as F  # noqa: E402
from scripts import run_sim_budget as SB  # noqa: E402
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json  # noqa: E402

OUT, ART = ROOT / "runs/fb-full", ROOT / "artifacts/fb-full"
CAP = 5.5                                                   # h: Lite2 pilot (same engine, padded slots) trained 3.64 h
F.OUT, F.ART, F.PILOT_OUT = OUT, ART, OUT
W.OUT, W.ART, W.TAG = OUT, ART, "FBF"
W.cli = SB.scoped_cli                                       # bounded auth/connect wrapper used by the FB sim


def real_frac(real):
    """Trained real rows / real rows over the full-scale fleet (cache/train_sizes.npy); same slot rule as run_sim_budget.cost."""
    n = np.load(ROOT / "cache/train_sizes.npy").astype(np.int64)
    r = np.asarray(real)[np.searchsorted(SB.TIER_MIN, n, side="right")]
    return float(((n // 64) * r + np.minimum(n % 64, r)).sum() / n.sum())


def rows(arms, quota):
    sim = {s["arm"]: s for s in json.loads((SB.OUT / "batch.json").read_text())}
    verdict = json.loads((SB.OUT / "verdict-r20.json").read_text())["arms"]
    lite = json.loads((ROOT / "runs/lite2-full/batch.json").read_text())[0]
    assert json.loads((ROOT / "runs/lite2-full/audit.json").read_text())["BU1P"]["verdict"] == "PASS"
    head, builder = W.git_head(), json.loads((ROOT / "runs/w-study/builder.json").read_text())
    slots = [q["user"] for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"]))
             if not (q.get("inventory_error") or q.get("unknown_recent") or q.get("gpu_reserved", 0) or q.get("active_recent"))
             and q.get("gpu_left", 0) >= CAP * 1.15 and q["user"] not in EXCLUDE]
    assert len(slots) >= len(arms), (arms, slots)
    batch = []
    for arm, owner in zip(arms, slots):
        s = sim[arm]
        assert verdict[arm]["verdict"] == "PASS" and s["overlay_sha256"] == {p: W.sha(ROOT / p) for p in SB.OVERLAYS}
        knobs = [h for h in s["hp"] if h.split("=")[0] in ("synth_tier_rows", "real_tier_rows")]
        real = json.loads(next(h for h in knobs if h.startswith("real_tier_rows")).split("=", 1)[1])
        hp = [h for h in lite["hp"] if h.split("=")[0] not in ("synth_tier_rows", "real_tier_rows")] + knobs
        slug = f"mh-fb-{SB.ARMS[arm][0]}-s42-r20"
        args = F.shlex.join(["scripts/run_q20.py", "--method", "mobihet_budget", "--out", f"/kaggle/working/{slug}", "--width", "1",
                             "--rounds", "50", "--until", "20", "--save-every", "5", "--compile", "--max-hours", "BUDGET",
                             "--cache", "", "--set", *hp])
        F.subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", head, "--run", slug,
                          "--max-hours", str(CAP), "--args", args, "--npy-kernel", builder["ref"],
                          "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *SB.OVERLAYS],
                         cwd=ROOT, check=True, capture_output=True)
        folder = ROOT / "notebooks" / slug
        nb, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
        doc = json.loads(nb.read_text())
        for c in doc["cells"]:
            if "BEGIN INLINE WANDB CREDENTIAL" in "".join(c["source"]):
                c["source"] = ['print("W&B off: FB full-scale (as Lite2/T3 pilot)")\n']
            if c["cell_type"] == "code":
                compile("".join(c["source"]), str(nb), "exec")
        nb.write_text(json.dumps(doc, indent=1) + "\n")
        m = json.loads(meta.read_text())
        runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
        assert m["is_private"] and m["id"] == f"{owner}/{slug}" and m["kernel_sources"] == [builder["ref"]]
        assert all(m[k] == runtime[k] for k in ("machine_shape", "docker_image"))
        batch.append(dict(arm=arm + "P", sim_arm=arm, seed=42, method="mobihet_budget", account=owner, slug=slug, version=1, hp=hp,
                          base_sha=head, overlay_sha256={p: W.sha(ROOT / p) for p in SB.OVERLAYS}, notebook_sha256=W.sha(nb),
                          metadata_sha256=W.sha(meta), cap_hours=CAP, endpoint=20, lr_schedule_rounds=50,
                          real_frac_estimator=real_frac(real), cost_x_fedavg=s["cost_x_fedavg"],
                          control="T3P seed 42 (b21-full/mh-k-b21-t3p-r20)", npy_kernel=builder["ref"],
                          npy_manifest_sha256=builder["manifest_sha256"]))
    return batch


EXCLUDE = set()                                             # owners reserved for the FeatureBERT pilots (set by launch argv)


def audit():
    F.audit()                                               # Lite2 checks + locked pilot gate (-.006 / -.006, 382 000 B)
    res = json.loads((OUT / "audit.json").read_text())
    for spec in json.loads((OUT / "batch.json").read_text()):
        a = res.get(spec["arm"])
        if not a or a.get("pulled") is False:
            continue
        f = next(iter(ART.glob(f"{spec['slug']}/**/rounds.jsonl")))
        fr = [json.loads(x).get("real_frac") for x in f.read_text().splitlines() if x.strip()]
        a["real_frac"] = fr[-1]
        a["checks"]["real_frac_eq_estimator"] = all(v is not None and abs(v - spec["real_frac_estimator"]) < 1e-6 for v in fr)
        a["checks"]["pass"] = all(bool(v) for k, v in a["checks"].items() if k != "pass")
        if not a["checks"]["pass"]:
            a["verdict"] = "FAIL"
        print(spec["arm"], "real_frac", fr[-1], "est", round(spec["real_frac_estimator"], 6), a["verdict"], flush=True)
    atomic_json(OUT / "audit.json", res)


def main():
    global EXCLUDE
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "quota":
        sys.argv = [sys.argv[0], "quota"]
        SB.OUT = OUT
        SB.main()
    elif cmd == "launch":
        arms = [a for a in sys.argv[2:] if not a.startswith("--exclude=")]
        EXCLUDE = {u for a in sys.argv[2:] if a.startswith("--exclude=") for u in a.split("=", 1)[1].split(",")}
        if not (OUT / "batch.json").exists():
            atomic_json(OUT / "batch.json", rows(arms, json.loads((OUT / "quota.json").read_text())))
        W.launch(json.loads((OUT / "batch.json").read_text()))
    elif cmd == "collect":
        sys.argv = [sys.argv[0], "collect"]
        F.main()
    else:
        assert cmd == "audit", cmd
        audit()


if __name__ == "__main__":
    main()
