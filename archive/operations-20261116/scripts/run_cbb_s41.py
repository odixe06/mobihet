"""CBB seed 41, full-scale r50 from r1 (owner 12/11/2026): same base SHA, overlay and hp as CBB42/CBB43 except seed.

    python scripts/run_cbb_s41.py launch    # build notebook, lock runs/cbb-s41/batch.json, push
    python scripts/run_cbb_s41.py collect   # one shot: status; pull a terminal kernel into artifacts/cbb-s41/<slug>
    python scripts/run_cbb_s41.py audit     # audit pulled run vs CBB42/43 r50 + 10-metric table -> runs/cbb-s41/{audit.json,report.md}
"""
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_w20 as W  # noqa: E402
from scripts import run_km50 as K  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

OUT, ART = ROOT / "runs/cbb-s41", ROOT / "artifacts/cbb-s41"
OWNER, SEED, CAP, SLUG = "thuyanh316", 41, 11.5, "mh-k-cbb-s41-r50"
BASE = "be3b7d5e34a094bab6ea8e6d2f6f921fd24f4a61"                 # CBB42/43 pilot base
W.OUT, W.ART, W.TAG = OUT, ART, "CBB41"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'


def row():
    ref = {r["arm"]: r for r in json.loads((ROOT / "runs/codec-combo50/batch.json").read_text())}["CBB42"]
    ref43 = {r["arm"]: r for r in json.loads((ROOT / "runs/codec-combo50/batch.json").read_text())}["CBB43"]
    strip = lambda hp: [x for x in hp if not x.startswith("seed=")]
    assert strip(ref["hp"]) == strip(ref43["hp"]), "CBB42/43 hp differ beyond seed"
    h = [f"seed={SEED}" if x.startswith("seed=") else x for x in ref["hp"]]
    assert h.count(f"seed={SEED}") == 1 and strip(h) == strip(ref["hp"])
    assert W.sha(ROOT / "methods/mobihet_sim.py") == ref["overlay_sha256"]["methods/mobihet_sim.py"]
    builder = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_sim", "--out", f"/kaggle/working/{SLUG}", "--width", "1",
                       "--rounds", "50", "--until", "50", "--save-every", "5", "--compile", "--max-hours", "BUDGET",
                       "--cache", "", "--set", *h])
    subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", OWNER, "--slug", SLUG, "--sha", BASE, "--run", SLUG,
                    "--max-hours", str(CAP), "--args", args, "--npy-kernel", builder["ref"],
                    "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *K.OVERLAYS],
                   cwd=ROOT, check=True, capture_output=True)
    folder = ROOT / "notebooks" / SLUG
    nb, meta = folder / (SLUG + ".ipynb"), folder / "kernel-metadata.json"
    doc = json.loads(nb.read_text())
    for c in doc["cells"]:
        if "BEGIN INLINE WANDB CREDENTIAL" in "".join(c["source"]):
            c["source"] = ['print("W&B off: CBB s41 r50 (as CBB42/43 r50)")\n']
        if c["cell_type"] == "code":
            compile("".join(c["source"]), str(nb), "exec")
    nb.write_text(json.dumps(doc, indent=1) + "\n")
    m = json.loads(meta.read_text())
    assert m["is_private"] and m["id"] == f"{OWNER}/{SLUG}" and m["kernel_sources"] == [builder["ref"]]
    return dict(arm="CBB41", seed=SEED, km=True, method="mobihet_sim", account=OWNER, slug=SLUG, version=1, hp=h,
                knobs=ref["knobs"], base_sha=BASE, overlay_sha256={p: W.sha(ROOT / p) for p in K.OVERLAYS},
                notebook_sha256=W.sha(nb), metadata_sha256=W.sha(meta), cap_hours=CAP, endpoint=50, lr_schedule_rounds=50,
                control="CBB42/CBB43 r50", npy_kernel=builder["ref"], npy_manifest_sha256=builder["manifest_sha256"])


def audit():
    """Exact rounds 1..50, finite, skip <= 1 %, CM support == CBB42 per round, 10 metrics recomputed from CM == log,
    LR per round == CBB42, launched hp/overlay; then r50 10-metric table for seeds 41/42/43 and their mean."""
    import numpy as np
    from core.metrics import TEN, summary
    spec = json.loads((OUT / "batch.json").read_text())[0]
    runs = {41: next(iter(ART.glob(f"{SLUG}/**/rounds.jsonl")), None)}
    for sd, k in ((42, "mh-k-cbb-s42-r50-c2/mh-k-cbb-s42-p20"), (43, "mh-k-cbb-s43-r50-c2/mh-k-cbb-s43-p20")):
        runs[sd] = ROOT / "artifacts/codec-combo50" / k / "rounds.jsonl"
    assert runs[41], "CBB41 not pulled: run collect first"
    d41, d42 = runs[41].parent, runs[42].parent
    rows = [json.loads(x) for x in runs[41].read_text().splitlines() if x.strip()]
    ref = [json.loads(x) for x in runs[42].read_text().splitlines() if x.strip()]
    cfg = json.loads((d41 / "config.json").read_text())
    chk = {"rounds_1_50": [r["round"] for r in rows] == list(range(1, 51)),
           "seed_41": cfg["hp"].get("seed") == 41, "n_params": cfg.get("n_params") == 395408,
           "lr_eq_cbb42": all(abs(a["lr"] - b["lr"]) < 1e-12 for a, b in zip(rows, ref)),
           "skip_le_1pct": all(r["skipped_frac"] <= .01 for r in rows),
           "finite_loss": all(np.isfinite(r["train_loss"]) for r in rows)}
    for kv in spec["hp"]:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        chk.setdefault("hp_eq_launched", True)
        chk["hp_eq_launched"] &= cfg["hp"].get(k) == v
    man = next(iter(ART.glob(f"{SLUG}/**/source_manifest.json")), None)
    chk["overlay_sha"] = bool(man) and spec["overlay_sha256"]["methods/mobihet_sim.py"] in man.read_text()
    cm_ok = metric_ok = True
    for r in rows:
        with np.load(d41 / f"cms_r{r['round']:03d}.npz") as z, np.load(d42 / f"cms_r{r['round']:03d}.npz") as y:
            cm, cmr = z["private"], y["private"]
            cm_ok &= bool(np.array_equal(cm.sum(2), cmr.sum(2)) and np.array_equal(z["ids"], y["ids"]))
            ten = summary(cm)
            metric_ok &= all(abs(ten[k] - r["private/" + k]) < 1e-9 for k in TEN)
    chk.update(cm_support_eq_cbb42=cm_ok, ten_metrics_from_cm=metric_ok)
    chk["pass"] = all(bool(v) for v in chk.values())
    table = {}
    for sd, f in runs.items():
        with np.load(f.parent / "cms_r050.npz") as z:
            table[sd] = summary(z["private"])
    lines = ["| seed | " + " | ".join(TEN) + " |", "|---" * (len(TEN) + 1) + "|"]
    for sd in (41, 42, 43):
        lines.append(f"| {sd} | " + " | ".join(f"{table[sd][k]:.4f}" for k in TEN) + " |")
    m = {k: float(np.mean([table[sd][k] for sd in table])) for k in TEN}
    sdv = {k: float(np.std([table[sd][k] for sd in table], ddof=1)) for k in TEN}
    lines.append("| mean ± sd | " + " | ".join(f"{m[k]:.4f} ± {sdv[k]:.4f}" for k in TEN) + " |")
    atomic_json(OUT / "audit.json", {"checks": chk, "r50": {str(k): v for k, v in table.items()}, "mean": m, "sd": sdv})
    (OUT / "report.md").write_text("CBB r50, own-test pooled CM (audit " + ("PASS" if chk["pass"] else "FAIL") + ")\n\n"
                                   + "\n".join(lines) + "\n")
    print(json.dumps(chk), "\n" + "\n".join(lines))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "launch":
        if not (OUT / "batch.json").exists():
            atomic_json(OUT / "batch.json", [row()])
        W.launch(json.loads((OUT / "batch.json").read_text()))
    elif sys.argv[1] == "collect":
        spec = json.loads((OUT / "batch.json").read_text())[0]
        st = query(spec)
        atomic_json(OUT / "status.json", [st])
        print(spec["slug"], st.get("status"), flush=True)
        if st.get("terminal") and not (ART / SLUG / "pulled.ok").exists() and W.collect(spec) == 0:
            (ART / SLUG / "pulled.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    elif sys.argv[1] == "audit":
        audit()


if __name__ == "__main__":
    main()
