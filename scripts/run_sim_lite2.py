"""Lite2 sim batch (rebuild 16/11/2026): bootstrap v2 (km2) and 8-row dose vs T3; fold train -> fold val, 200 vehicles,
cosine20, DAGSNet W1, seed 42.

    python scripts/run_sim_lite2.py quota      # scoped quota inventory -> runs/sim-kaggle-lite2/quota.json
    python scripts/run_sim_lite2.py launch     # lock batch.json, generate notebooks, push (idempotent markers)
    python scripts/run_sim_lite2.py collect    # pull finished outputs into artifacts/sim-kaggle-lite2/<slug>
    python scripts/run_sim_lite2.py gate       # after scripts/audit_sim_batch.py lite2: locked rule -> verdict-r20.json

Rule locked before push: docs/research/screening-lite2.md. Never reads the official test (scripts/sim_fold_val.py).
"""
import concurrent.futures
import glob
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

OUT, ART = ROOT / "runs/sim-kaggle-lite2", ROOT / "artifacts/sim-kaggle-lite2"
CAP = 2.75
OVERLAYS = ["methods/mobihet_compute.py", "methods/mobihet_lite.py"]
HP = (ROOT / "runs/lite-byte-probe/t3_sim_hp.txt").read_text().split()
KM2 = ["cop_kmeans_lite=0", "km_bytes=false", "km2=true"]
B = KM2 + ["cop_kmeans=6", "cop_bins=64", "km2_frac=0.25", "km2_tau=16", "km2_bits=6"]
A = KM2 + ["cop_kmeans=8", "cop_bins=64", "km2_frac=0.25", "km2_bits=5", "km2_mean8=true"]
A32, A24 = A + ["km2_tau=32"], A + ["km2_tau=24"]
D8, U1 = ["synth_tier_rows=[[64,64],[32,16],[16,8]]"], ["synth_tier_rows=[[64,32],[32,16],[16,16]]"]
ARMS = [("S0", "t3", "mobihet_compute", []), ("S1", "b", "mobihet_lite", B), ("S2", "a32", "mobihet_lite", A32),
        ("S3", "t3-d8", "mobihet_compute", D8), ("S4", "b-d8", "mobihet_lite", B + D8), ("S5", "a32-d8", "mobihet_lite", A32 + D8),
        ("S6", "t3-u1", "mobihet_compute", U1), ("S7", "b-u1", "mobihet_lite", B + U1), ("S8", "a24", "mobihet_lite", A24),
        ("I1", "i1-bins64", "mobihet_lite", KM2 + ["cop_bins=64"]), ("I2", "i2-k6", "mobihet_lite", KM2 + ["cop_kmeans=6"]),
        ("I3", "i3-tau16-frac25", "mobihet_lite", KM2 + ["km2_frac=0.25", "km2_tau=16"])]
GATE_TB, GATE_END = -.005, -.006                                         # vs S0: mean r16-20 and r20 pooled Macro-F1
W.OUT, W.ART, W.TAG = OUT, ART, "LITE2"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'


def hp(extra):
    keys = {e.split("=", 1)[0] for e in extra}
    return [h for h in HP if h.split("=", 1)[0] not in keys] + extra


def prepare(quota):
    if (OUT / "batch.json").exists():
        return json.loads((OUT / "batch.json").read_text())
    head = W.git_head()
    builder = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
    slots = []
    for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"])):
        if q.get("inventory_error") or q.get("unknown_recent") or q.get("gpu_reserved", 0):
            continue
        slots += [q["user"]] * max(0, min(2 - len(q.get("active_recent", [])), int(q.get("gpu_left", 0) // (CAP * 1.15))))
    assert len(slots) >= len(ARMS), f"need {len(ARMS)} slots, have {len(slots)}"
    batch = []
    for (arm, tag, method, extra), owner in zip(ARMS, slots):
        slug = f"mh-sim-lite2-{tag}-s42-r20"
        h = hp(extra)
        args = shlex.join(["scripts/sim_fold_val.py", "run", "--n", "200", "--method", method, "--out", f"/kaggle/working/{slug}",
                           "--rounds", "20", "--compile", "--max-hours", "BUDGET", "--cache", "", "--set", *h])
        subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", head, "--run", slug,
                        "--max-hours", str(CAP), "--args", args, "--npy-kernel", builder["ref"],
                        "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *OVERLAYS],
                       cwd=ROOT, check=True, capture_output=True)
        folder = ROOT / "notebooks" / slug
        notebook, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
        doc = json.loads(notebook.read_text())
        for c in doc["cells"]:                                        # no W&B in the simulator: no credential embedded
            src = "".join(c["source"])
            if "BEGIN INLINE WANDB CREDENTIAL" in src:
                c["source"] = ['print("W&B off: simulator sweep")\n']
            if c["cell_type"] == "code":
                compile("".join(c["source"]), str(notebook), "exec")
        notebook.write_text(json.dumps(doc, indent=1) + "\n")
        metadata = json.loads(meta.read_text())
        assert metadata["is_private"] and metadata["id"] == f"{owner}/{slug}" and metadata["kernel_sources"] == [builder["ref"]]
        assert all(metadata[k] == runtime[k] for k in ("machine_shape", "docker_image"))
        batch.append(dict(arm=arm, seed=42, method=method, account=owner, slug=slug, version=1, hp=h, base_sha=head,
                          overlay_sha256={p: W.sha(ROOT / p) for p in OVERLAYS}, notebook_sha256=W.sha(notebook),
                          metadata_sha256=W.sha(meta), cap_hours=CAP, endpoint=20, lr_schedule_rounds=20,
                          npy_kernel=builder["ref"], npy_manifest_sha256=builder["manifest_sha256"]))
    atomic_json(OUT / "batch.json", batch)
    print("LITE2_LOCKED", [(r["arm"], r["account"], r["slug"]) for r in batch], flush=True)
    return batch


def gate():
    """Locked rule on audited arms: pooled Macro-F1 on fold val vs S0 at the same rounds; guards are report-only."""
    batch = json.loads((OUT / "batch.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())["arms"]
    rows = {}
    for spec in batch:
        if not audit.get(spec["arm"], {}).get("pass"):
            continue
        f = glob.glob(str(ART / spec["slug"] / "**" / "rounds.jsonl"), recursive=True)[0]
        r = {x["round"]: x for x in map(json.loads, Path(f).read_text().splitlines()) if "private/f1_macro" in x}
        log = "".join(Path(p).read_text() for p in glob.glob(str(ART / spec["slug"] / "*.log")))
        km2 = [json.loads(line.split("KM2_BYTES ", 1)[1].split("\\n")[0].replace('\\"', '"'))
               for line in log.splitlines() if "KM2_BYTES " in line]
        rows[spec["arm"]] = dict(slug=spec["slug"], macro_r20=r[20]["private/f1_macro"],
                                 macro_tb=sum(r[i]["private/f1_macro"] for i in range(16, 21)) / 5,
                                 p10_r20=r[20]["private/client_f1_macro_p10"], f1_per_class_r20=r[20]["private/f1_per_class"],
                                 round_bytes=sum(r[i]["bytes_up"] + r[i]["bytes_down"] for i in range(1, 21)),
                                 bootstrap_bytes=km2[0]["total_mean"] if km2 else None,
                                 t_train_h=sum(r[i]["t_train"] for i in range(1, 21)) / 3600)
    assert "S0" in rows, "control S0 not audited: no verdict"
    c = rows["S0"]
    for arm, x in rows.items():
        x["d_tb"], x["d_r20"], x["d_p10"] = x["macro_tb"] - c["macro_tb"], x["macro_r20"] - c["macro_r20"], x["p10_r20"] - c["p10_r20"]
        x["verdict"] = "CONTROL" if arm == "S0" else "PASS" if x["d_tb"] >= GATE_TB and x["d_r20"] >= GATE_END else "FAIL"
        print(f'{arm:3s} {x["slug"]:36s} r20 {x["macro_r20"]:.4f} dTB {x["d_tb"]:+.4f} dr20 {x["d_r20"]:+.4f} dp10 {x["d_p10"]:+.4f} '
              f'boot {x["bootstrap_bytes"]} {x["verdict"]}', flush=True)
    atomic_json(OUT / "verdict-r20.json", {"rule": {"d_tb_r16_20_min": GATE_TB, "d_r20_min": GATE_END, "control": "S0"},
                                           "missing": [s["arm"] for s in batch if s["arm"] not in rows], "arms": rows})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "quota":
        for a in W.inventory()["accounts"]:
            print(a["user"], a.get("gpu_left"), a.get("active_recent"), a.get("inventory_error", ""))
    elif cmd == "launch":
        W.launch(prepare(json.loads((OUT / "quota.json").read_text())))
    elif cmd == "collect":
        batch = json.loads((OUT / "batch.json").read_text())
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            states = list(pool.map(query, batch))
        atomic_json(OUT / "status.json", states)
        for spec, st in zip(batch, states):
            print(spec["arm"], spec["slug"], st.get("status"), flush=True)
            if st.get("terminal") and not (ART / spec["slug"] / "pulled.ok").exists():
                rc = W.collect(spec)
                if rc == 0:
                    (ART / spec["slug"] / "pulled.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
                print("  pulled rc", rc, flush=True)
    else:
        assert cmd == "gate", cmd
        gate()


if __name__ == "__main__":
    main()
