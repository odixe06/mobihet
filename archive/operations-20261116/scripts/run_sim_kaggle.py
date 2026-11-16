"""Fold-train -> fold-val simulator sweep on Kaggle (200 vehicles, cosine20, seed 42 + Core seed-43 noise control).

    python scripts/run_sim_kaggle.py quota      # scoped quota inventory -> runs/sim-kaggle/quota.json
    python scripts/run_sim_kaggle.py launch     # lock batch.json, generate notebooks, push (idempotent markers)
    python scripts/run_sim_kaggle.py collect    # pull finished outputs into artifacts/sim-kaggle/<slug>

Never reads the official test: the sim driver maps test -> fold val (scripts/sim_fold_val.py). Rule: docs/research/screening-sim-foldval.md.
"""
import concurrent.futures
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

import os
BATCH = os.environ.get("SIM_BATCH", "b1")
OUT, ART = ROOT / ("runs/sim-kaggle" + ("" if BATCH == "b1" else "-" + BATCH)), ROOT / ("artifacts/sim-kaggle" + ("" if BATCH == "b1" else "-" + BATCH))
CAP = 1.25 if BATCH == "b1" else 2.25
OVERLAYS = ["methods/mobihet_sim.py", "scripts/sim_fold_val.py"]
METHOD = "mobihet_sim"
HP = (ROOT / "runs/sim/core_hp.txt").read_text().split()
ARMS = [("S0", "core", 42, []), ("S1", "core", 43, []),
        ("A256", "need256", 42, ["synth_need_T=256"]), ("A1024", "need1024", 42, ["synth_need_T=1024"]),
        ("A4096", "need4096", 42, ["synth_need_T=4096"]), ("CF", "cf", 42, ["synth_cf=true"])]
if BATCH == "b2":   # 08/11: synthetic weight (1 epoch kept) + oracle structure diagnostics
    ARMS = [("S0", "b2core", 42, []), ("S1", "b2core", 43, []),
            ("L2", "w2", 42, ["synth_weight=2"]), ("L4", "w4", 42, ["synth_weight=4"]),
            ("L2big", "w2big", 42, ["synth_weight=2", "synth_weight_min_rows=1024"]),
            ("L4big", "w4big", 42, ["synth_weight=4", "synth_weight_min_rows=1024"]),
            ("O1", "oracle", 42, ["synth=oracle"]), ("O2", "oracle-feat", 42, ["synth=oracle", "oracle_shuffle=feature"]),
            ("O3", "oracle-block", 42, ["synth=oracle", "oracle_shuffle=block"])]
if BATCH == "b3":   # 08/11: KM-copula generator (mixture in normal-score space); control = b2 S0/S1 (same sim, code path)
    ARMS = [("KM8", "km8", 42, ["cop_kmeans=8"]), ("KM16", "km16", 42, ["cop_kmeans=16"]),
            ("KM32", "km32", 42, ["cop_kmeans=32"])]
if BATCH == "b4":   # 08/11: R1 mixed-architecture cell (width per rank, vehicles split by rank); W1 homogeneous = b2 S0
    ARMS = [("M1", "mix-w1-w025", 42, [], ["--widths", "1,0.25"]), ("M2", "mix-w025-w1", 42, [], ["--widths", "0.25,1"]),
            ("H25", "homog-w025", 42, [], ["--widths", "0.25,0.25"])]
if BATCH == "b5":   # 08/11: KM16 robustness (seed 43 vs b2 S1, Lloyd 8, weaker shrinkage) and KM16 in the R1 mixed cell (vs b4 M1)
    ARMS = [("KS43", "km16", 43, ["cop_kmeans=16"]), ("KI8", "km16-it8", 42, ["cop_kmeans=16", "cop_kmeans_iters=8"]),
            ("KW05", "km16-w05", 42, ["cop_kmeans=16", "cop_kmeans_w0=0.5"]),
            ("KR1", "km16-mix", 42, ["cop_kmeans=16"], ["--widths", "1,0.25"])]
if BATCH == "b6":   # 08/11: KM16 shrinkage w0=0.5 replication (seed 43 vs b2 S1), w0=0.1, K24, compact covariances (bytes)
    ARMS = [("KW05S43", "km16-w05", 43, ["cop_kmeans=16", "cop_kmeans_w0=0.5"]),
            ("KW01", "km16-w01", 42, ["cop_kmeans=16", "cop_kmeans_w0=0.1"]),
            ("K24W05", "km24-w05", 42, ["cop_kmeans=24", "cop_kmeans_w0=0.5"]),
            ("KWI", "km16-within-w05", 42, ["cop_kmeans=16", "cop_kmeans_cov=within", "cop_kmeans_w0=0.5"]),
            ("KWI0", "km16-within-w0", 42, ["cop_kmeans=16", "cop_kmeans_cov=within", "cop_kmeans_w0=0"]),
            ("KDG", "km16-diag-w05", 42, ["cop_kmeans=16", "cop_kmeans_cov=diag", "cop_kmeans_w0=0.5"])]
if BATCH == "b7":   # 08/11: bytes - cluster covariance only inside the top-R class eigenvectors; K8 with w0=0.5
    ARMS = [(f"KP{r}", f"km16-pca{r}-w05", 42, ["cop_kmeans=16", f"cop_kmeans_cov=pca{r}", "cop_kmeans_w0=0.5"]) for r in (8, 16, 32, 48)]
    ARMS += [("K8W05", "km8-w05", 42, ["cop_kmeans=8", "cop_kmeans_w0=0.5"])]
if BATCH == "b8":   # 08/11: bytes - per-vehicle quantised upload of KM16 w0=0.5 full moments (fp16 1,74 MB / int8 0,88 MB)
    ARMS = [("KQ16", "km16-w05-fp16", 42, ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=fp16"]),
            ("KQ8", "km16-w05-int8", 42, ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8"]),
            ("KQ8S43", "km16-w05-int8", 43, ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8"]),
            ("K8Q8", "km8-w05-int8", 42, ["cop_kmeans=8", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8"])]
if BATCH == "b9":   # 09/11: KM-lite codec (bytes) + generator statistics from all 5343 vehicles (K32, 10 % participation)
    KQ = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8"]
    POP = KQ[1:] + ["cop_kmeans_lite=6", "cop_pop=-1"]
    ARMS = [("LT8", "km16-lite8", 42, KQ + ["cop_kmeans_lite=8"]), ("LT6", "km16-lite6", 42, KQ + ["cop_kmeans_lite=6"]),
            ("LT6S43", "km16-lite6", 43, KQ + ["cop_kmeans_lite=6"]),
            ("PK16", "km16-lite6-pop", 42, ["cop_kmeans=16"] + POP), ("PK32", "km32-lite6-pop", 42, ["cop_kmeans=32"] + POP),
            ("PP10", "km16-lite6-pop-p10", 42, ["cop_kmeans=16"] + POP + ["cop_kmeans_part=0.1"])]
    CAP = 2.75
if BATCH == "b10":  # 09/11: per-module candidates on KQ8 (EM, whitened k-means, benign share, cluster temperature, gap-fill, synthetic Q)
    KQ = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8"]
    ARMS = [("E1", "km16q8-em1", 42, KQ + ["cop_kmeans_em=1"]), ("E1S43", "km16q8-em1", 43, KQ + ["cop_kmeans_em=1"]),
            ("E2", "km16q8-em2", 42, KQ + ["cop_kmeans_em=2"]), ("WH", "km16q8-white", 42, KQ + ["cop_kmeans_white=true"]),
            ("SB", "km16q8-benign19", 42, KQ + ["synth_benign=19"]), ("ST", "km16q8-temp05", 42, KQ + ["cop_kmeans_temp=0.5"]),
            ("GF", "km16q8-gapfill", 42, KQ + ["synth_gapfill=true"]), ("QS", "km16q8-qsynth", 42, KQ + ["q_source=synth"])]
    CAP = 2.75
if BATCH == "b11":  # 09/11: non-generator modules on LT6 (local-prior decision, weight average, cluster Q, soft Q); both seeds
    LT = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "cop_kmeans_lite=6"]
    ARMS = []
    for s, sfx in ((42, ""), (43, "S43")):
        ARMS += [("C" + sfx, "km16-lite6-ph", s, LT + ["ph_grid=true"]), ("EM5" + sfx, "km16-lite6-ema05", s, LT + ["w_ema=0.5"]),
                 ("EM8" + sfx, "km16-lite6-ema08", s, LT + ["w_ema=0.8"]), ("QK" + sfx, "km16-lite6-qk", s, LT + ["q_cluster=true"]),
                 ("QT2" + sfx, "km16-lite6-qt2", s, LT + ["q_temp=2"])]
    CAP = 2.75
if BATCH == "b12":  # 09/11: direction a (session decision fusion, eval views on every arm) + direction b (compute dose grid)
    LT = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "cop_kmeans_lite=6", "fuse_sessions=cache/sessions/sim_val200.npz"]
    OVERLAYS = OVERLAYS + ["cache/sessions/sim_val200.npz"]
    ARMS = []
    for s, sfx in ((42, ""), (43, "S43")):
        ARMS += [("C" + sfx, "km16-lite6-fuse", s, LT)]
        ARMS += [(f"F{f}" + sfx, f"km16-lite6-floor{f}", s, LT + [f"synth_step_floor={f}"]) for f in (24, 32, 48, 64)]
        ARMS += [(f"R{a}" + sfx, f"km16-lite6-syn{a}", s, LT + [f"synth_rows_sim={a}"]) for a in (80, 96, 112, 128)]
    CAP = 2.75
if BATCH == "b13":  # CBB: aggregation, divergence, real CE regularization; one change per arm, two seeds
    METHOD = "mobihet_refine"
    OVERLAYS += ["methods/mobihet_refine.py"]
    cbb = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "cop_kmeans_lite=6", "q_bits=8", "km_bytes=true"]
    cells = [("C", []), ("SQ", ["refine_agg=sqrt_count"]), ("RK", ["refine_loss=reverse"]),
             ("SK", ["refine_loss=symmetric"]), ("JS", ["refine_loss=js4"]), ("LS", ["real_smoothing=0.05"])]
    ARMS = [(arm + ("S43" if seed == 43 else ""), "b13-" + arm.lower(), seed, cbb + extra)
            for seed in (42, 43) for arm, extra in cells]
    CAP = 2.75
if BATCH == "b14":  # new uplink statistic vs matched-fp16 aggregation; independent same-class interpolation
    METHOD = "mobihet_uplink"
    OVERLAYS += ["methods/mobihet_refine.py", "methods/mobihet_uplink.py"]
    cbb = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "cop_kmeans_lite=6", "q_bits=8", "km_bytes=true"]
    cells = [("C", []), ("MX", ["same_class_mix=0.5"]),
             ("GP", ["uplink_mode=gp16", "uplink_audit=true"]), ("LG", ["uplink_mode=lg16", "uplink_audit=true"])]
    ARMS = [(arm + ("S43" if seed == 43 else ""), "b14-" + arm.lower(), seed, cbb + extra)
            for seed in (42, 43) for arm, extra in cells]
    CAP = 2.75
if BATCH == "b15":  # client-local SAM on unchanged CBB; one fixed rho, two seeds
    METHOD = "mobihet_sam"
    OVERLAYS += ["methods/mobihet_refine.py", "methods/mobihet_sam.py"]
    cbb = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "cop_kmeans_lite=6", "q_bits=8", "km_bytes=true"]
    cells = [("C", ["sam_rho=0"]), ("SAM", ["sam_rho=0.05"])]
    ARMS = [(arm + ("S43" if seed == 43 else ""), "b15-" + arm.lower(), seed, cbb + extra)
            for seed in (42, 43) for arm, extra in cells]
    CAP = 3.75
if BATCH == "b16":  # class-mean student KL; half row-KL isolates one reduced-dose comparator
    METHOD = "mobihet_mean"
    OVERLAYS += ["methods/mobihet_refine.py", "methods/mobihet_mean.py"]
    cbb = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "cop_kmeans_lite=6", "q_bits=8", "km_bytes=true"]
    cells = [("C", ["mean_kd=row"]), ("MK", ["mean_kd=mean"]), ("HK", ["mean_kd=half"])]
    ARMS = [(arm + ("S43" if seed == 43 else ""), "b16-" + arm.lower(), seed, cbb + extra)
            for seed in (42, 43) for arm, extra in cells]
    CAP = 2.75
if BATCH == "b17":  # owner macro-first: prior coordinate and real NCKD, independently on CBB
    METHOD = "mobihet_macro"
    OVERLAYS += ["methods/mobihet_refine.py", "methods/mobihet_macro.py"]
    cbb = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "cop_kmeans_lite=6", "q_bits=8", "km_bytes=true"]
    cells = [("C", ["macro_mode=cbb"]), ("P", ["macro_mode=prior"]), ("NR", ["macro_mode=real_nckd"])]
    ARMS = [(arm + ("S43" if seed == 43 else ""), "b17-" + arm.lower(), seed, cbb + extra)
            for seed in (42, 43) for arm, extra in cells]
    CAP = 2.75
if BATCH == "b20":  # client compute on CBB: head-only / mixed synthetic gradient, fewer late synthetic rows
    METHOD = "mobihet_compute"
    OVERLAYS += ["methods/mobihet_refine.py", "methods/mobihet_compute.py"]
    cbb = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "cop_kmeans_lite=6", "q_bits=8", "km_bytes=true"]
    cells = [("C", ["synth_grad=full"]), ("SH", ["synth_grad=head"]), ("SM", ["synth_grad=mix16"]),
             ("SD", ["synth_rows_late=32", "synth_late_from=10"])]
    ARMS = [(arm + ("S43" if seed == 43 else ""), "b20-" + arm.lower(), seed, cbb + extra)
            for seed in (42, 43) for arm, extra in cells]
    CAP = 2.75
if BATCH == "b21":  # synthetic rows by vehicle size tier x schedule phase on CBB (+ one floor-48 reallocation arm)
    METHOD = "mobihet_compute"
    OVERLAYS += ["methods/mobihet_refine.py", "methods/mobihet_compute.py"]
    cbb = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8", "cop_kmeans_lite=6", "q_bits=8", "km_bytes=true"]
    tier = lambda rows: ["synth_tier_min=[4096,16384]", "synth_tier_rows=" + rows, "synth_late_from=10"]
    t4 = tier("[[64,64],[48,32],[32,16]]")
    cells = [("C", ["synth_grad=full"]), ("T1", tier("[[64,64],[64,32],[64,32]]")), ("T3", tier("[[64,64],[64,32],[48,16]]")),
             ("T4", t4), ("T5", tier("[[64,64],[32,16],[16,16]]")), ("T4F", t4 + ["synth_step_floor=48"])]
    ARMS = [(arm + ("S43" if seed == 43 else ""), "b21-" + arm.lower(), seed, cbb + extra)
            for seed in (42, 43) for arm, extra in cells]
    CAP = 2.75
W.OUT, W.ART, W.TAG = OUT, ART, "SIM"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|arch_rank\d\.json)$)'
if BATCH == "b14":
    W.COLLECT += r'|/wire/rank[01]_r\d{3}\.npz$'


def hp(seed, extra):
    keys = {e.split("=", 1)[0] for e in extra} | {"seed"}
    return [h for h in HP if h.split("=", 1)[0] not in keys] + [f"seed={seed}", *extra]


def prepare(quota):
    if (OUT / "batch.json").exists():
        return json.loads((OUT / "batch.json").read_text())
    head = W.git_head()
    builder = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    slots = []
    for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"])):
        if q.get("inventory_error") or q.get("unknown_recent") or q.get("gpu_reserved", 0):
            continue
        free = 2 - len(q.get("active_recent", []))
        n = min(free, int(q.get("gpu_left", 0) // (CAP * 1.15)))
        slots += [q["user"]] * max(0, n)
    assert len(slots) >= len(ARMS), f"need {len(ARMS)} slots, have {len(slots)}"
    batch = []
    for spec_arm, owner in zip(ARMS, slots):
        arm, tag, seed, extra = spec_arm[:4]
        cli_extra = list(spec_arm[4]) if len(spec_arm) > 4 else []
        slug = f"mh-sim-{tag}-s{seed}-r20"
        h = hp(seed, extra)
        args = shlex.join(["scripts/sim_fold_val.py", "run", "--n", "200", *cli_extra, "--method", METHOD, "--out",
                           f"/kaggle/working/{slug}", "--rounds", "20", "--compile", "--max-hours", "BUDGET", "--cache", "",
                           "--set", *h])
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
        runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
        assert metadata["is_private"] and metadata["id"] == f"{owner}/{slug}" and metadata["kernel_sources"] == [builder["ref"]]
        assert all(metadata[k] == runtime[k] for k in ("machine_shape", "docker_image"))
        batch.append(dict(arm=arm, seed=seed, method=METHOD, account=owner, slug=slug, version=1, hp=h, base_sha=head,
                          overlay_sha256={p: W.sha(ROOT / p) for p in OVERLAYS}, notebook_sha256=W.sha(notebook),
                          metadata_sha256=W.sha(meta), cap_hours=CAP, endpoint=20, lr_schedule_rounds=20,
                          npy_kernel=builder["ref"], npy_manifest_sha256=builder["manifest_sha256"]))
    atomic_json(OUT / "batch.json", batch)
    print("SIM_LOCKED", [(r["arm"], r["account"], r["slug"]) for r in batch], flush=True)
    return batch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "quota":
        q = W.inventory()
        for a in q["accounts"]:
            print(a["user"], a.get("gpu_left"), a.get("active_recent"), a.get("inventory_error", ""))
        return
    if cmd == "launch":
        batch = prepare(json.loads((OUT / "quota.json").read_text()))
        W.launch(batch)
        return
    if cmd == "collect":
        batch = json.loads((OUT / "batch.json").read_text())
        (OUT / "collection").mkdir(exist_ok=True)
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


if __name__ == "__main__":
    main()
