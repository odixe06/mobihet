"""Fixed-row-budget sim batch (rebuild 17/11/2026): real rows of large vehicles replaced by synthetic rows at <= 1,01 x FedAvg
client compute, on Lite2 B + U1; fold train -> fold val, 200 vehicles, cosine20, DAGSNet W1, seed 42.

    python scripts/run_sim_budget.py cost       # client compute of every arm (full-scale vehicle sizes), no training
    python scripts/run_sim_budget.py quota      # scoped quota inventory -> runs/sim-kaggle-fb/quota.json
    python scripts/run_sim_budget.py launch     # lock batch.json, generate notebooks, push (idempotent markers)
    python scripts/run_sim_budget.py collect    # pull finished outputs into artifacts/sim-kaggle-fb/<slug>
    python scripts/run_sim_budget.py gate       # after scripts/audit_sim_batch.py fb: locked rule -> verdict-r20.json

Rule locked before push: docs/research/screening-budget.md. Controls S0 (T3) and S7 (B + U1) are the audited Lite2 sim arms
(runs/sim-kaggle-lite2). Never reads the official test (scripts/sim_fold_val.py).
"""
import concurrent.futures
import glob
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_sim_lite2 as L  # noqa: E402
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

OUT, ART = ROOT / "runs/sim-kaggle-fb", ROOT / "artifacts/sim-kaggle-fb"
LITE = ROOT / "runs/sim-kaggle-lite2"
CAP = L.CAP
OVERLAYS = L.OVERLAYS + ["methods/mobihet_budget.py"]
TIER_MIN, FLOOR, MFLOP = [4096, 16384], 16, {"dagsnet": 13.297, "bert": 40.022}     # FedAvg client MFLOP / real row / round
T3, U1 = [[64, 64], [64, 32], [48, 16]], [[64, 32], [32, 16], [16, 16]]
# arm: (tag, synthetic rows per tier [early, late], real rows per step per tier, extra hp, role)
ARMS = {"F0": ("b-u1", U1, [64, 64, 64], [], "rerun"),
        "R48": ("r48", U1, [64, 48, 48], [], "isolate"),
        "N1": ("n1-48-8", [[64, 32], [8, 8], [8, 8]], [64, 48, 48], [], "candidate"),
        "N2": ("n2-40-16", [[64, 32], [16, 16], [16, 16]], [64, 40, 40], [], "candidate"),
        "N3": ("n3-45-8", [[64, 64], [8, 8], [8, 8]], [64, 45, 45], [], "candidate"),
        "N4": ("n4-37-16", [[64, 64], [16, 16], [16, 16]], [64, 37, 37], [], "candidate"),
        "Z0": ("z0-nosynth", U1, [64, 64, 64], ["synth_off=true", "synth_step_floor=0"], "reference"),
        "ZK": ("zk-nosynth-qreal", U1, [64, 64, 64], ["synth_off=true", "synth_step_floor=0", "q_real=true"], "reference")}
GATE_TB, GATE_END, GATE_COST = -.007, -.008, 1.01                        # vs S0: mean r16-20 and r20 pooled Macro-F1; x FedAvg
W.OUT, W.ART, W.TAG = OUT, ART, "FB"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'


def scoped_cli(owner):
    return [sys.executable, str(ROOT / "scripts/kaggle_as_bounded.py"), owner, "--",
            sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]


W.cli = scoped_cli


def cost(sizes, synth, real, synth_off=False):
    """Client rows through the model per real row, one round, mean over a schedule with half the rounds late: every row a
    vehicle trains on (real kept + synthetic kept, floor steps included) / its real rows. FedAvg = 1. Returns total and
    the share of each tier."""
    n = np.asarray(sizes, dtype=np.int64)
    tier = np.searchsorted(TIER_MIN, n, side="right")
    r = np.asarray(real)[tier]
    rows = (n // 64) * r + np.minimum(n % 64, r)
    if not synth_off:
        rows = rows + np.maximum(-(-n // 64), FLOOR) * np.asarray(synth, dtype=float).mean(1)[tier]
    return float(rows.sum() / n.sum()), [float(rows[tier == t].sum() / n.sum()) for t in range(3)]


def arm_hp(synth, real, extra):
    h = L.B + [f"synth_tier_rows={json.dumps(synth, separators=(',', ':'))}"]
    if real != [64, 64, 64]:
        h.append(f"real_tier_rows={json.dumps(real, separators=(',', ':'))}")
    return L.hp(h + extra)


def costs():
    sizes = np.load(ROOT / "cache/train_sizes.npy")
    out = {"T3": cost(sizes, T3, [64, 64, 64])}
    out.update({a: cost(sizes, s, r, "synth_off=true" in e) for a, (_, s, r, e, _) in ARMS.items()})
    return out


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
    c = costs()
    batch = []
    for (arm, (tag, synth, real, extra, role)), owner in zip(ARMS.items(), slots):
        slug = f"mh-sim-fb-{tag}-s42-r20"
        h = arm_hp(synth, real, extra)
        args = shlex.join(["scripts/sim_fold_val.py", "run", "--n", "200", "--method", "mobihet_budget", "--out",
                           f"/kaggle/working/{slug}", "--rounds", "20", "--compile", "--max-hours", "BUDGET", "--cache", "",
                           "--set", *h])
        subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", head, "--run", slug,
                        "--max-hours", str(CAP), "--args", args, "--npy-kernel", builder["ref"],
                        "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *OVERLAYS],
                       cwd=ROOT, check=True, capture_output=True)
        folder = ROOT / "notebooks" / slug
        notebook, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
        doc = json.loads(notebook.read_text())
        for cell in doc["cells"]:                                     # no W&B in the simulator: no credential embedded
            src = "".join(cell["source"])
            if "BEGIN INLINE WANDB CREDENTIAL" in src:
                cell["source"] = ['print("W&B off: simulator sweep")\n']
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(notebook), "exec")
        notebook.write_text(json.dumps(doc, indent=1) + "\n")
        metadata = json.loads(meta.read_text())
        assert metadata["is_private"] and metadata["id"] == f"{owner}/{slug}" and metadata["kernel_sources"] == [builder["ref"]]
        assert all(metadata[k] == runtime[k] for k in ("machine_shape", "docker_image"))
        batch.append(dict(arm=arm, role=role, seed=42, method="mobihet_budget", account=owner, slug=slug, version=1, hp=h,
                          base_sha=head, overlay_sha256={p: W.sha(ROOT / p) for p in OVERLAYS},
                          notebook_sha256=W.sha(notebook), metadata_sha256=W.sha(meta), cap_hours=CAP, endpoint=20,
                          lr_schedule_rounds=20, cost_x_fedavg=c[arm][0], npy_kernel=builder["ref"],
                          npy_manifest_sha256=builder["manifest_sha256"]))
    atomic_json(OUT / "batch.json", batch)
    print("FB_LOCKED", [(r["arm"], r["account"], r["slug"]) for r in batch], flush=True)
    return batch


def rounds(folder):
    f = glob.glob(str(folder / "**" / "rounds.jsonl"), recursive=True)[0]
    r = {x["round"]: x for x in map(json.loads, Path(f).read_text().splitlines()) if "private/f1_macro" in x}
    return dict(macro_r20=r[20]["private/f1_macro"], macro_tb=sum(r[i]["private/f1_macro"] for i in range(16, 21)) / 5,
                p10_r20=r[20]["private/client_f1_macro_p10"], f1_per_class_r20=r[20]["private/f1_per_class"],
                round_bytes=sum(r[i]["bytes_up"] + r[i]["bytes_down"] for i in range(1, 21)),
                real_frac=r[20].get("real_frac", 1.0), t_train_h=sum(r[i]["t_train"] for i in range(1, 21)) / 3600)


def gate():
    """Locked rule on audited arms: pooled Macro-F1 on fold val vs the audited Lite2 S0 at the same rounds, and client
    compute <= GATE_COST for candidates; rerun/isolate/reference arms and every guard are report-only."""
    lite = {s["arm"]: s for s in json.loads((LITE / "batch.json").read_text())}
    assert all(json.loads((LITE / "audit.json").read_text())["arms"][a]["pass"] for a in ("S0", "S7"))
    rows = {a: dict(rounds(ROOT / "artifacts/sim-kaggle-lite2" / lite[a]["slug"]), slug=lite[a]["slug"], role="control")
            for a in ("S0", "S7")}
    batch = json.loads((OUT / "batch.json").read_text())
    audit = json.loads((OUT / "audit.json").read_text())["arms"]
    for spec in batch:
        if audit.get(spec["arm"], {}).get("pass"):
            rows[spec["arm"]] = dict(rounds(ART / spec["slug"]), slug=spec["slug"], role=spec["role"],
                                     cost=spec["cost_x_fedavg"])
    c = rows["S0"]
    for arm, x in rows.items():
        x["d_tb"], x["d_r20"], x["d_p10"] = x["macro_tb"] - c["macro_tb"], x["macro_r20"] - c["macro_r20"], x["p10_r20"] - c["p10_r20"]
        ok = x["d_tb"] >= GATE_TB and x["d_r20"] >= GATE_END
        x["verdict"] = x["role"].upper() if x["role"] != "candidate" else \
            "PASS" if ok and x["cost"] <= GATE_COST else "FAIL"
        print(f'{arm:3s} {x["slug"]:36s} r20 {x["macro_r20"]:.4f} dTB {x["d_tb"]:+.4f} dr20 {x["d_r20"]:+.4f} dp10 {x["d_p10"]:+.4f} '
              f'cost {x.get("cost", float("nan")):.4f} real {x["real_frac"]:.3f} {x["verdict"]}', flush=True)
    passed = sorted((a for a, x in rows.items() if x["verdict"] == "PASS"), key=lambda a: -rows[a]["macro_tb"])
    atomic_json(OUT / "verdict-r20.json", {"rule": {"d_tb_r16_20_min": GATE_TB, "d_r20_min": GATE_END, "cost_max": GATE_COST,
                                                    "control": "S0 (Lite2 sim)", "select": "PASS candidate, highest macro TB"},
                                           "missing": [s["arm"] for s in batch if s["arm"] not in rows],
                                           "selected": passed[0] if passed else None, "arms": rows})
    print("SELECTED", passed[0] if passed else None, flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "cost":
        for arm, (tot, tiers) in costs().items():
            print(f'{arm:4s} {tot:.4f}x  S/M/L {tiers[0]:.4f} {tiers[1]:.4f} {tiers[2]:.4f}  '
                  f'DAGSNet {tot * MFLOP["dagsnet"]:.3f}  BERT {tot * MFLOP["bert"]:.3f} MFLOP/row/round', flush=True)
    elif cmd == "quota":
        ka = W.ka

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
