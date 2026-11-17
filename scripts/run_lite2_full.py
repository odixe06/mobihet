"""Lite2 full-scale ladder (docs/research/screening-lite2.md): pilot r1-20 cosine50, DAGSNet W1 seed 42, arm chosen by the sim rule;
then stage r50 = the PASS pilot resumed to round 50 on the same owner (same hp/source, cosine50).

    python scripts/run_lite2_full.py quota           # owner-scoped live quota -> runs/lite2-full/quota.json
    python scripts/run_lite2_full.py launch [r50]    # lock batch.json (T3P hp + arm knobs; r50: pilot spec + parent), push
    python scripts/run_lite2_full.py collect [r50]   # one shot: status; pull terminal kernels into artifacts/<stage>/<slug>
    python scripts/run_lite2_full.py audit [r50]     # artifact checks vs T3 seed 42 same rounds + locked verdict -> audit.json

Control = audited T3 seed 42 (pilot: artifacts/b21-full/mh-k-b21-t3p-r20; r50: its chain in artifacts/b21-chain; same core/methods
source as HEAD). Pilot verdict (locked before push): delta Macro-F1 mean r16-20 >= -.006 and delta r20 >= -.006, bootstrap <= 382 000 B
per vehicle. Full verdict (owner targets 16/11): Macro-F1 r50 > .8998, bootstrap + all rounds <= 400 000 B, compute <= 1.37x FedAvg.
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
from scripts import run_sim_lite2 as L  # noqa: E402  (arm knobs, overlays)
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

OUT, ART = ROOT / "runs/lite2-full", ROOT / "artifacts/lite2-full"
REF = ROOT / "artifacts/b21-full/mh-k-b21-t3p-r20/mh-k-b21-t3p-r20"          # T3 seed 42 r1-20, audited (runs/b21-full)
ARMS = [("BU1P", "bu1", 42, L.B + L.U1, 20, 5.5)]                             # arm, tag, seed, knobs, until, cap hours
GATE_TB, GATE_END, BOOT_MAX = -.006, -.006, 382_000
PILOT_OUT, STAGE = OUT, "pilot"
REF50 = ROOT / "artifacts/b21-chain/mh-k-b21-t3p-r20-s36/mh-k-b21-t3p-r20"    # T3 seed 42 r1-50 (r1-20 == REF), audited
CAP50, MACRO50, TOTAL50, COMPUTE50 = 7.5, .8998, 400_000, (1.364, 1.37)      # U1 analytic ratio (screening-lite2) vs owner ceiling
RESUME = """parent_states = list(Path("/kaggle/input").glob("**/{run}/state.pt"))
assert len(parent_states) == 1, "Missing or ambiguous parent state"
parent_dir = parent_states[0].parent
for rank, name in enumerate(("state.pt", "state_rank1.pt")):
    assert (parent_dir/name).is_file(), name
    state = torch.load(parent_dir/name, map_location="cpu", weights_only=True, mmap=True)
    assert state["round"] == {round} and "rng_cpu" in state and "rng_cuda" in state
    if rank == 0:
        assert state["method"] == "mobihet_lite" and state["hp"]["seed"] == {seed} and state["hp"]["rounds"] == 50
    del state
target = Path("/kaggle/working/{run}")
target.mkdir(exist_ok=True)
for item in parent_dir.iterdir():
    dest = target/item.name
    if item.is_dir(): shutil.copytree(item, dest, dirs_exist_ok=True, copy_function=shutil.copyfile)
    else: shutil.copyfile(item, dest)
print("resumed from verified two-rank checkpoint", str(parent_dir), "round", {round})
"""
W.OUT, W.ART, W.TAG = OUT, ART, "LITE2F"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'


def use(stage):
    global OUT, ART, REF, STAGE
    if stage == "r50":
        OUT, ART, REF, STAGE = ROOT / "runs/lite2-full-r50", ROOT / "artifacts/lite2-full-r50", REF50, stage
        W.OUT, W.ART, W.TAG = OUT, ART, "LITE2R50"


def rows_r50():
    """The PASS pilot resumed r21-50: same owner/hp/source, run dir = pilot slug, parent = pilot kernel output (two-rank state)."""
    p = json.loads((PILOT_OUT / "batch.json").read_text())[0]
    assert json.loads((PILOT_OUT / "audit.json").read_text())[p["arm"]]["verdict"] == "PASS", "pilot not PASS: no r50"
    assert W.git_head() == p["base_sha"] and all(W.sha(ROOT / f) == h for f, h in p["overlay_sha256"].items())
    owner, run, slug, parent = p["account"], p["slug"], p["slug"].replace("-r20", "-r50"), f'{p["account"]}/{p["slug"]}'
    args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_lite", "--out", f"/kaggle/working/{run}", "--width", "1",
                       "--rounds", "50", "--until", "50", "--save-every", "5", "--compile", "--max-hours", "BUDGET", "--cache", "",
                       "--set", *p["hp"]])
    subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", p["base_sha"], "--run", run,
                    "--max-hours", str(CAP50), "--args", args, "--npy-kernel", p["npy_kernel"], "--npy-manifest-sha256",
                    p["npy_manifest_sha256"], "--resume-kernel", parent, "--overlay", *p["overlay_sha256"]],
                   cwd=ROOT, check=True, capture_output=True)
    folder = ROOT / "notebooks" / slug
    nb, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
    doc, swapped = json.loads(nb.read_text()), 0
    for c in doc["cells"]:
        src = "".join(c["source"])
        if "BEGIN INLINE WANDB CREDENTIAL" in src:
            src = 'print("W&B off: Lite2 full-scale (as T3 pilot)")\n'
        if 'print("resumed from", prev[0])' in src:               # generator's copy2 import -> verified two-rank copyfile import
            a, b = src.index("prev = [p for p in glob.glob("), src.index('print("resumed from", prev[0])\n')
            src = src[:a] + RESUME.format(run=run, round=p["endpoint"], seed=p["seed"]) + src[b:].split("\n", 1)[1]
            swapped += 1
        c["source"] = src.splitlines(keepends=True)
        if c["cell_type"] == "code":
            compile(src, str(nb), "exec")
    assert swapped == 1
    nb.write_text(json.dumps(doc, indent=1) + "\n")
    m = json.loads(meta.read_text())
    assert m["is_private"] and m["id"] == f"{owner}/{slug}" and m["kernel_sources"] == [parent, p["npy_kernel"]]
    return [dict(p, arm="BU1F", slug=slug, run=run, parent=parent + "/1", parent_round=p["endpoint"], notebook_sha256=W.sha(nb),
                 metadata_sha256=W.sha(meta), cap_hours=CAP50, endpoint=50, control="T3 seed 42 r50 (b21-chain/mh-k-b21-t3p-r20-s36)")]


def rows(quota):
    ref = next(r for r in json.loads((ROOT / "runs/b21-full/batch.json").read_text()) if r["arm"] == "T3P")
    sim = json.loads((L.OUT / "batch.json").read_text())[0]
    assert all(W.sha(ROOT / p) == sim["overlay_sha256"][p] for p in L.OVERLAYS), "overlay differs from audited sim lite2"
    head, builder = W.git_head(), json.loads((ROOT / "runs/w-study/builder.json").read_text())
    assert head == sim["base_sha"], "HEAD differs from audited sim lite2"
    slots = [(q["user"], q.get("gpu_left", 0)) for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"]))
             if not (q.get("inventory_error") or q.get("unknown_recent") or q.get("gpu_reserved", 0) or q.get("active_recent"))]
    batch = []
    for (arm, tag, seed, knobs, until, cap), (owner, left) in zip(ARMS, slots):
        assert left >= cap * 1.15, (owner, left)
        slug = f"mh-lite2-{tag}-s{seed}-r{until}"
        keys = {k.split("=")[0] for k in knobs} | {"seed"}
        h = [x for x in ref["hp"] if x.split("=")[0] not in keys] + [f"seed={seed}", *knobs]
        args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_lite", "--out", f"/kaggle/working/{slug}",
                           "--width", "1", "--rounds", "50", "--until", str(until), "--save-every", "5", "--compile",
                           "--max-hours", "BUDGET", "--cache", "", "--set", *h])
        subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", head,
                        "--run", slug, "--max-hours", str(cap), "--args", args, "--npy-kernel", builder["ref"],
                        "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *L.OVERLAYS],
                       cwd=ROOT, check=True, capture_output=True)
        folder = ROOT / "notebooks" / slug
        nb, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
        doc = json.loads(nb.read_text())
        for c in doc["cells"]:
            if "BEGIN INLINE WANDB CREDENTIAL" in "".join(c["source"]):
                c["source"] = ['print("W&B off: Lite2 full-scale (as T3 pilot)")\n']
            if c["cell_type"] == "code":
                compile("".join(c["source"]), str(nb), "exec")
        nb.write_text(json.dumps(doc, indent=1) + "\n")
        m = json.loads(meta.read_text())
        assert m["is_private"] and m["id"] == f"{owner}/{slug}" and m["kernel_sources"] == [builder["ref"]]
        batch.append(dict(arm=arm, seed=seed, method="mobihet_lite", account=owner, slug=slug, version=1, hp=h, base_sha=head,
                          overlay_sha256={p: W.sha(ROOT / p) for p in L.OVERLAYS}, notebook_sha256=W.sha(nb),
                          metadata_sha256=W.sha(meta), cap_hours=cap, endpoint=until, lr_schedule_rounds=50,
                          control="T3P seed 42 (b21-full/mh-k-b21-t3p-r20)", npy_kernel=builder["ref"],
                          npy_manifest_sha256=builder["manifest_sha256"]))
    return batch


def audit():
    """Rounds 1..endpoint exact, finite, skip <= 1 %, hp == launched, overlay SHAs executed, CM support/ids == T3 control,
    10 metrics from CM == log, LR == control; then the locked pilot verdict."""
    from core.metrics import TEN, summary
    ref = [json.loads(x) for x in (REF / "rounds.jsonl").read_text().splitlines() if x.strip()]
    out = {}
    for spec in json.loads((OUT / "batch.json").read_text()):
        f = next(iter(ART.glob(f"{spec['slug']}/**/rounds.jsonl")), None)
        if f is None:
            out[spec["arm"]] = {"pulled": False}
            continue
        d, R = f.parent, spec["endpoint"]
        rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        cfg = json.loads((d / "config.json").read_text())
        chk = {"rounds_exact": [r["round"] for r in rs] == list(range(1, R + 1)), "n_params": cfg.get("n_params") == 395408,
               "clients": cfg.get("clients") == 5343, "method": cfg.get("method") == spec["method"],
               "finite_loss": all(np.isfinite(r["train_loss"]) for r in rs),
               "skip_le_1pct": all(r["skipped_frac"] <= .01 for r in rs),
               "clients_applied": all(r.get("clients_without_applied", 0) == 0 for r in rs),
               "lr_eq_control": all(abs(a["lr"] - b["lr"]) < 1e-12 for a, b in zip(rs, ref))}
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
        man = json.loads(man.read_text()) if man else {}
        chk["source"] = man.get("base_sha") == spec["base_sha"] and man.get("overlay") == spec["overlay_sha256"]
        macro, ctl, p10, ok_cm, ok_m = [], [], [], True, True
        for r in rs[:R]:
            with np.load(d / f"cms_r{r['round']:03d}.npz") as z, np.load(REF / f"cms_r{r['round']:03d}.npz") as y:
                ok_cm &= bool(np.array_equal(z["private"].sum(2), y["private"].sum(2)) and np.array_equal(z["ids"], y["ids"]))
                ten, tr = summary(z["private"]), summary(y["private"])
                ok_m &= all(abs(ten[k] - r["private/" + k]) < 1e-9 for k in TEN)
                macro.append(ten["f1_macro"]); ctl.append(tr["f1_macro"])
                p10.append(ten["client_f1_macro_p10"] - tr["client_f1_macro_p10"])
        chk.update(cm_support_eq_control=ok_cm, ten_metrics_from_cm=ok_m)
        chk["pass"] = all(bool(v) for v in chk.values())
        delta, boot = np.array(macro) - np.array(ctl), cfg["payload_bytes"]["bootstrap"]
        res = {"checks": chk, "endpoint": ten, "control_endpoint": tr, "delta_by_round": [float(x) for x in delta],
               "delta_end": float(delta[-1]), "delta_mean_last5": float(delta[-5:].mean()), "delta_p10_end": float(p10[-1]),
               "bootstrap_bytes": boot, "round_bytes": sum(r["bytes_up"] + r["bytes_down"] for r in rs),
               "control_round_bytes": sum(r["bytes_up"] + r["bytes_down"] for r in ref[:R]),
               "t_train_hours": sum(r["t_train"] for r in rs) / 3600,
               "control_t_train_hours": sum(r["t_train"] for r in ref[:R]) / 3600}
        ok = (chk["pass"] and res["delta_mean_last5"] >= GATE_TB and res["delta_end"] >= GATE_END
              and boot["up"] + boot["down"] <= BOOT_MAX)
        if STAGE == "r50":                                        # full verdict: owner targets, not a delta vs T3
            pd = next(iter((ROOT / "artifacts/lite2-full").glob(f"{spec['run']}/**/rounds.jsonl"))).parent
            chk["prefix_eq_pilot"] = (f.read_text().startswith((pd / "rounds.jsonl").read_text()) and all(
                W.sha(d / f"cms_r{r:03d}.npz") == W.sha(pd / f"cms_r{r:03d}.npz") for r in range(1, spec["parent_round"] + 1)))
            chk["pass"] = all(bool(v) for k, v in chk.items() if k != "pass")
            res.update(total_bytes=boot["up"] + boot["down"] + res["round_bytes"], compute_ratio_analytic=COMPUTE50[0],
                       targets={"macro_gt": MACRO50, "total_bytes_le": TOTAL50, "compute_le": COMPUTE50[1]})
            ok = chk["pass"] and ten["f1_macro"] > MACRO50 and res["total_bytes"] <= TOTAL50 and COMPUTE50[0] <= COMPUTE50[1]
        res["verdict"] = "PASS" if ok else "FAIL"
        out[spec["arm"]] = res
        print(spec["arm"], "audit", "PASS" if chk["pass"] else "FAIL", {k: v for k, v in chk.items() if not v},
              "macro %.4f vs T3 %.4f dTB %+.4f dEnd %+.4f boot %.0f B" % (ten["f1_macro"], tr["f1_macro"], res["delta_mean_last5"],
                                                                         res["delta_end"], boot["up"] + boot["down"]),
              res["verdict"], flush=True)
    atomic_json(OUT / "audit.json", out)


def main():
    cmd = sys.argv[1]
    use(sys.argv[2] if len(sys.argv) > 2 else "pilot")
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "quota":
        for a in W.inventory()["accounts"]:
            print(a["user"], a.get("gpu_left"), a.get("active_recent"), a.get("inventory_error", ""))
    elif cmd == "launch":
        if not (OUT / "batch.json").exists():
            atomic_json(OUT / "batch.json", rows_r50() if STAGE == "r50" else rows(json.loads((OUT / "quota.json").read_text())))
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
    else:
        assert cmd == "audit", cmd
        audit()


if __name__ == "__main__":
    main()
