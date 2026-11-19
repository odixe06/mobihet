"""FB-L L4 full r50 on DAGSNet (owner 19/11: "chọn L4, chạy r50 cả hai backbone, thêm seed 43"; + seed 41), docs/research/screening-fbl.md.

    python scripts/run_fbl_r50.py build S42 OWNER CAP   # pilot L4P resumed r21-50 (same owner: parent = pilot kernel output)
    python scripts/run_fbl_r50.py build S43|S41 OWNER CAP  # fresh r1-50, hp = L4P hp with seed=43/41 (T3 differs only in seed)
    python scripts/run_fbl_r50.py launch | collect | audit
    python scripts/run_fbl_r50.py hop S41h2 OWNER CAP   # session k >= 2 (cross-account): parent session COMPLETE -> dataset hop
    python scripts/run_fbl_r50.py launch-hop S41h2

Spec = audited L4P pilot row (runs/fbl-full/batch.json, audit PASS): same base SHA, overlay bytes, hp, cosine50, compact engine.
Verdict locked before push (= FB full r50 gate, docs/research/screening-budget.md): Macro-F1 r50 > .8998, bootstrap + all rounds
<= 400 000 B per vehicle, executed compute <= 1.01x FedAvg (L4 measured 0.997x, compute harness), artifact audit PASS
(rounds 1..50, hp/source, skip <= 1 %, applied clients, CM support == T3 same seed, 10 metrics from CM; S42: r1-20 == pilot bytes).
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
from scripts import run_lite2_full as F  # noqa: E402  (RESUME template)
from scripts import run_sim_budget as SB  # noqa: E402
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

OUT, ART, PILOT = ROOT / "runs/fbl-r50", ROOT / "artifacts/fbl-r50", ROOT / "runs/fbl-full"
REF = {42: ROOT / "artifacts/b21-chain/mh-k-b21-t3p-r20-s36/mh-k-b21-t3p-r20",      # T3 seed 42 r1-50, audited
       43: ROOT / "artifacts/b21-chain/mh-k-b21-t343-r50-s7/mh-k-b21-t343-r50",     # T3 seed 43 r1-50 (hp == seed 42 but seed)
       41: ROOT / "artifacts/t3-s41/mh-k-b21-t341-r50/mh-k-b21-t341-r50"}           # T3 seed 41 r1-50 (hp == seed 42 but seed)
MACRO50, TOTAL50, COMPUTE50 = .8998, 400_000, (0.997, 1.01)                         # executed x FedAvg (measured L4) vs ceiling
W.OUT, W.ART, W.TAG = OUT, ART, "FBLR50"
W.cli = SB.scoped_cli
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'


def pilot():
    p = next(r for r in json.loads((PILOT / "batch.json").read_text()) if r["arm"] == "L4P")
    assert json.loads((PILOT / "audit.json").read_text())["L4P"]["verdict"] == "PASS"
    assert W.git_head() == p["base_sha"] and all(W.sha(ROOT / f) == h for f, h in p["overlay_sha256"].items())
    return p


def build(arm, owner, cap):
    p = pilot()
    seed = int(arm[1:])
    slug = f"mh-fbl-l4-top-s{seed}-r50"
    run = p["slug"] if seed == 42 else slug                       # S42 keeps the pilot run dir (state glob + r1-20 prefix)
    hp = [h for h in p["hp"] if not h.startswith("seed=")] + [f"seed={seed}"]
    assert [h for h in hp if not h.startswith("seed=")] == [h for h in p["hp"] if not h.startswith("seed=")]
    args = shlex.join(["scripts/run_q20.py", "--method", p["method"], "--out", f"/kaggle/working/{run}", "--width", "1",
                       "--rounds", "50", "--until", "50", "--save-every", "5", "--compile", "--max-hours", "BUDGET",
                       "--cache", "", "--set", *hp])
    parent = f'{p["account"]}/{p["slug"]}'
    if seed == 42:
        assert owner == p["account"], "S42 resumes the pilot kernel output: same owner"
    subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", p["base_sha"], "--run", run,
                    "--max-hours", str(cap), "--args", args, "--npy-kernel", p["npy_kernel"], "--npy-manifest-sha256",
                    p["npy_manifest_sha256"], *(["--resume-kernel", parent] if seed == 42 else []),
                    "--overlay", *p["overlay_sha256"]], cwd=ROOT, check=True, capture_output=True)
    folder = ROOT / "notebooks" / slug
    nb, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
    doc, swapped = json.loads(nb.read_text()), 0
    for c in doc["cells"]:
        src = "".join(c["source"])
        if "BEGIN INLINE WANDB CREDENTIAL" in src:
            src = 'print("W&B off: FB-L full r50 (as pilot)")\n'
        if seed == 42 and 'print("resumed from", prev[0])' in src:   # verified two-rank import of the pilot r20 state
            a, b = src.index("prev = [p for p in glob.glob("), src.index('print("resumed from", prev[0])\n')
            tpl = F.RESUME.replace('"mobihet_lite"', f'"{p["method"]}"')
            src = src[:a] + tpl.format(run=run, round=p["endpoint"], seed=42) + src[b:].split("\n", 1)[1]
            swapped += 1
        c["source"] = src.splitlines(keepends=True)
        if c["cell_type"] == "code":
            compile(src, str(nb), "exec")
    assert swapped == (1 if seed == 42 else 0)
    nb.write_text(json.dumps(doc, indent=1) + "\n")
    m = json.loads(meta.read_text())
    assert m["is_private"] and m["id"] == f"{owner}/{slug}"
    assert m["kernel_sources"] == ([parent, p["npy_kernel"]] if seed == 42 else [p["npy_kernel"]])
    row = dict(p, arm=f"L4F{seed}", seed=seed, account=owner, slug=slug, run=run, hp=hp, version=1, notebook_sha256=W.sha(nb),
               metadata_sha256=W.sha(meta), cap_hours=cap, endpoint=50, control=f"T3 seed {seed} r50 ({REF[seed].parent.name})")
    if seed == 42:
        row.update(parent=parent + "/1", parent_round=p["endpoint"])
    return row


def stage(parent, owner, slug):
    """Parent session COMPLETE -> pull its <run>/ -> private dataset <owner>/<slug>-state (files at the root; Kaggle mounts it at
    <dataset slug>/, and unpacks a single zip into the root) -> verified two-rank state round r < 50, rounds 1..r contiguous."""
    import shutil
    d = ART.parent / "fbl-hop" / parent["slug"]
    d.mkdir(parents=True, exist_ok=True)
    if (d / "ready.json").exists():
        return json.loads((d / "ready.json").read_text())
    st = query(parent)
    assert st["status"] == "COMPLETE", st                       # QUERY_ERROR = unknown: rerun later, never hop
    src, ref = d / "pull" / parent["run"], f"{owner}/{slug}-state"
    kas = [sys.executable, str(W.SKILL / "kaggle_as.py")]
    if not (d / "pulled.ok").exists():
        subprocess.run([*kas, parent["account"], "--", sys.executable, str(W.SKILL / "pull_kernel_output.py"),
                        f"{parent['account']}/{parent['slug']}/{parent['version']}", "-p", str(d / "pull"), "-j", "16",
                        "--pattern", f"^{parent['run']}/"], cwd=ROOT, check=True)
        (d / "pulled.ok").write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    import torch
    s0 = torch.load(src / "state.pt", map_location="cpu", weights_only=True, mmap=True)
    s1 = torch.load(src / "state_rank1.pt", map_location="cpu", weights_only=True, mmap=True)
    rr = s0["round"]
    done = [json.loads(x)["round"] for x in (src / "rounds.jsonl").read_text().splitlines() if x.strip()]
    assert s0["method"] == parent["method"] and s0["hp"]["rounds"] == 50 and s0["hp"]["seed"] == parent["seed"] and s1["round"] == rr
    assert done == list(range(1, rr + 1)) and rr < 50, (done, rr)
    del s0, s1
    upload, intent = d / "upload", d / "create-intent.json"
    if not intent.exists():                                     # existing intent: dataset may exist; only re-verify below
        shutil.rmtree(upload, ignore_errors=True)
        shutil.copytree(src, upload)
        (upload / "dataset-metadata.json").write_text(json.dumps(dict(id=ref, title=f"fbl hop {slug}",
                                                                      licenses=[dict(name="CC0-1.0")]), indent=1))
        intent.write_text(json.dumps(dict(ref=ref, parent=st["ref"], round=rr)))
        r = subprocess.run([*kas, owner, "--", "kaggle", "datasets", "create", "-p", str(upload), "--dir-mode", "zip"],
                           cwd=ROOT, capture_output=True, text=True)
        (d / "create-reply.txt").write_text(r.stdout + r.stderr)
        assert r.returncode == 0, "dataset create failed: inspect create-reply.txt and the owner's datasets before retry"
    run = lambda *a: subprocess.run([*kas, owner, "--", *a], cwd=ROOT, capture_output=True, text=True, timeout=300)
    for _ in range(80):
        time.sleep(15)
        r = run("kaggle", "datasets", "status", ref)
        if r.returncode == 0 and r.stdout.strip() == "ready":
            break
    else:
        raise RuntimeError(("dataset not ready; inspect, no push", ref))
    ls = run(sys.executable, str(ROOT.parent / "fl_veremi_bert/scripts/list_dataset.py"), ref).stdout
    got = dict(x.split(",") for x in ls.split())
    assert all(int(got[f]) == (upload / f).stat().st_size for f in ("state.pt", "state_rank1.pt")), ls[:300]
    res = dict(ref=ref, round=rr, parent=st["ref"], state_bytes=(upload / "state.pt").stat().st_size, listing=ls)
    atomic_json(d / "ready.json", res)
    return res


def chain_rows():
    """Main batch rows plus every locked hop session (runs/fbl-r50/<S..h..>/batch.json), in session order."""
    rows = json.loads((OUT / "batch.json").read_text())
    for x in sorted(OUT.glob("S*h*/batch.json")):
        rows += json.loads(x.read_text())
    return rows


def build_hop(arm, owner, cap):
    """Session k >= 2 of a chain (arm S<seed>h<k>): parent = session k-1, same run dir/hp/source; resume from its state."""
    seed, k = arm[1:].split("h")
    seed, k = int(seed), int(k)
    rows = chain_rows()
    parent = next(r for r in rows if r["seed"] == seed and r.get("session", 1) == k - 1)
    slug = f"mh-fbl-l4-top-s{seed}-r50-h{k}"
    hop = stage(parent, owner, slug)
    run = parent["run"]
    args = shlex.join(["scripts/run_q20.py", "--method", parent["method"], "--out", f"/kaggle/working/{run}", "--width", "1",
                       "--rounds", "50", "--until", "50", "--save-every", "5", "--compile", "--max-hours", "BUDGET",
                       "--cache", "", "--set", *parent["hp"]])
    subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", parent["base_sha"],
                    "--run", run, "--max-hours", str(cap), "--args", args, "--npy-kernel", parent["npy_kernel"],
                    "--npy-manifest-sha256", parent["npy_manifest_sha256"], "--resume-dataset", hop["ref"],
                    "--overlay", *parent["overlay_sha256"]], cwd=ROOT, check=True, capture_output=True)
    folder = ROOT / "notebooks" / slug
    nb, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
    doc, swapped = json.loads(nb.read_text()), 0
    where = hop["ref"].split("/")[1]
    for c in doc["cells"]:
        src = "".join(c["source"])
        if "BEGIN INLINE WANDB CREDENTIAL" in src:
            src = 'print("W&B off: FB-L full r50 (as pilot)")\n'
        if 'print("resumed from", prev[0])' in src:
            a, b = src.index("prev = [p for p in glob.glob("), src.index('print("resumed from", prev[0])\n')
            tpl = F.RESUME.replace('"mobihet_lite"', f'"{parent["method"]}"').replace('glob("**/{run}/state.pt")', f'glob("**/{where}/state.pt")')
            src = src[:a] + tpl.format(run=run, round=hop["round"], seed=seed) + src[b:].split("\n", 1)[1]
            swapped += 1
        c["source"] = src.splitlines(keepends=True)
        if c["cell_type"] == "code":
            compile(src, str(nb), "exec")
    assert swapped == 1
    nb.write_text(json.dumps(doc, indent=1) + "\n")
    m = json.loads(meta.read_text())
    assert m["is_private"] and m["id"] == f"{owner}/{slug}" and m["kernel_sources"] == [parent["npy_kernel"]]
    assert hop["ref"] in m["dataset_sources"]
    return dict(parent, arm=f"L4F{seed}h{k}", session=k, account=owner, slug=slug, version=1, notebook_sha256=W.sha(nb),
                metadata_sha256=W.sha(meta), cap_hours=cap, parent=hop["parent"], parent_slug=parent["slug"],
                parent_round=hop["round"], resume_dataset=hop["ref"])


def audit():
    """Rounds 1..50 exact, finite, skip <= 1 %, applied clients, hp == launched, executed overlay SHAs, CM support/ids == T3 same seed,
    10 metrics from CM == log, LR == control; S42 r1-20 bytes == pilot; then the locked absolute r50 verdict."""
    sys.path.insert(0, str(ROOT))
    from core.metrics import TEN, summary
    out, rows = {}, chain_rows()
    last = {}
    for r in rows:
        if r.get("session", 1) >= last.get(r["seed"], {}).get("session", 1):
            last[r["seed"]] = r
    for spec in last.values():
        f = next(iter(ART.glob(f"{spec['slug']}/**/{spec['run']}/rounds.jsonl")), None)
        if f is None:
            out[spec["arm"]] = {"pulled": False}
            continue
        d, R, ref_d = f.parent, spec["endpoint"], REF[spec["seed"]]
        rs = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        ref = [json.loads(x) for x in (ref_d / "rounds.jsonl").read_text().splitlines() if x.strip()]
        cfg = json.loads((d / "config.json").read_text())
        chk = {"rounds_exact": [r["round"] for r in rs] == list(range(1, R + 1)), "n_params": cfg.get("n_params") == 395408,
               "clients": cfg.get("clients") == 5343, "method": cfg.get("method") == spec["method"],
               "seed": cfg["hp"].get("seed") == spec["seed"], "finite_loss": all(np.isfinite(r["train_loss"]) for r in rs),
               "skip_le_1pct": all(r["skipped_frac"] <= .01 for r in rs),
               "clients_applied": all(r.get("clients_without_applied", 0) == 0 for r in rs),
               "lr_eq_control": len(ref) >= R and all(abs(a["lr"] - b["lr"]) < 1e-12 for a, b in zip(rs, ref))}
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
        ok_cm = ok_m = True
        for r in rs:
            with np.load(d / f"cms_r{r['round']:03d}.npz") as z, np.load(ref_d / f"cms_r{r['round']:03d}.npz") as y:
                ok_cm &= bool(np.array_equal(z["private"].sum(2), y["private"].sum(2)) and np.array_equal(z["ids"], y["ids"]))
                ten, tr = summary(z["private"]), summary(y["private"])
                ok_m &= all(abs(ten[k] - r["private/" + k]) < 1e-9 for k in TEN)
        chk.update(cm_support_eq_control=ok_cm, ten_metrics_from_cm=ok_m)
        if spec["seed"] == 42:
            pd = next(iter((ROOT / "artifacts/fbl-full").glob(f"{spec['run']}/**/{spec['run']}/rounds.jsonl"))).parent
            pr = (pd / "rounds.jsonl").read_text()
            chk["prefix_eq_pilot"] = f.read_text().startswith(pr) and all(
                W.sha(d / f"cms_r{r:03d}.npz") == W.sha(pd / f"cms_r{r:03d}.npz") for r in range(1, 21))   # pilot = r1-20
        prev = [r for r in rows if r["seed"] == spec["seed"] and r.get("session", 1) < spec.get("session", 1)]
        if prev:                                                    # every earlier session's committed rows are a byte prefix
            chk["prefix_eq_sessions"] = all(f.read_text().startswith(
                next(iter(ART.glob(f"{r['slug']}/**/{r['run']}/rounds.jsonl"))).read_text()) for r in prev)
        chk["pass"] = all(bool(v) for v in chk.values())
        boot = cfg["payload_bytes"]["bootstrap"]
        res = {"checks": chk, "endpoint": ten, "control_endpoint": tr, "delta_end": ten["f1_macro"] - tr["f1_macro"],
               "bootstrap_bytes": boot, "round_bytes": sum(r["bytes_up"] + r["bytes_down"] for r in rs),
               "t_train_hours": sum(r["t_train"] for r in rs) / 3600,
               "control_t_train_hours": sum(r["t_train"] for r in ref[:R]) / 3600}
        res.update(total_bytes=boot["up"] + boot["down"] + res["round_bytes"], compute_x_fedavg_executed=COMPUTE50[0],
                   targets={"macro_gt": MACRO50, "total_bytes_le": TOTAL50, "compute_le": COMPUTE50[1]})
        ok = chk["pass"] and ten["f1_macro"] > MACRO50 and res["total_bytes"] <= TOTAL50 and COMPUTE50[0] <= COMPUTE50[1]
        res["verdict"] = "PASS" if ok else "FAIL"
        res["sessions"] = spec.get("session", 1)
        out[f"L4F{spec['seed']}"] = res
        print(spec["arm"], "audit", "PASS" if chk["pass"] else "FAIL", {k: v for k, v in chk.items() if not v},
              "macro r50 %.4f (T3 s%d %.4f) total %.0f B" % (ten["f1_macro"], spec["seed"], tr["f1_macro"], res["total_bytes"]),
              res["verdict"], flush=True)
    atomic_json(OUT / "audit.json", out)


def main():
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "build":
        arm, owner, cap = sys.argv[2], sys.argv[3], float(sys.argv[4])
        rows = json.loads((OUT / "batch.json").read_text()) if (OUT / "batch.json").exists() else []
        assert not (OUT / "launch-complete.json").exists() and all(r["arm"] != f"L4F{arm[1:]}" for r in rows), "already built"
        atomic_json(OUT / "batch.json", rows + [build(arm, owner, cap)])
    elif cmd == "launch":
        W.launch(json.loads((OUT / "batch.json").read_text()))
    elif cmd == "hop":                                              # lock runs/fbl-r50/<S41h2>/batch.json (stage + notebook)
        arm, owner, cap = sys.argv[2], sys.argv[3], float(sys.argv[4])
        assert not (OUT / arm / "batch.json").exists(), "hop already locked"
        row = build_hop(arm, owner, cap)
        (OUT / arm).mkdir(exist_ok=True)
        atomic_json(OUT / arm / "batch.json", [row])
    elif cmd == "launch-hop":
        W.OUT = OUT / sys.argv[2]
        W.launch(json.loads((W.OUT / "batch.json").read_text()))
    elif cmd == "collect":
        batch = chain_rows()
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
