"""FB-L full-scale pilot on DAGSNet (docs/research/screening-fbl.md): owner 19/11 -- every arm that passed the FB-L sim on both
models (L4 P4 Q4 L2 P2 Q2, compact engine), seed 42, 5 343 vehicles, r1-20 of the cosine-50 schedule. Same ladder as the FB pilot
(scripts/run_fb_full.py -> scripts/run_lite2_full.py): T3P seed 42 control, collector, audit, locked pilot gate; the arm hp are
the Lite2 pilot hp with the sim arm's FB-L knobs (tier rows, real_select[_q], compact).

    python scripts/run_fbl_full.py quota                                  # scoped live quota -> runs/fbl-full/quota.json
    python scripts/run_fbl_full.py launch L4 P4 ... [--exclude=u1,u2]     # lock batch.json, push once per slug
    python scripts/run_fbl_full.py collect | audit

Pilot verdict (locked before push): delta Macro-F1 mean r16-20 >= -.006 and r20 >= -.006 vs T3 seed 42, bootstrap <= 382 000 B
per vehicle, real_frac every round == full-scale estimator. Best arm: scripts/gate_fbl_full.py (both models).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_fb_full as FB  # noqa: E402
from scripts import run_lite2_full as F  # noqa: E402
from scripts import run_sim_budget as SB  # noqa: E402
from scripts import run_w20 as W  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json  # noqa: E402

OUT, ART, SIM = ROOT / "runs/fbl-full", ROOT / "artifacts/fbl-full", ROOT / "runs/sim-kaggle-fbl"
CAP = 5.0                                                   # h: FB N4/N2 pilot (mask engine) ran 3.55 h train + 0.16 h eval
KNOBS = ("synth_tier_rows", "real_tier_rows", "real_select", "real_select_q", "compact")
F.OUT, F.ART, F.PILOT_OUT = OUT, ART, OUT
FB.OUT, FB.ART = OUT, ART
W.OUT, W.ART, W.TAG = OUT, ART, "FBLF"
W.cli = SB.scoped_cli


def rows(arms, quota, exclude):
    sim = {s["arm"]: s for s in json.loads((SIM / "batch.json").read_text())}
    v = json.loads((SIM / "verdict-r20.json").read_text())
    lite = json.loads((ROOT / "runs/lite2-full/batch.json").read_text())[0]
    assert json.loads((ROOT / "runs/lite2-full/audit.json").read_text())["BU1P"]["verdict"] == "PASS"
    head, builder = W.git_head(), json.loads((ROOT / "runs/w-study/builder.json").read_text())
    slots = [q["user"] for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"]))
             if not (q.get("inventory_error") or q.get("unknown_recent") or q.get("gpu_reserved", 0) or q.get("active_recent"))
             and q.get("gpu_left", 0) >= CAP + 1 / 60 and q["user"] not in exclude]          # owner 19/11: 1-minute quota margin
    assert len(slots) >= len(arms), (arms, slots)
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
    batch = []
    for arm, owner in zip(arms, slots):
        s = sim[arm]
        assert v["dagsnet"][arm]["pass"] and v["bert"][arm]["pass"], arm
        assert s["overlay_sha256"] == {p: W.sha(ROOT / p) for p in SB.OVERLAYS}, "method bytes differ from the audited FB-L sim"
        knobs = [h for h in s["hp"] if h.split("=")[0] in KNOBS]
        assert "compact=true" in knobs
        real = json.loads(next(h for h in knobs if h.startswith("real_tier_rows")).split("=", 1)[1])
        hp = [h for h in lite["hp"] if h.split("=")[0] not in KNOBS] + knobs
        tag = s["slug"].removeprefix("mh-sim-fb-").removesuffix("-s42-r20")
        slug = f"mh-fbl-{tag}-s42-r20"
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
                c["source"] = ['print("W&B off: FB-L full-scale (as FB/Lite2/T3 pilot)")\n']
            if c["cell_type"] == "code":
                compile("".join(c["source"]), str(nb), "exec")
        nb.write_text(json.dumps(doc, indent=1) + "\n")
        m = json.loads(meta.read_text())
        assert m["is_private"] and m["id"] == f"{owner}/{slug}" and m["kernel_sources"] == [builder["ref"]]
        assert all(m[k] == runtime[k] for k in ("machine_shape", "docker_image"))
        batch.append(dict(arm=arm + "P", sim_arm=arm, seed=42, method="mobihet_budget", account=owner, slug=slug, version=1, hp=hp,
                          base_sha=head, overlay_sha256={p: W.sha(ROOT / p) for p in SB.OVERLAYS}, notebook_sha256=W.sha(nb),
                          metadata_sha256=W.sha(meta), cap_hours=CAP, endpoint=20, lr_schedule_rounds=50,
                          real_frac_estimator=FB.real_frac(real), cost_x_fedavg=s["cost_x_fedavg"],
                          control="T3P seed 42 (b21-full/mh-k-b21-t3p-r20)", npy_kernel=builder["ref"],
                          npy_manifest_sha256=builder["manifest_sha256"]))
    return batch


def main():
    cmd = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "quota":
        sys.argv = [sys.argv[0], "quota"]
        SB.OUT = OUT
        SB.main()
    elif cmd == "launch":
        arms = [a for a in sys.argv[2:] if not a.startswith("--exclude=")]
        exclude = {u for a in sys.argv[2:] if a.startswith("--exclude=") for u in a.split("=", 1)[1].split(",")}
        if not (OUT / "batch.json").exists():
            atomic_json(OUT / "batch.json", rows(arms, json.loads((OUT / "quota.json").read_text()), exclude))
        W.launch(json.loads((OUT / "batch.json").read_text()))
    elif cmd == "collect":
        sys.argv = [sys.argv[0], "collect"]
        F.main()
    else:
        assert cmd == "audit", cmd
        FB.audit()                                          # Lite2 checks + pilot gate + real_frac == estimator, on OUT/ART


if __name__ == "__main__":
    main()
