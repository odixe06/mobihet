"""Full-scale KM16-copula r50 (5343 vehicles, official own-test, cosine50 to r50, seeds 42/43) vs Core F2/F3 r50.

Configuration chosen on the fold-val simulator only (docs/research/screening-sim-foldval.md b5-b8): K16, shrinkage w0 0.5D,
per-vehicle int8 upload codec. Rule locked before launch: docs/research/screening-km50.md. Session over budget -> resume same owner.

    python scripts/run_km50.py launch | collect
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
BATCH = "km50"
OUT, ART = ROOT / "runs/km50", ROOT / "artifacts/km50"
CAP = 11.5
OVERLAYS = ["methods/mobihet_sim.py"]
from scripts import f50_common  # noqa: E402
OFF = {"diagnostics": "false", "teacher_diagnostics": "false", "external_diagnostics": "false"}
KM = ["cop_kmeans=16", "cop_kmeans_w0=0.5", "cop_kmeans_q=int8"]
ARMS = [("Q42", "km16q8", 42, KM), ("Q43", "km16q8", 43, KM)]
W.OUT, W.ART, W.TAG = OUT, ART, "KM50"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|arch_rank\d\.json)$)'


def hp(seed, extra):
    """F2/F3 Core hp (f50_common) with probes off (they restore every RNG: trajectory-neutral) + the generator knob."""
    base = [h for h in f50_common.hp_for("core", seed) if h.split("=", 1)[0] not in OFF]
    return base + [f"{k}={v}" for k, v in OFF.items()] + list(extra)


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
        slug = f"mh-k-{tag}-s{seed}-r50"
        h = hp(seed, extra)
        args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_sim", "--out", f"/kaggle/working/{slug}", "--width", "1",
                           "--rounds", "50", "--until", "50", "--save-every", "5", "--compile", "--max-hours", "BUDGET",
                           "--cache", "", "--set", *h])
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
        batch.append(dict(arm=arm, seed=seed, method="mobihet_sim", account=owner, slug=slug, version=1, hp=h, base_sha=head,
                          overlay_sha256={p: W.sha(ROOT / p) for p in OVERLAYS}, notebook_sha256=W.sha(notebook),
                          metadata_sha256=W.sha(meta), cap_hours=CAP, endpoint=50, lr_schedule_rounds=50,
                          npy_kernel=builder["ref"], npy_manifest_sha256=builder["manifest_sha256"]))
    atomic_json(OUT / "batch.json", batch)
    print("KM50_LOCKED", [(r["arm"], r["account"], r["slug"]) for r in batch], flush=True)
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
