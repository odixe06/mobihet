"""Full-scale KM16-lite6 pilot (5343 vehicles, official own-test, cosine50 stopped at r20, seed42) vs KM50 Q42 r20.

Codec `cop_kmeans_lite=6` chosen on the fold-val simulator only (sim b9, docs/research/screening-km-lite.md); every other hp is
byte-identical to KM50 Q42 (`runs/km50/batch.json`). Rule locked before push: screening-km-lite.md §"Pilot full-scale lite6".
State r20 of both ranks is saved (--save-every 5) so a PASS can be resumed to r50 on the same schedule.

    python scripts/run_km_lite_pilot20.py quota | launch | collect
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
from scripts import run_km50 as K  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

OUT, ART = ROOT / "runs/km-lite-pilot20", ROOT / "artifacts/km-lite-pilot20"
CAP = 7.5
OVERLAYS = ["methods/mobihet_sim.py"]
ARMS = [("L42", "km16lite6", 42, K.KM + ["cop_kmeans_lite=6"])]
W.OUT, W.ART, W.TAG = OUT, ART, "KMLITE"
W.COLLECT = K.W.COLLECT


def prepare(quota):
    if (OUT / "batch.json").exists():
        return json.loads((OUT / "batch.json").read_text())
    head = W.git_head()
    builder = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    control = {r["arm"]: r for r in json.loads((ROOT / "runs/km50/batch.json").read_text())}
    slots = []
    for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"])):
        if q.get("inventory_error") or q.get("unknown_recent") or q.get("active_recent") or q.get("gpu_reserved", 0):
            continue
        slots += [q["user"]] * max(0, min(2, int(q.get("gpu_left", 0) // (CAP * 1.15))))
    assert len(slots) >= len(ARMS), f"need {len(ARMS)} slots, have {len(slots)}"
    batch = []
    for (arm, tag, seed, extra), owner in zip(ARMS, slots):
        slug = f"mh-k-{tag}-s{seed}-p20"
        h = K.hp(seed, extra)
        ctrl = control["Q42" if seed == 42 else "Q43"]
        assert h == ctrl["hp"] + ["cop_kmeans_lite=6"], "hp must equal KM50 control + codec knob only"
        args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_sim", "--out", f"/kaggle/working/{slug}", "--width", "1",
                           "--rounds", "50", "--until", "20", "--save-every", "5", "--compile", "--max-hours", "BUDGET",
                           "--cache", "", "--set", *h])
        subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", head, "--run", slug,
                        "--max-hours", str(CAP), "--args", args, "--npy-kernel", builder["ref"],
                        "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *OVERLAYS],
                       cwd=ROOT, check=True, capture_output=True)
        folder = ROOT / "notebooks" / slug
        notebook, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
        doc = json.loads(notebook.read_text())
        for c in doc["cells"]:                                        # no W&B (as KM50): no credential embedded
            src = "".join(c["source"])
            if "BEGIN INLINE WANDB CREDENTIAL" in src:
                c["source"] = ['print("W&B off: KM-lite pilot")\n']
            if c["cell_type"] == "code":
                compile("".join(c["source"]), str(notebook), "exec")
        notebook.write_text(json.dumps(doc, indent=1) + "\n")
        metadata = json.loads(meta.read_text())
        runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
        assert metadata["is_private"] and metadata["id"] == f"{owner}/{slug}" and metadata["kernel_sources"] == [builder["ref"]]
        assert all(metadata[k] == runtime[k] for k in ("machine_shape", "docker_image"))
        batch.append(dict(arm=arm, seed=seed, method="mobihet_sim", account=owner, slug=slug, version=1, hp=h, base_sha=head,
                          overlay_sha256={p: W.sha(ROOT / p) for p in OVERLAYS}, notebook_sha256=W.sha(notebook),
                          metadata_sha256=W.sha(meta), cap_hours=CAP, endpoint=20, lr_schedule_rounds=50,
                          control=f'{ctrl["account"]}/{ctrl["slug"]}/{ctrl["version"]}',
                          npy_kernel=builder["ref"], npy_manifest_sha256=builder["manifest_sha256"]))
    atomic_json(OUT / "batch.json", batch)
    print("KMLITE_LOCKED", [(r["arm"], r["account"], r["slug"]) for r in batch], flush=True)
    return batch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "quota":                                                 # fresh receipt in this batch folder only
        for a in W.inventory()["accounts"]:
            print(a["user"], round(a.get("gpu_left", 0), 2), a.get("active_recent"), a.get("inventory_error", ""))
        return
    if cmd == "launch":
        W.launch(prepare(json.loads((OUT / "quota.json").read_text())))
        return
    if cmd == "collect":
        batch = json.loads((OUT / "batch.json").read_text())
        (OUT / "collection").mkdir(exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
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
