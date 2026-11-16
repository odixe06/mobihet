"""Codec combo: 6 full-scale pilots (5343 vehicles, official own-test, cosine50 stopped at r20, state saved for r50).

Owner 10/11: two KM16 combos x seeds 42/43 vs KM50 of the same seed, plus Core + q_bits=8 vs an in-batch Core control
(mobihet_sim, every knob off) so the Core/KM choice compares both at reduced bandwidth.
  CBA = cop_kmeans_lite=6 (CH6 + CM8 + CT8) + q_bits=8 + cop_kmeans_iters=0   (cheapest PASS level of every slot)
  CBB = cop_kmeans_lite=6 + q_bits=8 (4 Lloyd passes)
  C0  = Core (mobihet_sim, knobs off)      CQ8 = Core + q_bits=8
Rule locked before push: docs/research/screening-codec.md "Combo r20".

    python scripts/run_codec_combo20.py quota | launch | collect
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

OUT, ART = ROOT / "runs/codec-combo20", ROOT / "artifacts/codec-combo20"
CAP = 7.5
OVERLAYS = ["methods/mobihet_sim.py"]
B = ["km_bytes=true"]
CBA = ["cop_kmeans_lite=6", "q_bits=8", "cop_kmeans_iters=0"] + B
CBB = ["cop_kmeans_lite=6", "q_bits=8"] + B
# (arm, slug tag, seed, km on, knobs on top of the control hp)
ARMS = [("CBA42", "cba", 42, True, CBA), ("CBA43", "cba", 43, True, CBA),
        ("CBB42", "cbb", 42, True, CBB), ("CBB43", "cbb", 43, True, CBB),
        ("C0", "core", 42, False, []), ("CQ8", "cq8", 42, False, ["q_bits=8"])]
W.OUT, W.ART, W.TAG = OUT, ART, "COMBO"
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
    for (arm, tag, seed, km, extra), owner in zip(ARMS, slots):
        slug = f"mh-k-{tag}-s{seed}-p20"
        if km:
            h = K.hp(seed, K.KM + extra)
            ctrl = control["Q42" if seed == 42 else "Q43"]
            assert h == ctrl["hp"] + extra, "hp must equal KM50 control + codec knobs only"
            ctrl_ref = f'{ctrl["account"]}/{ctrl["slug"]}/{ctrl["version"]}'
        else:
            h = K.hp(seed, extra)
            assert h == K.hp(seed, []) + extra
            ctrl_ref = "C0 (this batch)" if extra else "F2 core s42 r50 (report only)"
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
                c["source"] = ['print("W&B off: codec combo")\n']
            if c["cell_type"] == "code":
                compile("".join(c["source"]), str(notebook), "exec")
        notebook.write_text(json.dumps(doc, indent=1) + "\n")
        metadata = json.loads(meta.read_text())
        runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
        assert metadata["is_private"] and metadata["id"] == f"{owner}/{slug}" and metadata["kernel_sources"] == [builder["ref"]]
        assert all(metadata[k] == runtime[k] for k in ("machine_shape", "docker_image"))
        batch.append(dict(arm=arm, seed=seed, km=km, method="mobihet_sim", account=owner, slug=slug, version=1, hp=h, knobs=extra,
                          base_sha=head, overlay_sha256={p: W.sha(ROOT / p) for p in OVERLAYS}, notebook_sha256=W.sha(notebook),
                          metadata_sha256=W.sha(meta), cap_hours=CAP, endpoint=20, lr_schedule_rounds=50, control=ctrl_ref,
                          npy_kernel=builder["ref"], npy_manifest_sha256=builder["manifest_sha256"]))
    atomic_json(OUT / "batch.json", batch)
    print("COMBO_LOCKED", [(r["arm"], r["account"], r["slug"]) for r in batch], flush=True)
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
