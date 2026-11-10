"""Codec combo r50: resume CBB42/CBB43/CQ8-s42 from their saved r20 state (same owner) + fresh CQ8 s43 r50 (own account).

    python scripts/run_codec_combo50.py launch    # build notebooks, lock batch.json, push
    python scripts/run_codec_combo50.py collect   # pull terminal kernels into artifacts/codec-combo50/<kernel>

Rule locked before push: docs/research/screening-codec.md "Combo r50". Same notebook bytes as the r20 pilots except the
verified import cell, --until 50 and the session budget (resume rows); the fresh row follows scripts/run_km50.py with Core + q_bits=8.
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
from scripts.resume_checkpoint_cells import kernel_import  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

PILOT = ROOT / "runs/codec-combo20"
OUT, ART = ROOT / "runs/codec-combo50", ROOT / "artifacts/codec-combo50"
CAPS = {"CBB42": 7.0, "CBB43": 7.0, "CQ8": 8.0}            # resume caps (live quota 10/11: annhannh331 16.2 h, trietbackup 16.3 h)
FRESH_OWNER, FRESH_CAP = "minhtrit06", 11.5                # CQ8 s43 from r1; live quota 24.3 h
W.OUT, W.ART, W.TAG = OUT, ART, "C50"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'


def resume_row(orig):
    owner, run, cap = orig["account"], orig["slug"], CAPS[orig["arm"]]
    kernel = run.replace("-p20", "-r50-c2")
    nb_path = ROOT / "notebooks" / run / (run + ".ipynb")
    assert W.sha(nb_path) == orig["notebook_sha256"], "pilot notebook changed"
    nb = json.loads(nb_path.read_text())
    idx = next(i for i, c in enumerate(nb["cells"]) if "NPY, =" in "".join(c.get("source", [])))
    nb["cells"].insert(idx, dict(cell_type="code", execution_count=None, metadata={}, outputs=[],
                                 source=kernel_import(run, 20, orig["method"], orig["seed"]).splitlines(keepends=True)))
    n_budget = n_until = 0
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        n_budget += src.count(f"budget = {orig['cap_hours']} -")
        n_until += src.count("--until 20 ")
        src = src.replace(f"budget = {orig['cap_hours']} -", f"budget = {cap} -").replace("--until 20 ", "--until 50 ")
        compile(src, kernel, "exec")
        c["source"] = src.splitlines(keepends=True)
    assert n_budget == 1 and n_until == 1, (n_budget, n_until)
    folder = ROOT / "notebooks" / kernel
    folder.mkdir(exist_ok=True)
    path = folder / (kernel + ".ipynb")
    path.write_text(json.dumps(nb, indent=1) + "\n")
    meta = json.loads((ROOT / "notebooks" / run / "kernel-metadata.json").read_text())
    meta.update(id=f"{owner}/{kernel}", title=kernel, code_file=kernel + ".ipynb", kernel_sources=[f"{owner}/{run}", orig["npy_kernel"]])
    assert meta["is_private"]
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1) + "\n")
    return dict(orig, slug=kernel, run_slug=run, version=1, cap_hours=cap, endpoint=50, resume_from=f"{owner}/{run}", resume_round=20,
                notebook_sha256=W.sha(path), metadata_sha256=W.sha(folder / "kernel-metadata.json"))


def fresh_row(prev):
    owner, seed, arm, slug = FRESH_OWNER, 43, "CQ8s43", "mh-k-cq8-s43-r50"
    head = W.git_head()
    builder = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    h = K.hp(seed, ["q_bits=8"])
    assert h == K.hp(seed, []) + ["q_bits=8"]
    args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_sim", "--out", f"/kaggle/working/{slug}", "--width", "1",
                       "--rounds", "50", "--until", "50", "--save-every", "5", "--compile", "--max-hours", "BUDGET",
                       "--cache", "", "--set", *h])
    subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", head, "--run", slug,
                    "--max-hours", str(FRESH_CAP), "--args", args, "--npy-kernel", builder["ref"],
                    "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *K.OVERLAYS],
                   cwd=ROOT, check=True, capture_output=True)
    folder = ROOT / "notebooks" / slug
    notebook, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
    doc = json.loads(notebook.read_text())
    for c in doc["cells"]:
        src = "".join(c["source"])
        if "BEGIN INLINE WANDB CREDENTIAL" in src:
            c["source"] = ['print("W&B off: codec combo r50")\n']
        if c["cell_type"] == "code":
            compile("".join(c["source"]), str(notebook), "exec")
    notebook.write_text(json.dumps(doc, indent=1) + "\n")
    metadata = json.loads(meta.read_text())
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
    assert metadata["is_private"] and metadata["id"] == f"{owner}/{slug}" and metadata["kernel_sources"] == [builder["ref"]]
    assert all(metadata[k] == runtime[k] for k in ("machine_shape", "docker_image"))
    return dict(arm=arm, seed=seed, km=False, method="mobihet_sim", account=owner, slug=slug, version=1, hp=h, knobs=["q_bits=8"],
                base_sha=head, overlay_sha256={p: W.sha(ROOT / p) for p in K.OVERLAYS}, notebook_sha256=W.sha(notebook),
                metadata_sha256=W.sha(meta), cap_hours=FRESH_CAP, endpoint=50, lr_schedule_rounds=50, control="F3 core s43 r50",
                npy_kernel=builder["ref"], npy_manifest_sha256=builder["manifest_sha256"])


def prepare():
    if (OUT / "batch.json").exists():
        return json.loads((OUT / "batch.json").read_text())
    k = W.ka()
    batch = []
    for orig in json.loads((PILOT / "batch.json").read_text()):
        if orig["arm"] in CAPS:
            batch.append(resume_row(orig))
    assert [r["arm"] for r in batch] == ["CBB42", "CBB43", "CQ8"], [r["arm"] for r in batch]
    batch.append(fresh_row(batch))
    need = {}
    for r in batch:
        need[r["account"]] = need.get(r["account"], 0) + r["cap_hours"] * 1.15
    for owner, hrs in need.items():                                  # live quota gate before locking
        q = k.account_quota(owner)
        assert q["gpu_left"] >= hrs, (owner, q["gpu_left"], hrs)
    atomic_json(OUT / "batch.json", batch)
    print("C50_LOCKED", [(r["arm"], r["account"], r["slug"], r["cap_hours"]) for r in batch], flush=True)
    return batch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "launch":
        W.launch(prepare())
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
