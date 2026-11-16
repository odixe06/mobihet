"""Continue the KM16 full-scale pilot K42/K43 from its saved r20 state to r50 (cosine50 schedule locked at launch), same owner.

    python scripts/resume_km_pilot50.py launch    # build -c2 notebooks from the exact pilot notebooks + verified import cell, push
    python scripts/resume_km_pilot50.py collect   # pull finished outputs into artifacts/km-pilot50/<kernel>

Same notebook bytes (overlay, hp, seed) as the pilot; only the import cell, --until 50 and the session budget change.
Rule (locked before launch): docs/research/screening-km-pilot50.md.
"""
import concurrent.futures
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_w20 as W  # noqa: E402
from scripts.resume_checkpoint_cells import kernel_import  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json, query  # noqa: E402

PILOT = ROOT / "runs/km-pilot20"
OUT, ART = ROOT / "runs/km-pilot50", ROOT / "artifacts/km-pilot50"
CAP = 8.0
W.OUT, W.ART, W.TAG = OUT, ART, "KM50"
W.COLLECT = r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz)$)'


def prepare():
    if (OUT / "batch.json").exists():
        return json.loads((OUT / "batch.json").read_text())
    batch = []
    for orig in json.loads((PILOT / "batch.json").read_text()):
        owner, run = orig["account"], orig["slug"]
        kernel = run + "-c2"
        q = W.ka().account_quota(owner)
        assert q["gpu_left"] >= 2 * CAP * 1.15, (owner, q["gpu_left"], "quota for both resume sessions")
        nb = json.loads((ROOT / "notebooks" / run / (run + ".ipynb")).read_text())
        assert W.sha(ROOT / "notebooks" / run / (run + ".ipynb")) == orig["notebook_sha256"], "pilot notebook changed"
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
            src = src.replace(f"budget = {orig['cap_hours']} -", f"budget = {CAP} -").replace("--until 20 ", "--until 50 ")
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
        batch.append(dict(orig, slug=kernel, run_slug=run, version=1, cap_hours=CAP, endpoint=50, resume_from=f"{owner}/{run}", resume_round=20,
                          notebook_sha256=W.sha(path), metadata_sha256=W.sha(folder / "kernel-metadata.json")))
    atomic_json(OUT / "batch.json", batch)
    print("KM50_LOCKED", [(r["arm"], r["account"], r["slug"]) for r in batch], flush=True)
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
