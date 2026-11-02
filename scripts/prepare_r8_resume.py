"""Freeze R8 (margin seed 42) resume notebook; run round 20 on takids1909 using kernel_sources."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/r-study/recovery-r8"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / "batch.json").exists(), "R8 recovery already frozen; inspect receipts"
    parent = next(s for s in json.loads((ROOT / "runs/r-study/batch.json").read_text()) if s["arm"] == "R8")
    for name, h in parent["overlay_sha256"].items():
        assert sha(ROOT / name) == h, name
    overlay = parent["overlay_sha256"]
    account = "takids1909"
    source_slug = parent["slug"]
    slug = "mh-r8-replay-margin-s42-resume1"
    run_dir = source_slug
    args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_rstudy", "--out", f"/kaggle/working/{run_dir}",
        "--width", "1", "--rounds", "50", "--until", "20", "--compile", "--wandb", "mobihet-v4",
        "--max-hours", "BUDGET", "--cache", "", "--set", *parent["hp"]])
    subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", account, "--slug", slug,
        "--run", run_dir, "--resume-kernel", f"{account}/{source_slug}", "--sha", parent["base_sha"],
        "--max-hours", "1.5", "--args", args, "--overlay", *overlay], check=True, cwd=ROOT, capture_output=True)
    folder = ROOT / "notebooks" / slug
    notebook, metadata = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
    document = json.loads(notebook.read_text())
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = "".join(cell["source"]).replace('TORCHINDUCTOR_COMPILE_THREADS="2"',
                'TORCHINDUCTOR_COMPILE_THREADS="1"').splitlines(keepends=True)
    ready = json.loads((ROOT / "runs/r-study/input-datasets" / (account + ".ready.json")).read_text())
    assert ready["private"] and ready["download_sha_verified"] and ready["decoded_sha256"] == parent["replay_source_sha256"]
    meta = json.loads(metadata.read_text())
    meta["dataset_sources"].append(ready["ref"])
    metadata.write_text(json.dumps(meta, indent=1) + "\n")
    for cell in document["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and "D.build_npy(" in source:
            cell["source"] = source.replace('from core import data as D',
                'from core import data as D\nfrom methods.mobihet_rstudy import load_schedule\n'
                + f'_history = load_schedule(42, "cuda:0", "{parent["replay_source_sha256"]}")\ndel _history').splitlines(keepends=True)
    notebook.write_text(json.dumps(document, indent=1) + "\n")
    assert json.loads(metadata.read_text())["is_private"] is True
    subprocess.run([sys.executable, str(SKILL / "embed_wandb_key.py"), str(notebook), "--metadata", str(metadata)],
                   check=True, capture_output=True)
    document, meta = json.loads(notebook.read_text()), json.loads(metadata.read_text())
    assert meta["is_private"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
    assert meta["kernel_sources"] == [f"{account}/{source_slug}"]
    assert ready["ref"] in meta["dataset_sources"]
    assert meta["docker_image"] == json.loads((ROOT / "knowledge/runtime.json").read_text())["docker_image"]
    assert document["metadata"]["kaggle"]["isGpuEnabled"]
    for cell in document["cells"]:
        if cell["cell_type"] == "code": compile("".join(cell["source"]), str(notebook), "exec")
    spec = parent | {"arm": "R8R1", "original_arm": "R8", "account": account, "slug": slug, "run_dir": run_dir, "version": 1,
        "resume_source": f"{account}/{source_slug}", "notebook_sha256": sha(notebook), "metadata_sha256": sha(metadata)}
    (OUT / "batch.json").write_text(json.dumps([spec], indent=2) + "\n")
    print("R8_RESUME_NOTEBOOK_LOCKED", flush=True)


if __name__ == "__main__": main()
