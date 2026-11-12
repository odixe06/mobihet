"""Generate one private Kaggle notebook (2xT4) that runs scripts/run_fl.py (or another repo script) at a pinned commit.

    python scripts/gen_notebook.py --owner odixe0502 --slug fl2-fedavg --method fedavg --max-hours 11.5
    python .claude/skills/kaggle-training-notebook/scripts/embed_wandb_key.py notebooks/fl2-fedavg/fl2-fedavg.ipynb \
        --metadata notebooks/fl2-fedavg/kernel-metadata.json
    python .claude/skills/kaggle-training-notebook/scripts/kaggle_as.py odixe0502 -- kaggle kernels push -p notebooks/fl2-fedavg

The notebook clones github.com/odixe06/mobihet at --sha (default: local HEAD, must be pushed), reads the v4
dataset mount, resumes from a previous session's output when --resume-kernel is given, and writes everything under
/kaggle/working/<run>. notebooks/ is gitignored: embedded notebooks hold a credential.
"""
import argparse
import base64
import hashlib
import json
import os
import subprocess
from pathlib import Path

import nbformat

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RUNTIME = json.load(open(os.path.join(ROOT, "knowledge", "runtime.json")))
DATASET = "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"
REPO = "https://github.com/odixe06/mobihet.git"
WANDB_PROJECT = "mobihet-v4"


def cells(a, sha):
    run = a.run or a.slug
    args = a.args or (f"scripts/run_fl.py --method {a.method} --out /kaggle/working/{run} --width {a.width} "
                      f"--rounds {a.rounds}" + (f" --until {a.until}" if a.until else "") + f" --compile --wandb {WANDB_PROJECT} --max-hours BUDGET --cache ''" + (" --set " + " ".join(a.set) if a.set else ""))
    yield "markdown", f"# {a.slug}\n\nRepo `{REPO}` @ `{sha}` · dataset `{DATASET}` · `torchrun --nproc_per_node=2 {args}`"
    yield "code", """import time, os, subprocess, sys, glob, shutil
T0 = time.monotonic()
import torch""" + ("" if a.build_npy else """
n = torch.cuda.device_count()
assert n == 2, f"expected 2 GPUs, got {n}"
assert all(torch.cuda.get_device_capability(i) == (7, 5) for i in range(n)), "expected Tesla T4"
print(torch.__version__, sys.version, [torch.cuda.get_device_name(i) for i in range(n)])""")
    overlay = {}
    for name in a.overlay:
        p = Path(name)
        assert not p.is_absolute() and ".." not in p.parts, name
        data = (Path(ROOT) / p).read_bytes()
        overlay[name] = {"base64": base64.b64encode(data).decode(), "sha256": hashlib.sha256(data).hexdigest()}
    yield "code", f"""subprocess.run(["git", "clone", "-q", "{REPO}", "/kaggle/temp/src"], check=True)
subprocess.run(["git", "-C", "/kaggle/temp/src", "checkout", "-q", "{sha}"], check=True)
print(subprocess.run(["git", "-C", "/kaggle/temp/src", "log", "-1", "--format=%H %s"], capture_output=True, text=True).stdout)
import base64, hashlib, json
from pathlib import Path
overlay = {overlay!r}
for name, spec in overlay.items():
    data = base64.b64decode(spec["base64"])
    assert hashlib.sha256(data).hexdigest() == spec["sha256"], name
    dest = Path("/kaggle/temp/src") / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
files = [p for folder in ("core", "methods", "scripts") for p in (Path("/kaggle/temp/src") / folder).glob("*.py")]
files += [Path("/kaggle/temp/src/knowledge") / n for n in ("input_order.json", "label_mapping.json", "runtime.json")]
files = list(dict.fromkeys(files + [Path("/kaggle/temp/src") / name for name in overlay]))
manifest = {{"base_sha": "{sha}", "overlay": {{n: s["sha256"] for n, s in overlay.items()}},
            "files": {{str(p.relative_to("/kaggle/temp/src")): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}}}
provenance = Path("/kaggle/working/provenance")
provenance.mkdir(exist_ok=True)
(provenance / "source_manifest.json").write_text(json.dumps(manifest, indent=2))
for p in files:
    dest = provenance / "source" / p.relative_to("/kaggle/temp/src")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dest)
print("source manifest", hashlib.sha256((provenance / "source_manifest.json").read_bytes()).hexdigest())"""
    if a.build_npy:   # cache builder: no credential, no training; output mounts into training kernels via --npy-kernel
        yield "code", """t = time.monotonic()
sys.path.insert(0, "/kaggle/temp/src")
from core import data as D
out = Path("/kaggle/working/v4npy")
D.build_npy(D.find_root(), str(out))
def digest(p):
    with p.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()
files = {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)} for p in sorted(out.iterdir())}
(out / "manifest.json").write_text(json.dumps({"base_sha": "%s", "files": files}, indent=1))
print(json.dumps(files, indent=1))
print(f"npy cache + manifest in {time.monotonic() - t:.0f}s")""" % sha
        return
    yield "code", """import wandb
# BEGIN INLINE WANDB CREDENTIAL — owner-authorized private notebook
raise SystemExit("W&B credential not embedded: run embed_wandb_key.py")
# END INLINE WANDB CREDENTIAL
print("W&B authenticated")"""
    resume = ""
    if a.resume_kernel:
        resume = f"""prev = [p for p in glob.glob("/kaggle/input/**/state.pt", recursive=True)]
assert len(prev) == 1, f"resume requested: expected 1 state.pt in inputs, found {{len(prev)}}: {{prev}}"
prev_dir = os.path.dirname(prev[0])
target_dir = "/kaggle/working/{run}"
os.makedirs(target_dir, exist_ok=True)
for item in os.listdir(prev_dir):
    s = os.path.join(prev_dir, item)
    d = os.path.join(target_dir, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)
print("resumed from", prev[0])
"""
    if a.npy_kernel:   # prebuilt cache from a --build-npy kernel: verify every byte the driver will read, skip the build
        yield "code", f"""t = time.monotonic()
NPY, = {{str(p.parent) for p in Path("/kaggle/input").glob("**/v4npy/manifest.json")}}
spec = json.loads(Path(NPY, "manifest.json").read_text())
assert spec["files"].keys() >= {{"done", *(f"{{s}}_{{k}}.npy" for s in ("train", "test") for k in ("X", "Y", "off", "ids", "F"))}}
assert {{p.name for p in Path(NPY).iterdir()}} == {{*spec["files"], "manifest.json"}}
for name, f in spec["files"].items():
    with open(Path(NPY, name), "rb") as fh:
        assert hashlib.file_digest(fh, "sha256").hexdigest() == f["sha256"], name
assert hashlib.sha256(Path(NPY, "manifest.json").read_bytes()).hexdigest() == "{a.npy_manifest_sha256}"
print(f"npy cache verified at {{NPY}} in {{time.monotonic() - t:.0f}}s")"""
    else:
        yield "code", """t = time.monotonic()
sys.path.insert(0, "/kaggle/temp/src")
from core import data as D
D.build_npy(D.find_root(), "/kaggle/temp/v4npy")    # one reader; every rank then memmaps (V4_NPY)
NPY = "/kaggle/temp/v4npy"
print(f"npy cache built in {time.monotonic() - t:.0f}s")"""
    yield "code", resume + f"""os.environ.update(TORCHINDUCTOR_COMPILE_THREADS="2", PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",   # GPU near-full: avoid allocator thrash (MobiHet 30/10)
                  V4_NPY=NPY)
budget = {a.max_hours} - (time.monotonic() - T0) / 3600
cmd = "cd /kaggle/temp/src && torchrun --standalone --nproc_per_node=2 {args}".replace("BUDGET", f"{{budget - 0.4:.2f}}")
print(cmd)
rc = subprocess.run(cmd, shell=True).returncode
assert rc == 0, f"training exited {{rc}}"
"""
    yield "code", f"""for f in glob.glob("/kaggle/working/{run}/*.jsonl"):
    print(f, open(f).read()[-3000:])"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--method", default="fedavg")
    ap.add_argument("--run", default="", help="run dir name (default slug); keep it across resume sessions")
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--until", type=int, default=0, help="screening: stop after this round (LR schedule of --rounds)")
    ap.add_argument("--max-hours", type=float, default=11.5)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--args", default="", help="replace the run_fl.py command line (e.g. a centralized script)")
    ap.add_argument("--sha", default="")
    ap.add_argument("--overlay", nargs="*", default=[], help="embed exact local source files over --sha; record hashes, no Git push")
    ap.add_argument("--resume-kernel", default="", help="<owner>/<slug> whose output holds <run>/state.pt")
    ap.add_argument("--input-kernel", default="", help="<owner>/<slug> mounted read-only under /kaggle/input (e.g. CANPerFL's FedAvg)")
    ap.add_argument("--build-npy", action="store_true", help="cache-builder notebook: parquet -> v4npy + manifest, no training/credential")
    ap.add_argument("--npy-kernel", default="", help="<owner>/<slug> of a --build-npy kernel; skip the in-kernel build")
    ap.add_argument("--npy-manifest-sha256", default="", help="sha256 of that kernel's v4npy/manifest.json (pulled and checked locally)")
    a = ap.parse_args()
    assert bool(a.npy_kernel) == bool(a.npy_manifest_sha256), "--npy-kernel needs the locally verified manifest sha256"
    sha = a.sha or subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(s) if k == "markdown" else nbformat.v4.new_code_cell(s) for k, s in cells(a, sha)]
    nb.metadata.update({
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": "3.12"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True, "language": "python",
                   "sourceType": "notebook"}})
    out = os.path.join(ROOT, "notebooks", a.slug)
    os.makedirs(out, exist_ok=True)
    nbformat.write(nb, os.path.join(out, f"{a.slug}.ipynb"))
    meta = {"id": f"{a.owner}/{a.slug}", "title": a.slug, "code_file": f"{a.slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4", "docker_image": RUNTIME["docker_image"],
            "dataset_sources": [DATASET], "kernel_sources": [k for k in (a.resume_kernel, a.input_kernel, a.npy_kernel) if k],
            "competition_sources": [], "model_sources": []}
    if a.build_npy:   # CPU only, no credential; public (owner 02/11) so every training account can mount its output
        meta.update(is_private=False, enable_gpu=False)
        del meta["machine_shape"]
        nb.metadata["kaggle"].update(accelerator="none", isGpuEnabled=False)
        nbformat.write(nb, os.path.join(out, f"{a.slug}.ipynb"))
    json.dump(meta, open(os.path.join(out, "kernel-metadata.json"), "w"), indent=1)
    print(out, sha)


if __name__ == "__main__":
    main()
