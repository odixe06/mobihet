"""Freeze one P21 seed43 recovery notebook; original batch and eight scientific overlays unchanged."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p-study/recovery-p21"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "batch.json").exists(), "Recovery already frozen; inspect before changing anything"
    parent = next(s for s in json.loads((ROOT / "runs/p-study/batch.json").read_text()) if s["arm"] == "P21")
    preflight = json.loads((OUT / "preflight.json").read_text())
    assert all(preflight[k] for k in ["finite_training_bitexact", "rng_and_metrics_bitexact",
                                     "eager_or_peer_failure_rejected", "nonfinite_packet_rejected"])
    for name, h in preflight["source_sha256"].items(): assert sha(ROOT / name) == h, name
    for name, h in parent["overlay_sha256"].items(): assert sha(ROOT / name) == h, name
    overlay = parent["overlay_sha256"] | {"scripts/run_p21_recovery.py": sha(ROOT / "scripts/run_p21_recovery.py")}
    slug = "mh-p21-hybrid-s43-recovery1"
    account = "accanniez"
    args = shlex.join(["scripts/run_p21_recovery.py", "--method", "mobihet_study", "--out", f"/kaggle/working/{slug}",
        "--width", "1", "--rounds", "50", "--until", "20", "--compile", "--wandb", "mobihet-v4",
        "--max-hours", "BUDGET", "--cache", "", "--set", *parent["hp"]])
    subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", account, "--slug", slug,
        "--method", "mobihet_study", "--sha", parent["base_sha"], "--max-hours", "7", "--args", args,
        "--overlay", *overlay], check=True, cwd=ROOT, capture_output=True)
    folder = ROOT / "notebooks" / slug
    notebook, metadata = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
    document = json.loads(notebook.read_text())
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            cell["source"] = "".join(cell["source"]).replace('TORCHINDUCTOR_COMPILE_THREADS="2"',
                'TORCHINDUCTOR_COMPILE_THREADS="1"').splitlines(keepends=True)
    notebook.write_text(json.dumps(document, indent=1) + "\n")
    assert json.loads(metadata.read_text())["is_private"] is True
    subprocess.run([sys.executable, str(SKILL / "embed_wandb_key.py"), str(notebook), "--metadata", str(metadata)],
                   check=True, capture_output=True)
    document, meta = json.loads(notebook.read_text()), json.loads(metadata.read_text())
    assert meta["is_private"] is True and meta["machine_shape"] == "NvidiaTeslaT4" and meta["kernel_sources"] == []
    assert meta["docker_image"] == json.loads((ROOT / "knowledge/runtime.json").read_text())["docker_image"]
    assert document["metadata"]["kaggle"]["isGpuEnabled"]
    for cell in document["cells"]:
        if cell["cell_type"] == "code": compile("".join(cell["source"]), str(notebook), "exec")
    spec = parent | {"arm": "P21R1", "original_arm": "P21", "account": account, "slug": slug, "version": 1,
        "overlay_sha256": overlay, "notebook_sha256": sha(notebook), "preflight_sha256": sha(OUT / "preflight.json"),
        "restart": "fresh r1, no compatible checkpoint exists", "guard": "strict compile gate + nonfinite metadata; finite path bitexact locally"}
    (OUT / "batch.json").write_text(json.dumps([spec], indent=2) + "\n")
    (ROOT / "docs/research/p21-recovery-launch.json").write_text(json.dumps({"batch_sha256": sha(OUT / "batch.json"),
        "batch": [spec], "scope": "One authorized missing seed, same locked P21 configuration; push/audit receipts required"}, indent=2) + "\n")
    print("P21_RECOVERY_NOTEBOOK_LOCKED", flush=True)


if __name__ == "__main__": main()
