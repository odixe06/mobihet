"""Push R8 resume kernel to takids1909, verify initial QUEUED/RUNNING status and write receipts."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.watch_kaggle_batch import status_from_cli, atomic_json

OUT = ROOT / "runs/r-study/recovery-r8"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    batch = json.loads((OUT / "batch.json").read_text())
    assert len(batch) == 1
    spec = batch[0]
    account = spec["account"]
    slug = spec["slug"]
    nb_dir = ROOT / "notebooks" / slug
    assert (nb_dir / f"{slug}.ipynb").exists()
    assert (nb_dir / "kernel-metadata.json").exists()

    cmd = [sys.executable, str(SKILL / "kaggle_as.py"), account, "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]

    # Check quota
    quota_proc = subprocess.run([sys.executable, str(SKILL / "kaggle_as.py"), account, "--", sys.executable,
        str(ROOT / "scripts/kaggle_cli_bounded.py"), "kernels", "list", "--mine", "--page-size", "5", "--csv"],
        capture_output=True, text=True)
    assert quota_proc.returncode == 0, f"Cannot reach Kaggle CLI for {account}"

    push_receipt = OUT / f"{slug}.push.json"
    if not push_receipt.exists():
        print(f"Pushing {account}/{slug}...", flush=True)
        push = subprocess.run([sys.executable, str(SKILL / "kaggle_as.py"), account, "--", sys.executable,
            str(ROOT / "scripts/kaggle_cli_bounded.py"), "kernels", "push", "-p", str(nb_dir)],
            capture_output=True, text=True)
        assert push.returncode == 0, f"Push failed: {push.stderr}\n{push.stdout}"
        print(f"Push output: {push.stdout.strip()}", flush=True)
        atomic_json(push_receipt, {
            "account": account, "slug": slug, "version": 1, "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "stdout": push.stdout, "stderr": push.stderr, "returncode": push.returncode,
            "notebook_sha256": sha(nb_dir / f"{slug}.ipynb"),
            "metadata_sha256": sha(nb_dir / "kernel-metadata.json")
        })

    # Verify status
    status_p = subprocess.run(["timeout", "45s", *cmd, "kernels", "status", f"{account}/{slug}/1"],
                              capture_output=True, text=True)
    status = status_from_cli(status_p.returncode, status_p.stdout)
    print(f"Initial status {account}/{slug}/1: {status}", flush=True)
    assert status in {"QUEUED", "RUNNING", "COMPLETE"}, f"Unexpected initial status: {status}"

    atomic_json(OUT / "launch-complete.json", {
        "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "account": account, "slug": slug, "ref": f"{account}/{slug}/1", "status": status
    })
    print("R8_RESUME_LAUNCH_DONE", flush=True)


if __name__ == "__main__":
    main()
