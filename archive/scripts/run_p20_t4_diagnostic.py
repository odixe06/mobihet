"""One authorized numerical diagnostic: push once, wait locally, collect even on ERROR."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/p-study/numerics-t4"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"


def owned_refs(cmd, owner):
    refs = []
    for page in range(1, 101):
        result = subprocess.run(["timeout", "90s", *cmd, "kernels", "list", "--mine", "--page-size", "100",
                                 "--page", str(page), "--csv"], capture_output=True, text=True)
        assert result.returncode == 0, "Owned-kernel inventory unavailable"
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        if not rows: return refs
        assert all(row.get("ref", "").startswith(owner + "/") for row in rows), "Unexpected owned-kernel listing"
        refs.extend(row["ref"] for row in rows)
    raise RuntimeError("Owned-kernel inventory pagination did not finish")


def main():
    spec = json.loads((OUT / "batch.json").read_text())[0]
    module = importlib.util.spec_from_file_location("ka", SKILL / "kaggle_account.py")
    ka = importlib.util.module_from_spec(module)
    module.loader.exec_module(ka)
    assert ka.active_user() == "dinhan25"
    quota = ka.account_quota(spec["account"])
    (OUT / "quota.json").write_text(json.dumps(quota, indent=2) + "\n")
    assert quota["gpu_left"] >= spec["quota_planning_hours"] * 1.15
    folder = ROOT / "notebooks" / spec["slug"]
    notebook = folder / (spec["slug"] + ".ipynb")
    assert hashlib.sha256(notebook.read_bytes()).hexdigest() == spec["notebook_sha256"]
    for name, digest in spec["overlay_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    cmd = [sys.executable, str(SKILL / "kaggle_as.py"), spec["account"], "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]
    receipt, intent = OUT / "push.json", OUT / "push-intent.json"
    assert not receipt.exists() and not intent.exists(), "Existing/ambiguous push: inspect instead of retry"
    ref = f'{spec["account"]}/{spec["slug"]}/1'
    # This CLI reports permission-denied for a missing private slug, not an explicit404.
    refs = owned_refs(cmd, spec["account"])
    assert f'{spec["account"]}/mh-p21-proxy-hybrid-s43' in refs, "Existing P21 absent from owner inventory"
    assert f'{spec["account"]}/{spec["slug"]}' not in refs, "Probe slug already exists"
    (OUT / "owned-refs.json").write_text(json.dumps(refs, indent=2) + "\n")
    access = subprocess.run(["timeout", "90s", *cmd, "datasets", "files", "-d", "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"], capture_output=True, text=True)
    assert access.returncode == 0, "Probe dataset access unavailable"
    intent.write_text(json.dumps({"ref": ref, "notebook_sha256": spec["notebook_sha256"], "quota": quota}, indent=2) + "\n")
    result = subprocess.run(["timeout", "--foreground", "--kill-after=10s", "240s", *cmd, "kernels", "push", "-p", str(folder)], capture_output=True, text=True)
    (OUT / "push.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0 and re.search(r"Kernel version 1 successfully pushed", result.stdout), "Push failed/ambiguous: inspect, never blind retry"
    receipt.write_text(json.dumps({"ref": ref, "notebook_sha256": spec["notebook_sha256"]}, indent=2) + "\n")
    print("P20_T4_PROBE_PUSHED", ref, flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/watch_kaggle_batch.py"), "--batch", str(OUT / "batch.json"),
                    "--out", str(OUT / "watch"), "--interval", "1200"], check=True, cwd=ROOT)
    dest = OUT / "artifacts"
    pattern = r"(^provenance/|^p20-numerics-t4/.*\.jsonl?$)"
    subprocess.run([sys.executable, str(SKILL / "kaggle_as.py"), spec["account"], "--", sys.executable,
                    str(ROOT / "scripts/run_bounded_python.py"), str(SKILL / "pull_kernel_output.py"), ref,
                    "-p", str(dest), "-j", "8", "--pattern", pattern], check=True, cwd=ROOT)
    manifest = json.loads((dest / "provenance/source_manifest.json").read_text())
    assert manifest["overlay"] == spec["overlay_sha256"]
    for name, digest in manifest["overlay"].items():
        assert hashlib.sha256((dest / "provenance/source" / name).read_bytes()).hexdigest() == digest
    status = json.loads((OUT / "watch/terminal.json").read_text())["runs"][0]["status"]
    contexts = [dest / "p20-numerics-t4" / f"rank{rank}.json" for rank in (0, 1)]
    assert all(path.is_file() for path in contexts), "No complete rank startup evidence; inspect log"
    for path in contexts:
        context = json.loads(path.read_text())
        assert context["world"] == 2 and context["seed"] == 43 and context["capability"] == [7, 5]
    (OUT / "collection-audit.json").write_text(json.dumps({"status": status, "source_verified": True,
        "scope": "Diagnostic evidence collected, not successful20round training or F1 utility"}, indent=2) + "\n")
    print("P20_T4_EVIDENCE_COLLECTED", status, flush=True)


if __name__ == "__main__": main()
