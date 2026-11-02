"""Lock O20 after real preflight and tests; embed exact replay source over an already-pushed base."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from methods.mobihet_replay import frozen, permutation


def main():
    out = ROOT / "runs/o-study"
    assert not (out / "batch.json").exists(), "Frozen batch exists; inspect before regenerating"
    suites = ET.parse(out / "tests.xml").getroot().findall("testsuite")
    assert suites and sum(int(s.attrib["tests"]) for s in suites) >= 61
    assert all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == 0 for s in suites)
    smoke = json.loads((out / "smoke-audit.json").read_text())
    assert smoke["manual_injection_bitexact"] and smoke["r1_four_cells_bitexact"] and len(smoke["cells"]) == 4
    assert smoke["source_sha256"] == frozen.SHA256
    assert smoke["replay_code_sha256"] == hashlib.sha256((ROOT / "methods/mobihet_replay.py").read_bytes()).hexdigest()
    quota = json.loads((out / "quota.json").read_text())
    accounts = quota["accounts"]
    assert len(accounts) == 8 and all(q["gpu_left"] >= 6.9 and q["gpu_reserved"] == 0 for q in accounts)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, text=True).split()[0] == sha
    parent = json.loads((ROOT / "runs/n-study/batch.json").read_text())[0]
    overlay = dict(parent["overlay_sha256"])
    for path, digest in overlay.items(): assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path in ("methods/mobihet_replay.py", "methods/mobihet_replay_data.py"):
        overlay[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    permsha = hashlib.sha256(permutation().astype("<i8").tobytes()).hexdigest()
    common = dict(s.split("=", 1) for s in parent["hp"])
    batch = []
    for cell, (real, mode) in enumerate(((False, "full"), (True, "full"), (False, "uniform"), (False, "permuted"))):
        for offset, seed in enumerate((42, 43)):
            i = 2 * cell + offset
            slug = f"mh-o{i}-" + ("gate" if real else "noreal") + f"-{mode}-s{seed}"
            settings = common | {"seed": str(seed), "real_kd": str(real).lower(), "real_kd_gate": str(real).lower(),
                "synth_target": "soft", "replay_mode": mode, "replay_source_sha256": frozen.SHA256}
            hp = [f"{k}={v}" for k, v in settings.items()]
            account = accounts[i]["user"]
            subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", account, "--slug", slug,
                "--method", "mobihet_replay", "--width", "1", "--rounds", "50", "--until", "20", "--max-hours", "6",
                "--sha", sha, "--overlay", *overlay, "--set", *hp], check=True, cwd=ROOT, capture_output=True)
            folder = ROOT / "notebooks" / slug; nb = folder / (slug + ".ipynb"); metadata = folder / "kernel-metadata.json"
            subprocess.run([sys.executable, str(ROOT / ".agents/skills/kaggle-training-notebook/scripts/embed_wandb_key.py"),
                            str(nb), "--metadata", str(metadata)], check=True, capture_output=True)
            document, meta = json.loads(nb.read_text()), json.loads(metadata.read_text())
            assert meta["is_private"] is True and meta["machine_shape"] == "NvidiaTeslaT4"
            assert document["metadata"]["kaggle"]["isGpuEnabled"] and meta["kernel_sources"] == []
            for block in document["cells"]:
                if block["cell_type"] == "code": compile("".join(block["source"]), str(nb), "exec")
            batch.append({"arm": f"O{i}", "cell": f"R{cell}", "account": account, "slug": slug, "version": 1,
                "hp": hp, "base_sha": sha, "source_sha256": overlay["methods/mobihet.py"], "overlay_sha256": overlay,
                "replay_source_sha256": frozen.SHA256, "permutation_sha256": permsha,
                "replay_setup_module_bytes": (ROOT / "methods/mobihet_replay_data.py").stat().st_size,
                "cap_hours": 6, "endpoint": 20, "lr_schedule_rounds": 50,
                "notebook_sha256": hashlib.sha256(nb.read_bytes()).hexdigest()})
    target = out / "batch.json"; assert not target.exists()
    target.write_text(json.dumps(batch, indent=2) + "\n")
    (ROOT / "docs/research/o20-launch.json").write_text(json.dumps({"batch_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "batch": batch, "quota": quota, "preflight": {k: v for k, v in smoke.items() if k != "cells"},
        "scope": "Frozen authorized batch; per-account push receipts are separate, not a claim all sessions started"}, indent=2) + "\n")
    print("O20_NOTEBOOKS_LOCKED", hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__": main()
