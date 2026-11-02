"""Freeze P20 only after all scientific/preflight gates; never overwrite an existing batch."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.p20_common import CELLS, settings


def main():
    out = ROOT / "runs/p-study"
    assert not (out / "batch.json").exists(), "Frozen P batch already exists"
    suites = ET.parse(out / "tests.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) >= 83
    assert all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == 0 for s in suites)
    smoke = json.loads((out / "smoke-audit.json").read_text())
    assert smoke["source_unchanged_during_preflight"] and smoke["proxy_carrier_3round_bitexact"]
    assert smoke["same_bank_r1_bitexact"] and smoke["stream_controls_r1_bitexact"] and len(smoke["cells"]) == 16
    assert all(row["strict_rebuild"] and row["diagnostic_state_unchanged"] and row["eval_state_roundtrip"] for row in smoke["cells"])
    for name, digest in smoke["source_sha256"].items(): assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    prior = json.loads((out / "prior-evidence.json").read_text())
    assert all(prior["benchmark_byte_identical"].values())
    quota = json.loads((out / "quota.json").read_text())
    accounts = sorted(quota["accounts"], key=lambda q: -q["gpu_left"])
    assert len(accounts) == 16 and all(q["gpu_left"] >= 16.1 and not q["active_recent"] and not q["unknown_recent"] for q in accounts)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    assert subprocess.check_output(["git", "ls-remote", "origin", "refs/heads/main"], cwd=ROOT, text=True).split()[0] == sha
    parent = json.loads((ROOT / "runs/n-study/batch.json").read_text())[0]
    names = list(parent["overlay_sha256"]) + ["methods/mobihet_study.py", "methods/mobihet_study_wire.py", "methods/mobihet_study_diagnostics.py"]
    overlay = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}
    common = dict(s.split("=", 1) for s in parent["hp"])
    common.pop("synth_target", None)
    batch = []
    for cell, mode in enumerate(CELLS):
        for offset, seed in enumerate((42, 43)):
            arm = f"P{cell * 2 + offset}"
            slug = f"mh-{arm.lower()}-{mode.replace('_', '-')}-s{seed}"
            # First16 distinct owners, then second slot per owner; no account exceeds2 runs or planned14h.
            account = accounts[(cell * 2 + offset) % 16]["user"]
            config = common | {k: json.dumps(v, separators=(",", ":")) if not isinstance(v, str) else v for k, v in settings(mode).items()} | {"seed": str(seed)}
            hp = [f"{k}={v}" for k, v in config.items()]
            subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", account, "--slug", slug,
                "--method", "mobihet_study", "--width", "1", "--rounds", "50", "--until", "20", "--max-hours", "7",
                "--sha", sha, "--overlay", *overlay, "--set", *hp], check=True, cwd=ROOT, capture_output=True)
            folder = ROOT / "notebooks" / slug
            nb = folder / (slug + ".ipynb"); metadata = folder / "kernel-metadata.json"
            document = json.loads(nb.read_text())
            for block in document["cells"]:
                if block["cell_type"] == "code":
                    block["source"] = "".join(block["source"]).replace('TORCHINDUCTOR_COMPILE_THREADS="2"', 'TORCHINDUCTOR_COMPILE_THREADS="1"').splitlines(keepends=True)
            nb.write_text(json.dumps(document, indent=1) + "\n")
            subprocess.run([sys.executable, str(ROOT / ".agents/skills/kaggle-training-notebook/scripts/embed_wandb_key.py"),
                            str(nb), "--metadata", str(metadata)], check=True, capture_output=True)
            document, meta = json.loads(nb.read_text()), json.loads(metadata.read_text())
            assert meta["is_private"] is True and meta["machine_shape"] == "NvidiaTeslaT4" and meta["kernel_sources"] == []
            assert document["metadata"]["kaggle"]["isGpuEnabled"]
            for block in document["cells"]:
                if block["cell_type"] == "code": compile("".join(block["source"]), str(nb), "exec")
            batch.append({"arm": arm, "cell": f"C{cell}", "mode": mode, "account": account, "slug": slug, "version": 1,
                "hp": hp, "base_sha": sha, "source_sha256": overlay["methods/mobihet.py"], "overlay_sha256": overlay,
                "cap_hours": 7, "endpoint": 20, "lr_schedule_rounds": 50,
                "notebook_sha256": hashlib.sha256(nb.read_bytes()).hexdigest()})
    target = out / "batch.json"; target.write_text(json.dumps(batch, indent=2) + "\n")
    (ROOT / "docs/research/p20-launch.json").write_text(json.dumps({"batch_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "batch": batch, "prior_evidence_sha256": hashlib.sha256((out / "prior-evidence.json").read_bytes()).hexdigest(),
        "preflight_sha256": hashlib.sha256((out / "smoke-audit.json").read_bytes()).hexdigest(),
        "scope": "Frozen authorized32run batch; per-run push/terminal/audit receipts still required"}, indent=2) + "\n")
    print("P20_NOTEBOOKS_LOCKED", hashlib.sha256(target.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__": main()
