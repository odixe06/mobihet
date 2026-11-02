"""Freeze Q20 after both-seed real preflight and full-batch quota allocation; no launch."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.q20_common import CELLS, settings

OUT = ROOT / "runs/q-study"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
CAP = 6


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "batch.json").exists(), "Q20 already frozen; inspect receipts instead of overwrite"
    smoke = json.loads((OUT / "preflight.json").read_text())
    assert smoke["source_unchanged"] and smoke["reference_and_query_3round_bitexact_both_seeds"] and smoke["mixed_first_round_bitexact_both_seeds"]
    assert len(smoke["cells"]) == 16 and all(c["strict_rebuild"] and c["state_roundtrip"] and c["diagnostic_state_rng_unchanged"] for c in smoke["cells"])
    for name, h in smoke["source_sha256"].items(): assert sha(ROOT / name) == h, name
    suites = ET.parse(OUT / "tests.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) >= 35 and all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == 0 for s in suites)
    quota = json.loads((OUT / "quota.json").read_text())
    eligible = sorted([q for q in quota["accounts"] if q["user"] != "accanniez" and not q["active_recent"]
                       and not q["unknown_recent"] and q["gpu_left"] >= CAP * 1.15], key=lambda q: -q["gpu_left"])
    # One slot per owner first; only then consume an owner's second fully budgeted slot.
    owners = [q["user"] for q in eligible] + [q["user"] for q in eligible if q["gpu_left"] >= 2 * CAP * 1.15]
    assert len(owners) >= 16, f"Whole16run batch needs {16 * CAP * 1.15}h and valid slots; available {len(owners)} slots"
    owners = owners[:16]
    parent = json.loads((ROOT / "runs/p-study/batch.json").read_text())[4]
    overlay = parent["overlay_sha256"] | {name: sha(ROOT / name) for name in
        ["scripts/run_p21_recovery.py", "methods/mobihet_next.py", "scripts/run_q20.py"]}
    for name, h in parent["overlay_sha256"].items(): assert sha(ROOT / name) == h
    common = dict(item.split("=", 1) for item in parent["hp"])
    batch = []
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
    for cell, mode in enumerate(CELLS):
        for offset, seed in enumerate((42, 43)):
            arm = f"Q{cell * 2 + offset}"; account = owners[cell * 2 + offset]
            slug = f"mh-{arm.lower()}-{mode.replace('_', '-')}-s{seed}"
            cfg = common | {k: json.dumps(v, separators=(",", ":")) if not isinstance(v, str) else v for k, v in settings(mode).items()} | {"seed": str(seed)}
            hp = [f"{k}={v}" for k, v in cfg.items()]
            args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_next", "--out", f"/kaggle/working/{slug}",
                "--width", "1", "--rounds", "50", "--until", "20", "--compile", "--wandb", "mobihet-v4",
                "--max-hours", "BUDGET", "--cache", "", "--set", *hp])
            subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", account, "--slug", slug,
                "--sha", parent["base_sha"], "--max-hours", str(CAP), "--args", args, "--overlay", *overlay],
                check=True, capture_output=True, cwd=ROOT)
            folder = ROOT / "notebooks" / slug; notebook = folder / (slug + ".ipynb"); metadata = folder / "kernel-metadata.json"
            nb = json.loads(notebook.read_text())
            for block in nb["cells"]:
                if block["cell_type"] == "code": block["source"] = "".join(block["source"]).replace(
                    'TORCHINDUCTOR_COMPILE_THREADS="2"', 'TORCHINDUCTOR_COMPILE_THREADS="1"').splitlines(keepends=True)
            notebook.write_text(json.dumps(nb, indent=1) + "\n")
            assert json.loads(metadata.read_text())["is_private"] is True
            subprocess.run([sys.executable, str(SKILL / "embed_wandb_key.py"), str(notebook), "--metadata", str(metadata)],
                           check=True, capture_output=True)
            nb, meta = json.loads(notebook.read_text()), json.loads(metadata.read_text())
            assert meta["is_private"] is True and meta["id"] == f"{account}/{slug}" and meta["kernel_sources"] == []
            assert meta["machine_shape"] == runtime["machine_shape"] and meta["docker_image"] == runtime["docker_image"]
            assert nb["metadata"]["kaggle"]["isGpuEnabled"]
            for block in nb["cells"]:
                if block["cell_type"] == "code": compile("".join(block["source"]), str(notebook), "exec")
            batch.append({"arm": arm, "mode": mode, "account": account, "slug": slug, "version": 1, "hp": hp,
                "base_sha": parent["base_sha"], "source_sha256": parent["source_sha256"], "overlay_sha256": overlay,
                "notebook_sha256": sha(notebook), "metadata_sha256": sha(metadata), "cap_hours": CAP,
                "endpoint": 20, "lr_schedule_rounds": 50})
    (OUT / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    (ROOT / "docs/research/q20-launch.json").write_text(json.dumps({"batch_sha256": sha(OUT / "batch.json"),
        "batch": batch, "preflight_sha256": sha(OUT / "preflight.json"), "quota_sha256": sha(OUT / "quota.json"),
        "scope": "Authorized16run Q20 frozen design; push/terminal/full20round audit receipts required"}, indent=2) + "\n")
    print("Q20_NOTEBOOKS_LOCKED", flush=True)


if __name__ == "__main__": main()
