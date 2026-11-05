"""Freeze the eight common-teacher replay arms after real preflight and live quota."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/r-study"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
CAP = 5.3


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "batch.json").exists(), "Frozen batch exists; inspect receipts"
    smoke = json.loads((OUT / "preflight.json").read_text())
    assert smoke["source_unchanged"] and smoke["cuda_frozen_decode_exact_both_seeds"] and smoke["replay_first_round_bitexact"]
    assert len(smoke["cells"]) == 12 and all(c.get("bitexact", c.get("manual_injection_bitexact")) for c in smoke["cells"])
    for name, h in smoke["source_sha256"].items(): assert sha(ROOT / name) == h, name
    suites = ET.parse(OUT / "tests.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) >= 45
    assert all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == int(s.attrib.get("skipped", 0)) == 0 for s in suites)
    quota = json.loads((OUT / "quota.json").read_text())
    eligible = sorted([q for q in quota["accounts"] if not q["active_recent"] and not q["unknown_recent"]
        and q["gpu_left"] >= CAP * 1.15], key=lambda q: -q["gpu_left"])
    assert len(eligible) >= 8, "Need eight complete owner allocations, no round/reserve reduction"
    parent = json.loads((ROOT / "runs/q-study/batch.json").read_text())[6]
    overlay = parent["overlay_sha256"] | {name: sha(ROOT / name) for name in
        ("methods/mobihet_rstudy.py", "methods/mobihet_rstudy_data.py")}
    common = dict(item.split("=", 1) for item in parent["hp"])
    from methods.mobihet_rstudy_data import SHA256
    from scripts.q20_common import settings
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text()); batch = []
    recovery = (OUT / "push-recovery/recovery.json").exists()
    for cell, mode in enumerate(("mixed_input", "real", "margin", "strength")):
        for offset, seed in enumerate((42, 43)):
            index = cell * 2 + offset; arm = f"R{4 + index}"; account = eligible[index]["user"]
            slug = f"mh-{arm.lower()}-replay-{mode.replace('_', '-')}-s{seed}"
            if recovery: slug += "-input1"
            cfg = common | {k: v if isinstance(v, str) else json.dumps(v) for k, v in settings(mode).items()} | {
                "seed": str(seed), "r_kind": "replay", "r_source_sha256": SHA256}
            hp = [f"{k}={v}" for k, v in cfg.items()]
            args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_rstudy", "--out", f"/kaggle/working/{slug}",
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
            if recovery:
                ready = json.loads((OUT / "input-datasets" / (account + ".ready.json")).read_text())
                assert ready["private"] and ready["download_sha_verified"] and ready["decoded_sha256"] == SHA256
                meta = json.loads(metadata.read_text()); meta["dataset_sources"].append(ready["ref"])
                metadata.write_text(json.dumps(meta, indent=1) + "\n")
                for block in nb["cells"]:
                    source = "".join(block["source"])
                    if block["cell_type"] == "code" and "D.build_npy(" in source:
                        block["source"] = source.replace('from core import data as D',
                            'from core import data as D\nfrom methods.mobihet_rstudy import load_schedule\n'
                            + f'_history = load_schedule({seed}, "cuda:0", "{SHA256}")\ndel _history').splitlines(keepends=True)
            notebook.write_text(json.dumps(nb, indent=1) + "\n")
            assert json.loads(metadata.read_text())["is_private"] is True
            subprocess.run([sys.executable, str(SKILL / "embed_wandb_key.py"), str(notebook), "--metadata", str(metadata)],
                check=True, capture_output=True)
            nb, meta = json.loads(notebook.read_text()), json.loads(metadata.read_text())
            assert meta["is_private"] and meta["id"] == f"{account}/{slug}" and not meta["kernel_sources"]
            assert meta["docker_image"] == runtime["docker_image"] and meta["machine_shape"] == "NvidiaTeslaT4"
            assert nb["metadata"]["kaggle"]["isGpuEnabled"]
            assert notebook.stat().st_size < 1_000_000, "Keep bulk replay inputs outside notebook code"
            for block in nb["cells"]:
                if block["cell_type"] == "code": compile("".join(block["source"]), str(notebook), "exec")
            batch.append({"arm": arm, "mode": mode, "r_kind": "replay", "account": account, "slug": slug, "version": 1,
                "hp": hp, "base_sha": parent["base_sha"], "source_sha256": parent["source_sha256"], "overlay_sha256": overlay,
                "cap_hours": CAP, "endpoint": 20, "lr_schedule_rounds": 50, "notebook_sha256": sha(notebook),
                "metadata_sha256": sha(metadata), "replay_source_sha256": SHA256})
            if recovery: batch[-1]["replay_input"] = ready
    (OUT / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    manifest = "r20-input1-launch.json" if recovery else "r20-launch.json"
    (ROOT / "docs/research" / manifest).write_text(json.dumps({"batch_sha256": sha(OUT / "batch.json"),
        "batch": batch, "preflight_sha256": sha(OUT / "preflight.json"), "quota_sha256": sha(OUT / "quota.json"),
        "scope": "Eight replay arms only; repeats/validation not launched, separate quota/split gate required"}, indent=2) + "\n")
    print("R20_NOTEBOOKS_LOCKED", flush=True)


if __name__ == "__main__": main()
