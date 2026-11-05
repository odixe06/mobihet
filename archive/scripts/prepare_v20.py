"""Freeze four clean-validation controls, paired seeds, only on owners outside active R."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.preflight_v20 import CELLS, settings

OUT = ROOT / "runs/v-study"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "batch.json").exists(), "Frozen V batch exists; inspect receipts"
    smoke = json.loads((OUT / "preflight.json").read_text()); split = json.loads((OUT / "split-audit.json").read_text())
    assert smoke["source_unchanged"] and len(smoke["cells"]) == 8
    assert all(c["state_roundtrip"] and c["strict_rebuild"] and c["train_only_class_counts_exact"] for c in smoke["cells"])
    for n, h in smoke["source_sha256"].items(): assert sha(ROOT / n) == h, n
    suites = ET.parse(OUT / "tests.xml").getroot().findall("testsuite")
    assert sum(int(s.attrib["tests"]) for s in suites) == 1
    assert all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == int(s.attrib.get("skipped", 0)) == 0 for s in suites)
    assert split["train_plus_val_counts_exact"] and split["all_clients_temporal_train_before_val"]
    for n, h in split["counts_sha256"].items(): assert sha(OUT / n) == h
    busy = {r["account"] for r in json.loads((ROOT / "runs/r-study/batch.json").read_text())}
    quota = json.loads((OUT / "quota.json").read_text())
    eligible = sorted([q for q in quota["accounts"] if q["user"] not in busy and not q["active_recent"]
        and not q["unknown_recent"]], key=lambda q: q["gpu_left"])
    assert len(eligible) == 8 and all(q["gpu_left"] >= (4.1 if i < 2 else 4.6) * 1.15 for i, q in enumerate(eligible)), "Whole V phase quota/slots insufficient"
    parent = json.loads((ROOT / "runs/q-study/batch.json").read_text())[0]
    overlay = parent["overlay_sha256"] | {n: sha(ROOT / n) for n in ("methods/mobihet_vstudy.py", "scripts/run_v20.py")}
    common = dict(item.split("=", 1) for item in parent["hp"])
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text()); batch = []
    for cell, mode in enumerate(CELLS):
        for offset, seed in enumerate((42, 43)):
            index = cell * 2 + offset; arm = f"V{index}"; owner = eligible[index]["user"]
            cap = 4.1 if mode == "real" else 4.6; slug = f"mh-{arm.lower()}-val-{mode.replace('_', '-')}-s{seed}"
            cfg = common | {k: v if isinstance(v, str) else json.dumps(v) for k, v in settings(mode).items()} | {"seed": str(seed)}
            hp = [f"{k}={v}" for k, v in cfg.items()]
            args = shlex.join(["scripts/run_v20.py", "--method", "mobihet_vstudy", "--out", f"/kaggle/working/{slug}",
                "--width", "1", "--rounds", "50", "--until", "20", "--compile", "--wandb", "mobihet-v4",
                "--max-hours", "BUDGET", "--cache", "", "--set", *hp])
            subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", owner, "--slug", slug,
                "--sha", parent["base_sha"], "--max-hours", str(cap), "--args", args, "--overlay", *overlay],
                cwd=ROOT, capture_output=True, check=True)
            folder = ROOT / "notebooks" / slug; nbpath = folder / (slug + ".ipynb"); metapath = folder / "kernel-metadata.json"
            nb = json.loads(nbpath.read_text()); replaced = 0
            for block in nb["cells"]:
                if block["cell_type"] != "code": continue
                source = "".join(block["source"]).replace('TORCHINDUCTOR_COMPILE_THREADS="2"', 'TORCHINDUCTOR_COMPILE_THREADS="1"')
                if 'D.build_npy(D.find_root(), "/kaggle/temp/v4npy")' in source:
                    source = source.replace('D.build_npy(D.find_root(), "/kaggle/temp/v4npy")',
                        'from scripts.run_v20 import build_cache\nbuild_cache(D.find_root(), "/kaggle/temp/v4npy")')
                    replaced += 1
                block["source"] = source.splitlines(keepends=True)
            assert replaced == 1, "No officialtest cache build in clean diagnostic"
            nbpath.write_text(json.dumps(nb, indent=1) + "\n")
            assert json.loads(metapath.read_text())["is_private"]
            subprocess.run([sys.executable, str(SKILL / "embed_wandb_key.py"), str(nbpath), "--metadata", str(metapath)],
                capture_output=True, check=True)
            nb, meta = json.loads(nbpath.read_text()), json.loads(metapath.read_text())
            assert meta["id"] == f"{owner}/{slug}" and meta["is_private"] and not meta["kernel_sources"]
            assert meta["docker_image"] == runtime["docker_image"] and meta["machine_shape"] == "NvidiaTeslaT4"
            assert nb["metadata"]["kaggle"]["isGpuEnabled"] and nbpath.stat().st_size < 1_000_000
            for block in nb["cells"]:
                if block["cell_type"] == "code": compile("".join(block["source"]), str(nbpath), "exec")
            batch.append({"arm": arm, "mode": mode, "account": owner, "slug": slug, "version": 1, "hp": hp,
                "base_sha": parent["base_sha"], "source_sha256": parent["source_sha256"], "overlay_sha256": overlay,
                "notebook_sha256": sha(nbpath), "metadata_sha256": sha(metapath), "cap_hours": cap,
                "endpoint": 20, "lr_schedule_rounds": 50, "split": "originalfoldtrain-to-foldval"})
    (OUT / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    (ROOT / "docs/research/v20-launch.json").write_text(json.dumps({"batch_sha256": sha(OUT / "batch.json"), "batch": batch,
        "preflight_sha256": sha(OUT / "preflight.json"), "quota_sha256": sha(OUT / "quota.json"),
        "split_audit_sha256": sha(OUT / "split-audit.json"), "scope": "Separate diagnostic branch; no headline test or selection change"}, indent=2) + "\n")
    print("V20_NOTEBOOKS_LOCKED", flush=True)


if __name__ == "__main__": main()
