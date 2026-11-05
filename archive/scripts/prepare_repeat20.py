"""Freeze four predeclared R0-R3 repeated controls on newly authorized owners."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/r-repeat"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
CAP = 5.3


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not (OUT / "batch.json").exists(), "Immutable repeat phase exists; inspect receipts"
    smoke = json.loads((ROOT / "runs/r-study/preflight.json").read_text())
    assert smoke["source_unchanged"]
    repeats = [c for c in smoke["cells"] if c["kind"] == "repeat"]
    assert {(c["repeat"], c["seed"]) for c in repeats} == {(k, s) for k in ("A", "B") for s in (42, 43)}
    assert all(c["bitexact"] for c in repeats)
    for name, h in smoke["source_sha256"].items(): assert sha(ROOT / name) == h, name
    for name, count in (("tests.xml", 45), ("audit-tests.xml", 7)):
        path = ROOT / "runs/r-study" / name
        suites = ET.parse(path).getroot().findall("testsuite")
        assert sum(int(s.attrib["tests"]) for s in suites) == count
        assert all(int(s.attrib[k]) == 0 for s in suites for k in ("failures", "errors", "skipped"))
        shutil.copyfile(path, OUT / name)
    shutil.copyfile(ROOT / "runs/r-study/preflight.json", OUT / "preflight.json")
    quota = json.loads((OUT / "quota.json").read_text())
    authorized = set(json.loads((OUT / "authorized-owners.json").read_text())["owners"])
    busy = {r["account"] for branch in ("r-study", "v-study")
            for r in json.loads((ROOT / "runs" / branch / "batch.json").read_text())}
    eligible = sorted([q for q in quota["accounts"] if q["user"] in authorized - busy
        and not q["active_recent"] and not q["unknown_recent"] and q["gpu_left"] >= CAP * 1.15],
        key=lambda q: (-q["gpu_left"], q["user"]))
    assert len(eligible) >= 4, "Four full owner allocations required; do not reduce rounds/reserve or omit seed"
    parents = json.loads((ROOT / "runs/q-study/batch.json").read_text())[:2]
    runtime = json.loads((ROOT / "knowledge/runtime.json").read_text()); batch = []
    for index, (repeat, seed) in enumerate((k, s) for k in ("A", "B") for s in (42, 43)):
        parent = parents[seed - 42]
        assert parent["mode"] == "stream"
        overlay = parent["overlay_sha256"] | {"methods/mobihet_rstudy.py": sha(ROOT / "methods/mobihet_rstudy.py")}
        assert all(sha(ROOT / name) == h for name, h in overlay.items())
        owner = eligible[index]["user"]; arm = f"R{index}"; slug = f"mh-{arm.lower()}-repeat-{repeat.lower()}-s{seed}"
        hp = parent["hp"] + ["r_kind=repeat"]
        args = shlex.join(["scripts/run_q20.py", "--method", "mobihet_rstudy", "--out", f"/kaggle/working/{slug}",
            "--width", "1", "--rounds", "50", "--until", "20", "--compile", "--wandb", "mobihet-v4",
            "--max-hours", "BUDGET", "--cache", "", "--set", *hp])
        subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", owner, "--slug", slug,
            "--sha", parent["base_sha"], "--max-hours", str(CAP), "--args", args, "--overlay", *overlay],
            check=True, capture_output=True, cwd=ROOT)
        folder = ROOT / "notebooks" / slug; notebook = folder / (slug + ".ipynb"); metadata = folder / "kernel-metadata.json"
        nb = json.loads(notebook.read_text())
        for cell in nb["cells"]:
            if cell["cell_type"] == "code": cell["source"] = "".join(cell["source"]).replace(
                'TORCHINDUCTOR_COMPILE_THREADS="2"', 'TORCHINDUCTOR_COMPILE_THREADS="1"').splitlines(keepends=True)
        notebook.write_text(json.dumps(nb, indent=1) + "\n")
        assert json.loads(metadata.read_text())["is_private"] is True
        subprocess.run([sys.executable, str(SKILL / "embed_wandb_key.py"), str(notebook), "--metadata", str(metadata)],
            check=True, capture_output=True)
        nb, meta = json.loads(notebook.read_text()), json.loads(metadata.read_text())
        assert meta["is_private"] is True and meta["id"] == f"{owner}/{slug}" and meta["kernel_sources"] == []
        assert meta["machine_shape"] == runtime["machine_shape"] and meta["docker_image"] == runtime["docker_image"]
        assert nb["metadata"]["kaggle"]["isGpuEnabled"] and notebook.stat().st_size < 1_000_000
        for cell in nb["cells"]:
            if cell["cell_type"] == "code": compile("".join(cell["source"]), str(notebook), "exec")
        batch.append({"arm": arm, "mode": "stream", "repeat": repeat, "r_kind": "repeat", "account": owner,
            "slug": slug, "version": 1, "hp": hp, "base_sha": parent["base_sha"], "source_sha256": parent["source_sha256"],
            "overlay_sha256": overlay, "notebook_sha256": sha(notebook), "metadata_sha256": sha(metadata),
            "cap_hours": CAP, "endpoint": 20, "lr_schedule_rounds": 50, "reference_arm": parent["arm"]})
    (OUT / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    (ROOT / "docs/research/r20-repeat-launch.json").write_text(json.dumps({"batch_sha256": sha(OUT / "batch.json"),
        "batch": batch, "preflight_sha256": sha(OUT / "preflight.json"), "quota_sha256": sha(OUT / "quota.json"),
        "authorization": "Owner02Nov authorized newly saved accounts; complete predeclared R0-R3, preserve active R/V",
        "scope": "Four fresh stream controls, two repeats per seed; 20round/cosine50, no frozen teacher attachment or method change"}, indent=2) + "\n")
    print("REPEAT20_NOTEBOOKS_LOCKED", flush=True)


if __name__ == "__main__": main()
