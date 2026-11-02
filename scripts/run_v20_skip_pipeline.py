"""Two spare owners: frozen one-epoch CE probes, 20min watch, pull and numerical audit."""
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch_v20 import owned_refs
from scripts.watch_kaggle_batch import atomic_json, status_from_cli

OUT = ROOT / "runs/v-study/numerics-guard"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
CAP = 1.2


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def scoped(owner):
    return [sys.executable, str(SKILL / "kaggle_as.py"), owner, "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]


def main():
    assert not (OUT / "batch.json").exists(), "Inspect immutable probe batch and push intents before retry"
    smoke = json.loads((OUT / "preflight.json").read_text())
    assert len(smoke["cells"]) == 2 and all(c["weights_bn_rng_stats_bitexact"] for c in smoke["cells"])
    assert all(sha(ROOT / k) == h for k, h in smoke["source_sha256"].items())
    suites = ET.parse(OUT / "tests.xml").getroot().findall("testsuite")
    assert sum(int(s.get("tests", 0)) for s in suites) == 2
    assert all(int(s.get(k, 0)) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    spec = importlib.util.spec_from_file_location("ka", SKILL / "kaggle_account.py")
    ka = importlib.util.module_from_spec(spec); spec.loader.exec_module(ka)
    before = ka.active_user()
    authorized = set(json.loads((ROOT / "runs/r-repeat/authorized-owners.json").read_text())["owners"])
    busy = {r["account"] for branch in ("r-study", "v-study", "r-repeat")
        for r in json.loads((ROOT / "runs" / branch / "batch.json").read_text())}
    owners = sorted(authorized - busy)
    assert len(owners) == 2 and set(owners) <= set(ka.known_users())
    parents = json.loads((ROOT / "runs/v-study/batch.json").read_text())[:2]
    quotas, gates = [], []
    for owner in owners:
        q = ka.account_quota(owner); assert q["gpu_left"] >= CAP * 1.15
        cmd = scoped(owner); refs = owned_refs(cmd, owner); statuses = []
        for ref in refs:
            p = subprocess.run(["timeout", "45s", *cmd, "kernels", "status", ref], capture_output=True, text=True)
            status = status_from_cli(p.returncode, p.stdout); statuses.append({"ref": ref, "status": status})
            assert status in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}, (owner, "Active or unknown owned session")
        p = subprocess.run(["timeout", "90s", *cmd, "datasets", "files", "-d", "odixe0502/veremi-nextgen-v4-scaled-per-vehicle"], capture_output=True, text=True)
        assert p.returncode == 0, (owner, "dataset access")
        quotas.append(q); gates.append({"owner": owner, "quota": q, "owned_refs": refs, "statuses": statuses})
    atomic_json(OUT / "quota.json", {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "accounts": quotas})
    atomic_json(OUT / "launch-gates.json", gates)
    batch = []
    for parent, owner in zip(parents, owners):
        seed = next(int(s.split("=", 1)[1]) for s in parent["hp"] if s.startswith("seed="))
        slug = f"mh-vnum-ce-s{seed}"; assert f"{owner}/{slug}" not in gates[len(batch)]["owned_refs"]
        overlay = parent["overlay_sha256"] | {"scripts/probe_v20_skips.py": sha(ROOT / "scripts/probe_v20_skips.py")}
        assert all(sha(ROOT / k) == h for k, h in overlay.items())
        args = shlex.join(["scripts/probe_v20_skips.py", "--out", f"/kaggle/working/{slug}", "--seed", str(seed),
            "--parent-arm", parent["arm"], "--max-hours", "BUDGET", "--cache", "", "--set", *parent["hp"]])
        subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", owner, "--slug", slug,
            "--sha", parent["base_sha"], "--max-hours", str(CAP), "--args", args, "--overlay", *overlay],
            check=True, capture_output=True, cwd=ROOT)
        folder = ROOT / "notebooks" / slug; nbpath = folder / (slug + ".ipynb"); meta = folder / "kernel-metadata.json"
        nb = json.loads(nbpath.read_text()); replaced = 0
        for cell in nb["cells"]:
            if cell["cell_type"] != "code": continue
            source = "".join(cell["source"]).replace('TORCHINDUCTOR_COMPILE_THREADS="2"', 'TORCHINDUCTOR_COMPILE_THREADS="1"')
            if 'D.build_npy(D.find_root(), "/kaggle/temp/v4npy")' in source:
                source = source.replace('D.build_npy(D.find_root(), "/kaggle/temp/v4npy")',
                    'from scripts.run_v20 import build_cache\nbuild_cache(D.find_root(), "/kaggle/temp/v4npy")'); replaced += 1
            cell["source"] = source.splitlines(keepends=True)
        assert replaced == 1
        nbpath.write_text(json.dumps(nb, indent=1) + "\n")
        metadata = json.loads(meta.read_text()); assert metadata["is_private"] is True and not metadata["kernel_sources"]
        subprocess.run([sys.executable, str(SKILL / "embed_wandb_key.py"), str(nbpath), "--metadata", str(meta)], check=True, capture_output=True)
        for cell in json.loads(nbpath.read_text())["cells"]:
            if cell["cell_type"] == "code": compile("".join(cell["source"]), str(nbpath), "exec")
        batch.append(parent | {"arm": f"VN{seed}", "parent_arm": parent["arm"], "account": owner, "slug": slug,
            "version": 1, "endpoint": 1, "cap_hours": CAP, "overlay_sha256": overlay,
            "notebook_sha256": sha(nbpath), "metadata_sha256": sha(meta), "diagnostic_only": True})
    atomic_json(OUT / "batch.json", batch)
    atomic_json(ROOT / "docs/research/v20-numerics-launch.json", {"batch_sha256": sha(OUT / "batch.json"), "batch": batch,
        "preflight_sha256": sha(OUT / "preflight.json"), "quota_sha256": sha(OUT / "quota.json"),
        "scope": "Two first-epoch numerical observations only; no accepted20round results, no threshold/HP/core change"})
    def push(row):
        folder = ROOT / "notebooks" / row["slug"]; ref = f"{row['account']}/{row['slug']}/1"
        assert sha(folder / (row["slug"] + ".ipynb")) == row["notebook_sha256"]
        with (OUT / (row["slug"] + ".push-intent.json")).open("x") as f: json.dump({"ref": ref}, f)
        p = subprocess.run(["timeout", "--foreground", "--kill-after=10s", "240s", *scoped(row["account"]), "kernels", "push", "-p", str(folder)], capture_output=True, text=True)
        log = OUT / (row["slug"] + ".push.log"); log.write_text(p.stdout + p.stderr); log.chmod(0o600)
        assert p.returncode == 0 and re.search(r"Kernel version 1 successfully pushed", p.stdout), "Ambiguous push: inspect intent/owned version before retry"
        atomic_json(OUT / (row["slug"] + ".push.json"), {"ref": ref, "notebook_sha256": row["notebook_sha256"]})
        print("V20_NUMERICS_PUSHED", ref, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: list(pool.map(push, batch))
    atomic_json(OUT / "launch-complete.json", {"runs": 2, "batch_sha256": sha(OUT / "batch.json"),
        "global_cli_before": before, "global_cli_after": ka.active_user()})
    subprocess.run([sys.executable, str(ROOT / "scripts/watch_kaggle_batch.py"), "--batch", str(OUT / "batch.json"),
        "--out", str(OUT / "watch"), "--interval", "1200"], check=True, cwd=ROOT)
    terminal = json.loads((OUT / "watch/terminal.json").read_text()); records = []
    for row, status in zip(batch, terminal["runs"]):
        dest = ROOT / "artifacts/v20-numerics" / row["slug"]; dest.mkdir(parents=True, exist_ok=True)
        with (OUT / (row["slug"] + ".collect.log")).open("w") as log:
            p = subprocess.run([*scoped(row["account"])[:4], sys.executable,
                str(ROOT / "scripts/run_bounded_python.py"), str(SKILL / "pull_kernel_output.py"),
                status["ref"], "-p", str(dest), "-j", "8", "--pattern", r'(^provenance/|/(fold_rank[01]\.json|numerics_rank[01]\.json)$)'], stdout=log, stderr=subprocess.STDOUT)
        assert p.returncode == 0
        records.append({"spec": row, "status": status, "root": str(dest)})
    atomic_json(OUT / "collected.json", records)
    subprocess.run([sys.executable, str(ROOT / "scripts/audit_v20_skips.py")], check=True, cwd=ROOT)
    print("V20_NUMERICS_PIPELINE_DONE", flush=True)


if __name__ == "__main__": main()
