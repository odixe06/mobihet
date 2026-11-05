"""Freeze fresh20round CE recovery on the two now-terminal numerical probe owners."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/v-study/recovery-ce"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts"
CAP = 4.1


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not (OUT / "batch.json").exists(), "Inspect frozen recovery/push receipts before retry"
    smoke = json.loads((OUT / "preflight.json").read_text())
    assert len(smoke["cells"]) == 2 and all(c["weights_bn_rng_stats_bitexact"] and c["state_roundtrip"] and c["strict_rebuild"] for c in smoke["cells"])
    assert all(sha(ROOT / k) == h for k, h in smoke["source_sha256"].items())
    for name, n in (("tests.xml", 14), ("audit-tests.xml", 5)):
        suites = list(ET.parse(OUT / name).getroot().iter("testsuite"))
        assert sum(int(s.get("tests", 0)) for s in suites) == n
        assert all(int(s.get(k, 0)) == 0 for s in suites for k in ("failures", "errors", "skipped"))
    probes = json.loads((ROOT / "runs/v-study/numerics-guard/batch.json").read_text())
    report = json.loads((ROOT / "runs/v-study/numerics-guard/audit.json").read_text())
    assert len(report) == 2 and all(r["all_reported_finite"] and not r["zero_applied_client_ids"] and r["skipped_fraction"] <= .01 for r in report)
    for row in probes:
        folder = ROOT / "artifacts/v20-numerics" / row["slug"]
        entry = next(r for r in report if r["arm"] == row["arm"])
        assert all(sha(folder / k) == h for k, h in entry["files_sha256"].items())
    spec = importlib.util.spec_from_file_location("ka", SKILL / "kaggle_account.py")
    ka = importlib.util.module_from_spec(spec); spec.loader.exec_module(ka)
    quotas = [ka.account_quota(p["account"]) for p in probes]
    assert all(q["gpu_left"] >= CAP * 1.15 for q in quotas)
    (OUT / "quota.json").write_text(json.dumps({"accounts": quotas}, indent=2) + "\n")
    parents = json.loads((ROOT / "runs/v-study/batch.json").read_text())[:2]; batch = []
    for parent, probe in zip(parents, probes):
        assert probe["parent_arm"] == parent["arm"]
        owner = probe["account"]; seed = next(int(v.split("=", 1)[1]) for v in parent["hp"] if v.startswith("seed="))
        slug = f"mh-{parent['arm'].lower()}-val-real-s{seed}-guard1"
        overlay = parent["overlay_sha256"] | {n: sha(ROOT / n) for n in ("scripts/probe_v20_skips.py", "methods/mobihet_vrecover.py")}
        assert all(sha(ROOT / k) == h for k, h in overlay.items())
        args = shlex.join(["scripts/run_v20.py", "--method", "mobihet_vrecover", "--out", f"/kaggle/working/{slug}",
            "--width", "1", "--rounds", "50", "--until", "20", "--compile", "--wandb", "mobihet-v4",
            "--max-hours", "BUDGET", "--cache", "", "--set", *parent["hp"]])
        subprocess.run([sys.executable, str(ROOT / "scripts/gen_notebook.py"), "--owner", owner, "--slug", slug,
            "--sha", parent["base_sha"], "--max-hours", str(CAP), "--args", args, "--overlay", *overlay], check=True, capture_output=True, cwd=ROOT)
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
        nbpath.write_text(json.dumps(nb, indent=1) + "\n"); assert json.loads(meta.read_text())["is_private"] is True
        subprocess.run([sys.executable, str(SKILL / "embed_wandb_key.py"), str(nbpath), "--metadata", str(meta)], check=True, capture_output=True)
        for cell in json.loads(nbpath.read_text())["cells"]:
            if cell["cell_type"] == "code": compile("".join(cell["source"]), str(nbpath), "exec")
        batch.append(parent | {"arm": parent["arm"] + "R", "parent_arm": parent["arm"], "account": owner, "slug": slug,
            "version": 1, "overlay_sha256": overlay, "notebook_sha256": sha(nbpath), "metadata_sha256": sha(meta),
            "cap_hours": CAP, "restart_kind": "fresh-no-verified-checkpoint", "guard_policy": "common_global_skip_le_0.01_finite_positive_applied_per_client"})
    (OUT / "batch.json").write_text(json.dumps(batch, indent=2) + "\n")
    (ROOT / "docs/research/v20-ce-recovery-launch.json").write_text(json.dumps({"batch_sha256": sha(OUT / "batch.json"),
        "batch": batch, "preflight_sha256": sha(OUT / "preflight.json"), "quota_sha256": sha(OUT / "quota.json"),
        "probe_audit_sha256": sha(ROOT / "runs/v-study/numerics-guard/audit.json"),
        "scope": "Two fresh20round CE recoveries; core/HP unchanged, explicit1%global+finite+positive-applied guard; keep original failures"}, indent=2) + "\n")
    print("V20_CE_RECOVERY_NOTEBOOKS_LOCKED", flush=True)


if __name__ == "__main__": main()
