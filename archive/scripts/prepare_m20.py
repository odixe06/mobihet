"""Freeze the authorized M20 diagnostic batch and validate its private notebooks; never launch it."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/m-study"
OVERLAY = ["methods/mobihet.py", "methods/mobihet_diagnostics.py", "methods/mobihet_teacher.py",
           "methods/mobihet_external_diagnostic.py", "scripts/run_fl.py"]
ACCOUNTS = ["nguyenminhtran060113", "thuyanh316", "minhtriethihi", "annhtnt", "takids1909", "annhannh331",
            "trietbackup", "khanhmay0304", "catbaochau", "odixeuit", "accanniez", "minhtran0601",
            "anhtruong1909", "dinhan25", "minhtrit06", "odixe0502"]
CELLS = [("aug", dict(no_kd=True, real_kd=False)),
         ("soft", dict(real_kd=False)),
         ("gate", dict(real_kd_gate=True)),
         ("fallback", dict(real_kd_gate=True, synth_target="fallback")),
         ("post", dict(real_kd_gate=True, q_collect="postepoch")),
         ("sqrt", dict(real_kd_gate=True, q_agg="sqrt_count")),
         ("uniform", dict(real_kd_gate=True, synth_target="fallback", kd_support="uniform")),
         ("local", dict(real_kd_gate=True, synth_target="fallback", kd_support="local"))]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not list(OUT.glob("*.push.json")), "Do not overwrite a launched manifest/notebook"
    q = json.loads((OUT / "quota.json").read_text())
    quota = {r["user"]: r for r in q["accounts"]}
    assert q["active"] == "dinhan25"
    assert all(quota[u]["gpu_left"] >= 6.9 and quota[u]["gpu_reserved"] == 0 for u in ACCOUNTS)
    ids = json.loads((ROOT / "runs/l-screening/diagnostic-clients.json").read_text())["ids"]
    assert len(ids) == 34 and len(set(ids)) == 34
    base = {"arm": "none", "group_rows": 4096, "synth": "copula", "synth_rows": 64, "q_codec": "mean_fp16",
            "kd_weight": "fixed", "diagnostics": True, "diagnostic_ids": ids,
            "diagnostic_rounds": [1, 2, 5, 10, 15, 20], "teacher_diagnostics": True, "external_diagnostics": True}
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {p: sha(ROOT / p) for p in OVERLAY}
    batch = []
    for name, delta in CELLS:
        for seed in (42, 43):
            i = len(batch)
            hp = dict(base, **delta, seed=seed)
            settings = [f"{k}={v if isinstance(v, str) else json.dumps(v, separators=(',', ':'))}" for k, v in hp.items()]
            slug = f"mh-m{i}-{name}-s{seed}"
            spec = {"arm": f"M{i}", "account": ACCOUNTS[i], "slug": slug, "version": 1,
                    "hp": settings, "base_sha": head, "source_sha256": hashes["methods/mobihet.py"],
                    "overlay_sha256": hashes, "cap_hours": 6, "endpoint": 20, "lr_schedule_rounds": 50}
            subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", spec["account"], "--slug", slug,
                            "--method", "mobihet", "--rounds", "50", "--until", "20", "--max-hours", "6",
                            "--overlay", *OVERLAY, "--set", *settings], cwd=ROOT, check=True)
            folder = ROOT / "notebooks" / slug
            nb, meta = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
            subprocess.run([sys.executable, ".agents/skills/kaggle-training-notebook/scripts/embed_wandb_key.py",
                            str(nb), "--metadata", str(meta)], cwd=ROOT, check=True)
            os.chmod(nb, 0o600)
            doc, metadata = json.loads(nb.read_text()), json.loads(meta.read_text())
            assert metadata["is_private"] is True and metadata["machine_shape"] == "NvidiaTeslaT4"
            assert metadata["id"] == spec["account"] + "/" + slug and metadata["kernel_sources"] == []
            assert metadata["dataset_sources"] == ["odixe0502/veremi-nextgen-v4-scaled-per-vehicle"]
            assert doc["metadata"]["kaggle"]["isGpuEnabled"] is True
            for cell in doc["cells"]:
                if cell["cell_type"] == "code":
                    compile("".join(cell["source"]), str(nb), "exec")
            assert hashes == {p: sha(ROOT / p) for p in OVERLAY}
            spec["notebook_sha256"] = sha(nb)
            batch.append(spec)
    (OUT / "batch.json").write_text(json.dumps(batch, indent=2))
    (OUT / "quota-plan.json").write_text(json.dumps({"at": q["at"], "total_cap_hours": 96,
        "sessions": [{"arm": s["arm"], "account": s["account"], "cap_hours": 6,
                      "free_hours": quota[s["account"]]["gpu_left"]} for s in batch]}, indent=2))
    print("M20:16 private notebooks validated; no kernels launched; credential values suppressed")


if __name__ == "__main__":
    main()
