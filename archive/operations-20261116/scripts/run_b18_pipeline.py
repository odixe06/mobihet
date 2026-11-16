"""B18 frozen local gates -> two private seed notebooks -> deterministic collect/audit."""
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_w20 as W
from scripts.watch_kaggle_batch import atomic_json, query

RECOVERY = os.environ.get("B18_RECOVERY") == "packaging1"
TAG = "b18r1" if RECOVERY else "b18"
OUT = ROOT / f"runs/sim-kaggle-{TAG}"
ART = ROOT / f"artifacts/sim-kaggle-{TAG}"
OVERLAYS = ["methods/mobihet_sim.py", "methods/mobihet_refine.py", "methods/mobihet_source.py",
            "scripts/sim_fold_val.py", "scripts/b18_partition.py", "scripts/b18_phase.py",
            "scripts/run_b18_seed.py", "scripts/audit_b18.py", "runs/sim/core_hp.txt"]
SOURCES = OVERLAYS + ["scripts/run_b18_pipeline.py", "scripts/check_b18_bundle.py",
                      "scripts/gen_notebook.py", "scripts/audit_b18_resources.py",
                      "tests/test_mobihet_source.py", "docs/research/screening-source-b18.md"]
CAP = 7.
W.OUT, W.ART, W.TAG = OUT, ART, "B18"
W.COLLECT = (r'(^provenance/|/(config\.json|rounds\.jsonl|cms_r\d{3}\.npz|partition\.(json|npz)|'
             r'SEED_DONE\.json|source-audit\.json|shadow\.log|[CEH]\.log)$|/source/rank[01]_r\d{3}\.npz$|/recovery/rank[01]\.jsonl$)')


def run(args, log):
    print("STAGE", log, flush=True)
    with (OUT / log).open("x") as f:
        subprocess.run([sys.executable, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def prepare(quota, frozen):
    available = [q for q in sorted(quota["accounts"], key=lambda q: (-q.get("gpu_left", 0), q["user"]))
                 if not q.get("inventory_error") and not q.get("unknown_recent") and not q.get("gpu_reserved", 0)
                 and len(q.get("active_recent", [])) < 2 and q.get("gpu_left", 0) >= CAP * 1.15]
    assert len(available) >= 2, "Need two owners with full seed budgets; do not cut rounds/seeds"
    head = W.git_head()
    builder = json.loads((ROOT / "runs/w-study/builder.json").read_text())
    result = []
    for seed, account in zip((42, 43), available):
        owner = account["user"]
        slug = f"mh-sim-{TAG}-source-s{seed}-r20"
        args = shlex.join(["scripts/run_b18_seed.py", "--out", f"/kaggle/working/{slug}", "--seed", str(seed),
                           "--world", "2", "--until", "20", "--max-hours", "BUDGET"])
        subprocess.run([sys.executable, "scripts/gen_notebook.py", "--owner", owner, "--slug", slug, "--sha", head,
                        "--run", slug, "--max-hours", str(CAP), "--args", args, "--npy-kernel", builder["ref"],
                        "--npy-manifest-sha256", builder["manifest_sha256"], "--overlay", *OVERLAYS],
                       cwd=ROOT, capture_output=True, check=True)
        folder = ROOT / "notebooks" / slug
        nb, mp = folder / (slug + ".ipynb"), folder / "kernel-metadata.json"
        doc = json.loads(nb.read_text())
        replaced = 0
        for cell in doc["cells"]:
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            if "BEGIN INLINE WANDB CREDENTIAL" in src:
                src = 'print("W&B off: B18 simulator")\n'
            needle = "&& torchrun --standalone --nproc_per_node=2 scripts/run_b18_seed.py"
            if needle in src:
                src = src.replace(needle, "&& python scripts/run_b18_seed.py")
                replaced += 1
            compile(src, str(nb), "exec")
            cell["source"] = src.splitlines(keepends=True)
        assert replaced == 1, "Notebook parent runs once; phase children use torchrun2"
        nb.write_text(json.dumps(doc, indent=1) + "\n")
        metadata = json.loads(mp.read_text())
        runtime = json.loads((ROOT / "knowledge/runtime.json").read_text())
        assert metadata["is_private"] and metadata["id"] == f"{owner}/{slug}"
        assert metadata["kernel_sources"] == [builder["ref"]] and metadata["dataset_sources"] == ["odixe0502/veremi-nextgen-v4-scaled-per-vehicle"]
        assert all(metadata[k] == runtime[k] for k in ("machine_shape", "docker_image"))
        assert doc["metadata"]["kaggle"]["isGpuEnabled"] and nb.stat().st_size < 600000
        result.append(dict(arm=f"S{seed}", seed=seed, account=owner, slug=slug, version=1, cap_hours=CAP,
                           base_sha=head, overlay_sha256={p: frozen[p] for p in OVERLAYS},
                           notebook_sha256=W.sha(nb), metadata_sha256=W.sha(mp),
                           endpoint=20, lr_schedule_rounds=20, phases=["shadow", "C", "E", "H"],
                           npy_kernel=builder["ref"], npy_manifest_sha256=builder["manifest_sha256"]))
    atomic_json(OUT / "batch.json", result)
    return result


def analyse(batch):
    from scripts.audit_b18 import audit
    from scripts.audit_b18_resources import audit as audit_resources
    from core import metrics
    from scripts.research_metrics import client_f1
    from scripts.sim_fold_val import fold_counts
    import numpy as np
    counts = fold_counts()
    report = {"seeds": {}, "scope": "r20 macro first; other metrics report-only; E mechanistic control"}
    lines = ["# B18 matched-source result", "", "| seed | phase | macro F1 | ΔvsC | ΔvsE | bytes up+down/round incl shadow |", "|---:|---|---:|---:|---:|---:|"]
    partition = None
    passed = []
    for spec in batch:
        base = ART / spec["slug"]
        assert (base / "pulled.ok").exists()
        resource_proof = audit_resources(base, spec, ROOT / "notebooks" / spec["slug"] / (spec["slug"] + ".ipynb"))
        atomic_json(OUT / f"resource-audit-s{spec['seed']}.json", resource_proof)
        root = base / spec["slug"]
        done = json.loads((root / "SEED_DONE.json").read_text())
        assert done["rounds"] == 20 and done["world"] == 2 and done["seed"] == spec["seed"]
        proof = audit(root, 20, 2)
        if partition is None:
            partition = proof["partition_sha256"]
        assert partition == proof["partition_sha256"], "Same data/probes across seeds"
        stats = {}
        for phase in ("C", "E", "H"):
            ten = proof["endpoint_ten"][phase]
            rows = [json.loads(s) for s in (root / phase / "rounds.jsonl").read_text().splitlines()]
            with np.load(root / phase / "cms_r020.npz") as z:
                cm, ids = z["private"], z["ids"]
            pooled = cm.sum(0)
            f = 2 * np.diag(pooled) / np.maximum(pooled.sum(0) + pooled.sum(1), 1)
            cf = client_f1(cm)
            absent = counts[ids, 0] == 0
            stats[phase] = {"ten": ten, "p10": float(np.quantile(cf, .1)),
                            "low256": float(cf[counts[ids, 0].sum(1) < 256].mean()),
                            "f1_per_class": f.tolist(), "absent_rows": int((cm.sum(2) * absent).sum()),
                            "absent_accuracy": float((cm.diagonal(axis1=1, axis2=2) * absent).sum() / max((cm.sum(2) * absent).sum(), 1)),
                            "mean16_20_macro": float(np.mean([r["private/f1_macro"] for r in rows[-5:]])),
                            "bytes_round": float(np.mean([r["bytes_up"] + r["bytes_down"] for r in rows])),
                            "main_train_seconds": sum(r["t_train"] for r in rows)}
        dc = float(stats["H"]["ten"]["f1_macro"] - stats["C"]["ten"]["f1_macro"])
        de = float(stats["H"]["ten"]["f1_macro"] - stats["E"]["ten"]["f1_macro"])
        ok = bool(dc >= .002 and de > 0)
        passed.append(ok)
        report["seeds"][str(spec["seed"])] = {"stats": stats, "H_delta_C": dc, "H_delta_E": de, "pass": ok}
        for phase, s in stats.items():
            macro = s["ten"]["f1_macro"]
            lines.append(f"| {spec['seed']} | {phase} | {macro:.8f} | {macro - stats['C']['ten']['f1_macro']:+.8f} | {macro - stats['E']['ten']['f1_macro']:+.8f} | {s['bytes_round']:.2f} |")
    report["candidate_H_pass"] = all(passed)
    atomic_json(OUT / "verdict-r20.json", report)
    lines += ["", "## All ten metrics at r20", "",
              "| seed | phase | " + " | ".join(metrics.TEN) + " |",
              "|---:|---|" + "|".join(["---:"] * len(metrics.TEN)) + "|"]
    with (OUT / "metrics-r20.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed", "phase", *metrics.TEN])
        for seed, result in report["seeds"].items():
            for phase, s in result["stats"].items():
                values = [s["ten"][k] for k in metrics.TEN]
                writer.writerow([seed, phase, *values])
                lines.append("| " + seed + " | " + phase + " | " + " | ".join(f"{v:.8f}" for v in values) + " |")
    (OUT / "report-r20.md").write_text("\n".join(lines) + "\n\nClass/guard metrics: verdict-r20.json. No best-round selection.\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {p: W.sha(ROOT / p) for p in SOURCES}
    if sys.argv[1:] == ["--collect-only"]:
        assert (OUT / "launch-complete.json").exists()
        assert frozen == json.loads((OUT / "source-lock.json").read_text())
        batch = json.loads((OUT / "batch.json").read_text())
    else:
        assert sys.argv[1:] == [] and not (OUT / "source-lock.json").exists(), "Inspect existing work before retry"
        atomic_json(OUT / "source-lock.json", frozen)
        run(["-m", "pytest", "tests/test_mobihet_source.py", "tests/test_mobihet_refine.py", "tests/test_mobihet_teacher.py", "-q"], "unit-tests.log")
        run(["scripts/check_b18_bundle.py"], "deployment-import.log")
        if RECOVERY:
            # Packaging-only fix: re-audit existing classifier output, never retrain
            # simply to certify a resource added to the notebook overlay.
            previous = ROOT / "runs/sim-kaggle-b18"
            old = json.loads((previous / "source-lock.json").read_text())
            assert all(W.sha(ROOT / p) == old[p] for p in OVERLAYS if p in old)
            smoke = previous / "local-smoke"
            from scripts.b18_phase import CBB_HP
            for phase in ("shadow", "C", "E", "H"):
                cfg = json.loads((smoke / phase / "config.json").read_text())
                for item in CBB_HP:
                    key, value = item.split("=", 1)
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        pass
                    assert cfg["hp"][key] == value, ("Smoke HP mismatch", phase, key)
            run(["scripts/audit_b18.py", "--root", str(smoke), "--rounds", "2", "--world", "1"], "reused-smoke-audit.log")
        else:
            smoke = OUT / "local-smoke"
            run(["scripts/run_local_guarded.py", sys.executable, "scripts/run_b18_seed.py", "--out", str(smoke),
                 "--seed", "42", "--world", "1", "--until", "2", "--max-hours", ".8"], "local-smoke.log")
        proof = json.loads((smoke / "source-audit.json").read_text())
        assert proof["pass"] and proof["rounds"] == 2 and proof["world"] == 1
        assert {p: W.sha(ROOT / p) for p in SOURCES} == frozen
        atomic_json(OUT / "preflight.json", {"pass": True, "source_sha256": frozen,
                    "reused_local_classifier": RECOVERY, "smoke_path": str(smoke),
                    "scope": "DAGSNetW1/200clients/all4phases/2rounds/compiledAMP/actualpacketCMaudit; one sm86GPU, not2T4"})
        quota = W.inventory()
        batch = prepare(quota, frozen)
        assert {p: W.sha(ROOT / p) for p in SOURCES} == frozen
        W.launch(batch)
    (OUT / "collection").mkdir(exist_ok=True)
    while True:
        assert {p: W.sha(ROOT / p) for p in SOURCES} == frozen
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            states = list(pool.map(query, batch))
        atomic_json(OUT / "status.json", states)
        for spec, state in zip(batch, states):
            print("B18_STATE", state["ref"], state["status"], flush=True)
            marker = ART / spec["slug"] / "pulled.ok"
            if state["terminal"] and not marker.exists():
                assert W.collect(spec) == 0, "Collection failed; inspect retained partial download"
                marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        if any(s["status"] in {"ERROR", "CANCEL_ACKNOWLEDGED"} for s in states):
            raise RuntimeError("Terminal remote error collected; inspect artifacts, leave other seed running")
        if all(s["terminal"] for s in states):
            break
        time.sleep(1200)
    analyse(batch)
    atomic_json(OUT / "DONE", {"audit_pass": True, "next": "Owner agent inspect H verdict; PASS two seeds -> lock fullscale pilot"})
    print("B18_AUDITED", flush=True)


if __name__ == "__main__":
    main()
