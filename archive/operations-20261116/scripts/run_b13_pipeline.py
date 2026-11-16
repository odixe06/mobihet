"""Authorized module screen (SIM_BATCH=b13..b17): local gates -> quota -> push -> collect -> audit.

Run through codex-long-jobs. No model calls, automatic retries or training restarts.
Completion returns judgment/promotion to the owning Codex thread.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BATCH = os.environ.setdefault("SIM_BATCH", "b13")
assert BATCH in {"b13", "b14", "b15", "b16", "b17", "b20", "b21"}
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
from scripts import run_sim_kaggle as S  # noqa: E402
from scripts.watch_kaggle_batch import atomic_json  # noqa: E402

OUT = ROOT / f"runs/sim-kaggle-{BATCH}"


def run(args, log):
    print("STAGE", log, flush=True)
    with (OUT / log).open("w") as f:
        subprocess.run([sys.executable, *args], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    assert not (OUT / "launch-complete.json").exists(), "Existing batch: recover collection, never rerun launch"
    assert sys.argv[1:] in ([], ["--reuse-local-smoke"])
    reuse = bool(sys.argv[1:])
    source = S.OVERLAYS + ["scripts/run_sim_kaggle.py", "scripts/run_b13_pipeline.py", "scripts/analyse_b13.py",
                           "scripts/audit_sim_batch.py", "tests/test_mobihet_refine.py", "scripts/report_module_screen.py"]
    tests = ["tests/test_mobihet_refine.py", "tests/test_mobihet_teacher.py"]
    if BATCH == "b14":
        source += ["tests/test_mobihet_uplink.py", "scripts/audit_uplink.py", "tests/test_uplink_audit.py",
                   "tests/fixtures/uplink-fallback-r1.npz"]
        tests += ["tests/test_mobihet_uplink.py", "tests/test_uplink_audit.py"]
    if BATCH == "b15":
        source += ["tests/test_mobihet_sam.py", "scripts/check_sam_cuda.py", "scripts/audit_sam.py",
                   "docs/research/screening-sam-b15.md"]
        tests += ["tests/test_mobihet_sam.py"]
    if BATCH == "b16":
        source += ["tests/test_mobihet_mean.py", "scripts/audit_mean.py", "docs/research/screening-mean-b16.md"]
        tests += ["tests/test_mobihet_mean.py"]
    if BATCH == "b17":
        source += ["tests/test_mobihet_macro.py", "scripts/audit_macro.py", "docs/research/screening-macro-b17.md"]
        tests += ["tests/test_mobihet_macro.py"]
    if BATCH == "b20":
        source += ["tests/test_mobihet_compute.py", "tests/test_mobihet_macro.py", "docs/research/screening-compute-b20.md"]
        tests += ["tests/test_mobihet_compute.py"]
    if BATCH == "b21":
        source += ["tests/test_mobihet_compute.py", "tests/test_mobihet_macro.py", "docs/research/screening-dose-b21.md"]
        tests += ["tests/test_mobihet_compute.py"]
    digest = lambda: {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source}
    frozen = digest()
    if reuse:
        assert not any((OUT / name).exists() for name in ("batch.json", "quota.json")), "Only local-gate recovery here"
        assert not list(OUT.glob("*.push-intent.json")), "Inspect ambiguous pushes before recovery"
        previous = json.loads((OUT / "source-lock.json").read_text())
        assert all(previous[p] == frozen[p] for p in S.OVERLAYS), "Training source changed; old smoke cannot certify it"
    lock = OUT / ("source-lock-recovery.json" if reuse else "source-lock.json")
    assert not lock.exists(), "Preserve previous source lock; inspect before another attempt"
    atomic_json(lock, frozen)
    run(["-m", "pytest", *tests, "-q"], "unit-tests.log")
    if BATCH == "b15":
        run(["scripts/run_local_guarded.py", sys.executable, "scripts/check_sam_cuda.py"], "cuda-replay.log")
    smoke = OUT / "local-smoke"
    knobs = {"b13": ["refine_agg=sqrt_count", "refine_loss=js4", "real_smoothing=0.05"],
             "b14": ["same_class_mix=0.5", "uplink_mode=lg16", "uplink_audit=true"],
             "b15": ["sam_rho=0.05"], "b16": ["mean_kd=mean"], "b17": ["macro_mode=prior"],
             "b20": ["synth_grad=head"], "b21": []}[BATCH]
    hp = S.hp(42, S.ARMS[0][3] + knobs)
    if BATCH == "b20":
        hp = S.hp(42, S.ARMS[1][3])                                    # SH: the new loss path (SM shares it)
    if BATCH == "b21":
        hp = S.hp(42, S.ARMS[5][3])                                    # T4F: tier mask + floor 48
        assert S.ARMS[5][0] == "T4F"
    if BATCH in {"b16", "b17"}:
        assert not reuse, "This batch has two independent smoke gates; inspect failed stage before recovery"
        hp = S.hp(42, S.ARMS[1][3])
    if reuse:
        config = json.loads((smoke / "config.json").read_text())
        assert config["method"] == S.METHOD and config["hp"]["rounds"] == 20 and config["n_params"] == 395408
        for item in hp:
            key, value = item.split("=", 1)
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            assert config["hp"][key] == value, ("Smoke HP changed", key)
        print("REUSE_VERIFIED_SMOKE_NO_TRAINING", flush=True)
    else:
        assert not (smoke / "rounds.jsonl").exists(), "Preserve previous smoke evidence; inspect before retry"
        run(["scripts/run_local_guarded.py", sys.executable, "scripts/sim_fold_val.py", "run", "--n", "200",
             "--method", S.METHOD, "--out", str(smoke), "--rounds", "20", "--until", "2", "--compile",
             "--set", *hp], "local-smoke.log")
    import numpy as np
    import torch
    from core import metrics
    from scripts.sim_fold_val import fold_counts
    rows = [json.loads(x) for x in (smoke / "rounds.jsonl").read_text().splitlines() if x.strip()]
    assert [r["round"] for r in rows] == [1, 2]
    cnt = fold_counts()
    for row in rows:
        assert row["skipped_frac"] <= .01 and row["clients_without_applied"] == 0
        with np.load(smoke / f"cms_r{row['round']:03d}.npz") as z:
            assert np.array_equal(z["private"].sum(2), cnt[z["ids"], 1])
            ten = metrics.summary(z["private"])
            assert all(abs(ten[k] - row["private/" + k]) < 1e-9 for k in metrics.TEN)
    state = torch.load(smoke / "state.pt", map_location="cpu", weights_only=True)
    assert state["round"] == 2 and state["method"] == S.METHOD
    tensors = [v for v in state["method_state"].values() if torch.is_tensor(v)]
    assert tensors and all(bool(torch.isfinite(t).all()) for t in tensors)
    assert "compile gate rank0: PASS" in (OUT / "local-smoke.log").read_text()
    if BATCH == "b14":
        run(["scripts/audit_uplink.py", "--local", str(smoke)], "local-wire-audit.log")
    if BATCH == "b15":
        run(["scripts/audit_sam.py", "--local", str(smoke)], "local-sam-audit.log")
    extra_smoke_files = {}
    if BATCH == "b16":
        run(["scripts/audit_mean.py", "--local", str(smoke)], "local-mean-audit.log")
        half = OUT / "local-half-smoke"
        assert not (half / "rounds.jsonl").exists(), "Preserve old half smoke; inspect before retry"
        run(["scripts/run_local_guarded.py", sys.executable, "scripts/sim_fold_val.py", "run", "--n", "200",
             "--method", S.METHOD, "--out", str(half), "--rounds", "20", "--until", "2", "--compile",
             "--set", *S.hp(42, S.ARMS[2][3])], "local-half-smoke.log")
        assert "compile gate rank0: PASS" in (OUT / "local-half-smoke.log").read_text()
        run(["scripts/audit_mean.py", "--local", str(half)], "local-half-audit.log")
        extra_smoke_files = {str(p.relative_to(half)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                             for p in half.rglob("*") if p.is_file()}
    if BATCH == "b20":
        other = OUT / "local-mix-smoke"
        assert not (other / "rounds.jsonl").exists(), "Preserve previous smoke; inspect before retry"
        run(["scripts/run_local_guarded.py", sys.executable, "scripts/sim_fold_val.py", "run", "--n", "200",
             "--method", S.METHOD, "--out", str(other), "--rounds", "20", "--until", "2", "--compile",
             "--set", *S.hp(42, S.ARMS[2][3])], "local-mix-smoke.log")
        assert "compile gate rank0: PASS" in (OUT / "local-mix-smoke.log").read_text()
        extra_smoke_files = {str(p.relative_to(other)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                             for p in other.rglob("*") if p.is_file()}
    if BATCH == "b17":
        run(["scripts/audit_macro.py", "--local", str(smoke)], "local-prior-audit.log")
        other = OUT / "local-nckd-smoke"
        assert not (other / "rounds.jsonl").exists(), "Preserve previous smoke; inspect before retry"
        run(["scripts/run_local_guarded.py", sys.executable, "scripts/sim_fold_val.py", "run", "--n", "200",
             "--method", S.METHOD, "--out", str(other), "--rounds", "20", "--until", "2", "--compile",
             "--set", *S.hp(42, S.ARMS[2][3])], "local-nckd-smoke.log")
        assert "compile gate rank0: PASS" in (OUT / "local-nckd-smoke.log").read_text()
        run(["scripts/audit_macro.py", "--local", str(other)], "local-nckd-audit.log")
        extra_smoke_files = {str(p.relative_to(other)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                             for p in other.rglob("*") if p.is_file()}
    assert digest() == frozen, "Source changed during preflight"
    smoke_files = {str(p.relative_to(smoke)): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                   for p in smoke.rglob("*") if p.is_file()}
    atomic_json(OUT / "preflight.json", {"pass": True, "source_sha256": frozen, "rounds": 2, "clients": 200,
               "model": "DAGSNet W1", "gpu_scope": "one sm86 GPU, not two T4 equivalence",
               "unit_tests_log": "unit-tests.log", "smoke_log": "local-smoke.log", "reused_smoke": reuse,
               "source_lock": lock.name, "smoke_training_sources": {p: frozen[p] for p in S.OVERLAYS},
               "smoke_sha256": smoke_files, "extra_smoke_sha256": extra_smoke_files})
    print("LOCAL_PREFLIGHT_PASS", flush=True)
    # Drop audit tensors before network inventory/collection; no training state retained in worker RAM.
    del state, tensors
    run(["scripts/run_sim_kaggle.py", "quota"], "quota.log")
    assert digest() == frozen, "Source changed before notebook generation"
    batch = S.prepare(json.loads((OUT / "quota.json").read_text()))
    assert all(r["overlay_sha256"] == {p: frozen[p] for p in S.OVERLAYS} for r in batch)
    for spec in batch:
        notebook = ROOT / "notebooks" / spec["slug"] / (spec["slug"] + ".ipynb")
        assert notebook.stat().st_size < 600000
        doc = json.loads(notebook.read_text())
        for cell in doc["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(notebook), "exec")
        assert doc["metadata"]["kaggle"]["isGpuEnabled"]
        assert "BEGIN INLINE WANDB CREDENTIAL" not in notebook.read_text()
    assert digest() == frozen
    S.W.launch(batch)
    while True:
        run(["scripts/run_sim_kaggle.py", "collect"], "collect-last.log")
        states = json.loads((OUT / "status.json").read_text())
        print("KAGGLE_STATES", [(s["arm"], s["status"]) for s in states], flush=True)
        if all(s["terminal"] for s in states):
            break
        time.sleep(1200)
    assert digest() == frozen, "Source changed during remote batch; inspect before audit"
    run(["scripts/audit_sim_batch.py", BATCH, "20"], "audit.log")
    if BATCH == "b14":
        run(["scripts/audit_uplink.py"], "wire-audit.log")
    if BATCH == "b15":
        run(["scripts/audit_sam.py"], "sam-audit.log")
    if BATCH == "b16":
        run(["scripts/audit_mean.py"], "mean-audit.log")
    if BATCH == "b17":
        run(["scripts/audit_macro.py"], "macro-audit.log")
    run(["scripts/analyse_b13.py", BATCH], "analysis.log")
    run(["scripts/report_module_screen.py", BATCH], "report.log")
    atomic_json(OUT / "DONE", {"at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "audit_pass": True,
                              "next": "Owner Codex inspect verdict; PASS two seeds -> full-scale r20"})
    print(BATCH.upper() + "_AUDITED", flush=True)


if __name__ == "__main__":
    main()
