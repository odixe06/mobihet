"""Authorized deterministic stages: unchanged-teacher preflight -> private inputs -> immutable push -> collect/audit."""
import concurrent.futures
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/r-study"
SKILL = ROOT / ".agents/skills/kaggle-training-notebook/scripts/kaggle_as.py"


def run(*args): subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--reuse-verified-preflight", action="store_true")
    args = parser.parse_args()
    receipt = json.loads((OUT / "push-recovery/recovery.json").read_text())
    assert receipt["no_remote_kernel_created"] and len(receipt["inventory"]) == 8
    assert not (OUT / "launch-complete.json").exists(), "Already launched: use collect-only instead"
    if args.reuse_verified_preflight:
        import xml.etree.ElementTree as ET
        smoke = json.loads((OUT / "preflight.json").read_text())
        assert smoke["source_unchanged"] and smoke["cuda_frozen_decode_exact_both_seeds"] and smoke["replay_first_round_bitexact"]
        assert len(smoke["cells"]) == 12 and all(c.get("bitexact", c.get("manual_injection_bitexact")) for c in smoke["cells"])
        for name, sha in smoke["source_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
        for file, count in (("tests.xml", 45), ("audit-tests.xml", 7)):
            suites = ET.parse(OUT / file).getroot().findall("testsuite")
            assert sum(int(s.attrib["tests"]) for s in suites) == count
            assert all(int(s.attrib["failures"]) == int(s.attrib["errors"]) == int(s.attrib.get("skipped", 0)) == 0 for s in suites)
        print("R20_EXISTING_PREFLIGHT_SOURCE_REVERIFIED", flush=True)
    else:
        run("scripts/run_local_guarded.py", sys.executable, "scripts/preflight_r20.py")
        run("-m", "pytest", "-q", "tests/test_r20_audit.py", "--junitxml=" + str(OUT / "audit-tests.xml"))
    previous = json.loads((OUT / "push-recovery/rejected-batch.json").read_text())
    def stage(spec):
        with (OUT / "push-recovery" / (spec["account"] + ".stage.log")).open("a") as log:
            log.write("R20_PRIVATE_INPUT_VERIFICATION_STAGE\n"); log.flush()
            p = subprocess.run([sys.executable, str(SKILL), spec["account"], "--", sys.executable,
                str(ROOT / "scripts/run_bounded_python.py"), str(ROOT / "scripts/stage_r20_input.py"), "--owner", spec["account"]],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        assert p.returncode == 0, f"Private input stage failed for {spec['account']}; inspect before retry"
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(stage, previous))
    run("scripts/prepare_r20.py")
    run("scripts/launch_r20.py")
    # Separate completion can resume authorized validation preparation while remote R kernels run.
    controller = "/home/odixe/.codex/skills/codex-long-jobs/scripts/codex-long-jobs"
    subprocess.run(["bash", controller, "start", "--name", "v20-local-preflight-20261102", "--log",
        str(ROOT / "runs/v-study/preflight.log"), "--success-pattern", "^V20_REAL_PREFLIGHT_PASS$", "--delivery", "auto",
        "--viewer", "none", "--", sys.executable, "scripts/run_local_guarded.py", sys.executable,
        "scripts/preflight_v20.py"], cwd=ROOT, check=True)
    run("scripts/run_r20_batch.py", "--collect-only")
    print("R20_RECOVERY_PIPELINE_DONE", flush=True)


if __name__ == "__main__": main()
