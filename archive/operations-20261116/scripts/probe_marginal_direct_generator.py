"""Reuse the frozen direct-probe gates with an explicitly fingerprinted sampler override."""
import hashlib
import json
from pathlib import Path

from scripts import probe_direct_generator as study
from scripts.marginal_direct_generator import Generator


def main():
    study.OUT = study.ROOT / "runs/marginal-direct-generator"
    study.OUT.mkdir(exist_ok=True)
    names = ("scripts/probe_marginal_direct_generator.py", "scripts/marginal_direct_generator.py",
             "scripts/probe_direct_generator.py", "scripts/direct_generator.py", "scripts/probe_pair_generator.py")
    before = {n: hashlib.sha256((study.ROOT/n).read_bytes()).hexdigest() for n in names}
    plan = {"sources": before, "sampler_override": "scripts.marginal_direct_generator.Generator",
            "base_runner": "scripts.probe_direct_generator", "gates": "unchanged from direct trial",
            "calibration": {"rows_per_class": 32768, "synthetic_only_seed": 9200, "quantiles": 4097,
                            "dimensions": "remaining58 only", "extra_wire_statistics": 0},
            "reason": "Direct forest reduced pair TV but conditional Gaussian distorted R marginal, class12/15"}
    path = study.OUT / "variant-plan.json"
    if path.exists():
        assert json.loads(path.read_text()) == plan
    else:
        path.write_text(json.dumps(plan, indent=2)+"\n")
    study.Generator = Generator
    study.main()
    for name, sha in before.items():
        assert hashlib.sha256((study.ROOT/name).read_bytes()).hexdigest() == sha
    result_path = study.OUT / "result.json"
    result = json.loads(result_path.read_text())
    result["variant_plan_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result_path.write_text(json.dumps(result, indent=2)+"\n")
    print("MARGINAL_PROBE_DONE", json.dumps(result["gates"]), flush=True)


if __name__ == "__main__":
    main()
