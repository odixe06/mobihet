"""Recover a completed probe from its saved CMs after numpy.bool_ JSON failure; no refit."""
import hashlib
import json
from pathlib import Path

import numpy as np

from core.metrics import ten
from scripts.probe_pair_generator import ROOT, load


def main():
    out = ROOT / "runs/marginal-direct-generator"
    report = json.loads((out / "statistical-result.json").read_text())
    plan = out / "plan.json"
    variant = out / "variant-plan.json"
    assert hashlib.sha256(plan.read_bytes()).hexdigest() == report["plan_sha256"]
    for name, sha in json.loads(variant.read_text())["sources"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha
    _, ((xt, yt), (xv, yv)), _ = load()
    digest = hashlib.sha256()
    for a in (xt, yt, xv, yv):
        digest.update(a.tobytes())
    assert digest.hexdigest() == report["sample_sha256"]
    support = np.bincount(yv, minlength=16)
    logged = {}
    for line in (out / "probe.log").read_text().splitlines():
        if line.startswith("DIRECT_TSTR "):
            _, name, body = line.split(" ", 2)
            logged[name] = json.loads(body)
    scores, hashes = {}, {}
    for name in ("real", "copula", "direct"):
        path = out / (name+"-val-cm.npy")
        cm = np.load(path)
        assert cm.shape == (16, 16) and np.issubdtype(cm.dtype, np.integer) and (cm >= 0).all()
        np.testing.assert_array_equal(cm.sum(1), support)
        scores[name] = ten(cm)
        for k, v in scores[name].items():
            np.testing.assert_allclose(v, logged[name][k], atol=1e-14, rtol=0)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report["tstr"] = scores
    report["gates"]["tstr"] = bool(scores["direct"]["f1_macro"] >= scores["copula"]["f1_macro"]+.003)
    report["pass_feasibility"] = bool(all(report["gates"].values()))
    report["variant_plan_sha256"] = hashlib.sha256(variant.read_bytes()).hexdigest()
    report["recovery"] = {"reason": "numpy.bool_ tstr gate rejected by JSON; no model rerun",
                          "cm_sha256": hashes, "metrics_match_original_log": True, "data_sample_sha_verified": True}
    (out / "result.json").write_text(json.dumps(report, indent=2)+"\n")
    print("MARGINAL_ARTIFACT_RECOVERED", report["gates"], flush=True)


if __name__ == "__main__":
    main()
