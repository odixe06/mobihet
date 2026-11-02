"""Freeze only audited Q6/Q7 aggregate teachers and bank identities, never private states."""
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.audit_q20 import check_decoded_probability


def main():
    target = ROOT / "methods/mobihet_rstudy_data.py"
    assert not target.exists(), "Frozen history exists; inspect rather than overwrite"
    batch = json.loads((ROOT / "runs/q-study/batch.json").read_text())
    closure = json.loads((ROOT / "runs/q-study/closure.json").read_text())
    values = {key: [] for key in ("Q", "class_body", "target", "body")}
    sources = []
    for spec in batch[6:8]:
        assert spec["mode"] == "mixed_input" and closure["runs"][spec["arm"]]["audited"]
        root = ROOT / "artifacts/q20" / spec["slug"] / spec["slug"]
        receipt = json.loads((ROOT / "runs/q-study/collection" / spec["arm"] / "diagnostic-audit.json").read_text())[0]
        source = {"ref": f'{spec["account"]}/{spec["slug"]}/1', "banks": [], "snapshots": []}
        history = {key: [] for key in values}
        for r in range(1, 21):
            path = root / "models" / f"r{r:03}.pt"
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            assert sha == receipt["snapshots"][r - 1]["sha256"]
            s = torch.load(path, weights_only=True, map_location="cpu")
            body = (s["class_sums"] / s["class_totals"][:, None]).float().half()
            check_decoded_probability(s["Q"], body)
            check_decoded_probability(s["bank_target"], s["bank_body"])
            for key, value in (("Q", s["Q"]), ("class_body", body), ("target", s["bank_target"]), ("body", s["bank_body"])):
                history[key].append(value.numpy())
            source["banks"].append(hashlib.sha256(s["bank_X"].numpy().tobytes()).hexdigest())
            source["snapshots"].append({"round": r, "sha256": sha})
        for key in values: values[key].append(history[key])
        sources.append(source)
    raw = b"".join(np.asarray(value, dtype="<f2" if key.endswith("body") else "<f4").tobytes()
                   for key, value in values.items())
    digest = hashlib.sha256(raw).hexdigest()
    target.write_text('"""Frozen Q6/Q7 train-only teachers; generated, never edit."""\n'
        + f"SHA256 = {digest!r}\nSOURCES = {sources!r}\nPAYLOAD = {base64.b64encode(zlib.compress(raw)).decode()!r}\n")
    out = ROOT / "runs/r-study"; out.mkdir(exist_ok=True)
    (out / "teacher-source.json").write_text(json.dumps({"sha256": digest, "decoded_bytes": len(raw),
        "module_bytes": target.stat().st_size, "sources": sources,
        "scope": "Train-only aggregate targets/bank SHA; no private states or evaluation inputs"}, indent=2) + "\n")
    print("R20_TEACHER_FROZEN", digest, len(raw), flush=True)


if __name__ == "__main__": main()
