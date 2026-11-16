"""B18 source proof: executed Python bytes, embedded HP resource + runtime config.

Old notebook provenance copies Python only. Never fabricate a missing executed
resource file or silently skip its hash: inspect the sealed notebook payload.
"""
import ast
import base64
import hashlib
import json
from pathlib import Path

HP = "runs/sim/core_hp.txt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def value(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def audit(base, spec, notebook):
    base, notebook = Path(base), Path(notebook)
    manifest = json.loads((base / "provenance/source_manifest.json").read_text())
    assert manifest["base_sha"] == spec["base_sha"] and manifest["overlay"] == spec["overlay_sha256"]
    assert sha(notebook) == spec["notebook_sha256"], "Sealed pushed notebook changed"
    embedded = []
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "overlay" for t in node.targets):
                embedded.append(ast.literal_eval(node.value))
    assert len(embedded) == 1
    overlay = embedded[0]
    assert set(overlay) == set(spec["overlay_sha256"])
    kinds = {}
    for name, expected in spec["overlay_sha256"].items():
        data = base64.b64decode(overlay[name]["base64"], validate=True)
        assert overlay[name]["sha256"] == expected == hashlib.sha256(data).hexdigest()
        executed = base / "provenance/source" / name
        if executed.exists():
            assert sha(executed) == expected, ("Executed file changed", name)
            kinds[name] = "executed_file_and_embedded_payload"
        else:
            assert name == HP and name not in manifest["files"], ("Missing executed source", name)
            kinds[name] = "sealed_embedded_payload_and_runtime_config; executed_resource_copy_unavailable"
    expected_hp = dict(item.split("=", 1) for item in base64.b64decode(overlay[HP]["base64"]).decode().split())
    # Additional fixed CBB knobs are literal in the actual executed phase module.
    phase_source = base / "provenance/source/scripts/b18_phase.py"
    assignment = next(n for n in ast.parse(phase_source.read_text()).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == "CBB_HP" for t in n.targets))
    assert isinstance(assignment.value, ast.BinOp) and isinstance(assignment.value.op, ast.Add)
    expected_hp.update(dict(item.split("=", 1) for item in ast.literal_eval(assignment.value.right)))
    expected_hp["seed"] = str(spec["seed"])
    expected_hp = {k: value(v) for k, v in expected_hp.items()}
    for phase in ("shadow", "C", "E", "H"):
        cfg = json.loads((base / spec["slug"] / phase / "config.json").read_text())
        assert all(cfg["hp"][k] == v for k, v in expected_hp.items()), ("Effective HP mismatch", phase)
    return {"pass": True, "overlay_proofs": kinds, "effective_hp": expected_hp,
            "notebook_sha256": spec["notebook_sha256"],
            "scope": "Python executed bytes retained; HP bytes from sealed pushed notebook + manifest + four runtime configs. No claim of retained executed HP file."}
