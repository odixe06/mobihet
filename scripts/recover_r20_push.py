"""Fail-safe packaging recovery: prove rejected slugs absent, preserve attempt, extract identical teacher bytes."""
import base64
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.launch_r20 import owned_refs


def main():
    out = ROOT / "runs/r-study"; preserved = out / "push-recovery/attempt1"
    assert not preserved.exists(), "Recovery already recorded; inspect before retry"
    batch = json.loads((out / "batch.json").read_text())
    skill = ROOT / ".agents/skills/kaggle-training-notebook/scripts/kaggle_as.py"
    def inspect(spec):
        cmd = [sys.executable, str(skill), spec["account"], "--", sys.executable, str(ROOT / "scripts/kaggle_cli_bounded.py")]
        refs = owned_refs(cmd, spec["account"])
        assert f'{spec["account"]}/{spec["slug"]}' not in refs, "Remote slug exists: collect exact version, never repush"
        return {"arm": spec["arm"], "owner": spec["account"], "rejected_slug_absent": True, "refs": refs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: inventory = list(pool.map(inspect, batch))
    preserved.mkdir(parents=True)
    paths = [out / n for n in ("batch.json", "preflight.json", "tests.xml", "audit-tests.xml", "worker.log", "launch-gates.json")]
    paths += list(out.glob("*.push*")) + [ROOT / "docs/research/r20-launch.json", ROOT / "methods/mobihet_rstudy_data.py"]
    for path in paths:
        dest = preserved / path.name; shutil.copy2(path, dest)
        assert path.read_bytes() == dest.read_bytes()
    from methods import mobihet_rstudy_data as frozen
    compressed = base64.b64decode(frozen.PAYLOAD); raw = zlib.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == frozen.SHA256
    source = out / "teacher-input"; source.mkdir(exist_ok=True)
    name = f"r20-teacher-{frozen.SHA256[:12]}.bin"
    (source / name).write_bytes(compressed)
    packed_sha = hashlib.sha256(compressed).hexdigest()
    wrapper = ('"""Frozen Q6/Q7 aggregates. Large immutable payload mounted from a private input dataset."""\n'
        + 'import base64\nimport glob\nimport hashlib\nfrom pathlib import Path\n'
        + f'SHA256 = {frozen.SHA256!r}\nSOURCES = {frozen.SOURCES!r}\nFILE = {name!r}\nCOMPRESSED_SHA256 = {packed_sha!r}\n'
        + 'local = Path(__file__).resolve().parents[1] / "runs/r-study/teacher-input" / FILE\n'
        + 'paths = [local] if local.is_file() else [Path(p) for p in glob.glob("/kaggle/input/**/" + FILE, recursive=True)]\n'
        + 'assert len(paths) == 1, "Missing or ambiguous private replay input; refuse training"\n'
        + 'compressed = paths[0].read_bytes()\n'
        + 'assert hashlib.sha256(compressed).hexdigest() == COMPRESSED_SHA256\n'
        + 'PAYLOAD = base64.b64encode(compressed).decode()\n')
    (ROOT / "methods/mobihet_rstudy_data.py").write_text(wrapper)
    # Old immutable attempt remains under preserved/. New receipts use new slugs.
    (out / "batch.json").rename(out / "push-recovery/rejected-batch.json")
    receipt = {"inventory": inventory, "no_remote_kernel_created": True, "decoded_sha256": frozen.SHA256,
        "compressed_sha256": packed_sha, "compressed_bytes": len(compressed), "decoded_bytes": len(raw),
        "old_module_bytes": (preserved / "mobihet_rstudy_data.py").stat().st_size,
        "new_module_bytes": len(wrapper.encode()), "scope": "Same teacher bytes; API400 cause not yet proven, packaging change only"}
    (out / "push-recovery/recovery.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("R20_PACKAGING_RECOVERY_READY", flush=True)


if __name__ == "__main__": main()
