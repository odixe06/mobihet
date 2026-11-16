"""Exercise B18 imports from committed clone plus ONLY declared notebook overlays."""
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_b18_pipeline import OVERLAYS


def main():
    archive = subprocess.check_output(["git", "archive", "HEAD"], cwd=ROOT)
    with tempfile.TemporaryDirectory(prefix="b18-deployment-") as folder:
        stage = Path(folder)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(stage, filter="data")
        for rel in OVERLAYS:
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / rel, dest)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(stage)
        code = ("import sys; sys.path.insert(0, '.'); "
                "from scripts.b18_phase import CBB_HP; "
                "from methods.mobihet_source import Method; "
                "from scripts.audit_b18 import audit; "
                "assert 'q_bits=8' in CBB_HP; print('B18_DEPLOYMENT_IMPORT_PASS')")
        subprocess.run([sys.executable, "-c", code], cwd=stage, env=env, check=True)


if __name__ == "__main__":
    main()
