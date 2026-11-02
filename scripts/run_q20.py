"""Q20 entrypoint: same shared driver, guard metadata in the run output."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", required=True)
    args, _ = parser.parse_known_args()
    os.environ["MOBIHET_RECOVERY_DIR"] = str(Path(args.out) / "recovery")
    from scripts import run_fl
    run_fl.main()


if __name__ == "__main__": main()
