"""Use the shared eight-run one-shot launcher with O's frozen manifest and receipts."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.launch_n20 import main

if __name__ == "__main__": main(Path(__file__).resolve().parents[1] / "runs/o-study")
