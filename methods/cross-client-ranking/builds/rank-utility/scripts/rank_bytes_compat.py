"""PyTorch 2.10 bytes checkpoint compatibility; no model/source or loader-mode change."""
import runpy
import sys
import torch


if __name__ == '__main__':
    script = sys.argv.pop(1)
    sys.argv[0] = script
    with torch.serialization.safe_globals([bytes]):
        runpy.run_path(script, run_name='__main__')
