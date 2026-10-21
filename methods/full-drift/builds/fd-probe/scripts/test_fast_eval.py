"""fd_eval_fast.FastDAGSNet computes DAGSNet exactly (fp32, no autocast): random weights and BatchNorm statistics, weights
loaded into buffers built from another model. The fp16 run-level agreement is tested by test_final_pipeline.py (smoke) and
measured on real rounds by compare_eval.py."""
from pathlib import Path
import sys

import torch

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
sys.path[:0] = [str(ROOT / '.compat'), str(BASE / 'source/proj'), str(ROOT / 'methods/shared-baselines/builds/federated-baselines-full/source/proj')]
torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
from fd_eval_fast import FastDAGSNet
from mh import pilot as P


def main():
    torch.manual_seed(0)
    for trial in range(3):
        m = P.make('dagsnet').cuda().eval()
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, torch.nn.BatchNorm1d):
                    mod.running_mean.uniform_(-.5, .5); mod.running_var.uniform_(.3, 3.); mod.weight.uniform_(.5, 1.5); mod.bias.uniform_(-.3, .3)
            fast = FastDAGSNet(m if trial == 0 else P.make('dagsnet').cuda().eval()).cuda(); fast.load(m)
            x = torch.randn(4096, P.D, device='cuda'); ref = m(x)[0]; out = fast(x)
        error = float((out - ref).abs().max())
        assert error < 1e-5 and torch.equal(out.argmax(1), ref.argmax(1)), (trial, error)
    print('PASS FastDAGSNet == DAGSNet in fp32 (max abs', error, ')', flush=True)


if __name__ == '__main__': main()
