"""Extended operation counter, one convention for every method (docs/research/compute-harness.md, owner 19/11/2026).

Every aten op dispatched while a phase is set is counted once, forward and backward alike (TorchDispatchMode is
propagated to the autograd threads):
  mm     ops with a torch.utils.flop_counter formula (mm/bmm/addmm/baddbmm/conv/SDPA ...): that formula's FLOPs, the
         convention of the earlier cost.py tables (2 x multiply-adds);
  other  every other op except views/aliases/metadata and uninitialised allocations: one op per element it produces,
         or per element it reads for reductions (sum/mean/amax/var/norm/topk/bincount/histc ...); indexing reads
         count the rows they gather, scatters the values they write. Data movement (copy/cat/cast) counts as an op per
         element. Not FLOPs in the strict sense: report mm and other separately, total = mm + other.
Ops are counted on whatever device they run (server code on CPU included); device is recorded alongside.
"""
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten
from torch.utils.flop_counter import flop_registry

aten = torch.ops.aten
FREE = {"empty", "empty_like", "empty_strided", "new_empty", "new_empty_strided", "detach", "alias", "lift_fresh",
        "_local_scalar_dense", "set_", "resize_", "is_same_size", "_has_compatible_shallow_copy_type", "record_stream",
        "sym_size", "sym_stride", "sym_numel", "sym_storage_offset", "is_nonzero", "_to_dense"}
READS_ALL = {"sum", "mean", "amax", "amin", "max", "min", "prod", "var", "std", "var_mean", "std_mean", "norm",
             "linalg_vector_norm", "logsumexp", "argmax", "argmin", "any", "all", "topk", "kthvalue", "median",
             "bincount", "histc", "histogram", "unique", "_unique2", "unique_consecutive", "count_nonzero", "nonzero",
             "nll_loss_forward", "nll_loss2d_forward", "native_layer_norm", "native_batch_norm", "searchsorted",
             "bucketize", "dist", "cdist", "_cdist_forward", "equal", "aminmax", "nansum", "cummax", "cummin"}
WRITES = {"index_put", "index_put_", "_index_put_impl_", "scatter", "scatter_", "scatter_add", "scatter_add_",
          "scatter_reduce", "scatter_reduce_", "index_add", "index_add_", "index_copy", "index_copy_", "masked_scatter",
          "masked_scatter_", "index_fill", "index_fill_"}


def _numel(tree):
    return [t.numel() for t in tree_flatten(tree)[0] if isinstance(t, torch.Tensor)]


def _device(tree):
    for t in tree_flatten(tree)[0]:
        if isinstance(t, torch.Tensor):
            return t.device.type
    return "cpu"


class OpCounter(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.phase = None
        self.mm = defaultdict(int)          # (phase, device) -> FLOPs
        self.other = defaultdict(int)       # (phase, device) -> element ops

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        ph = self.phase
        if ph is None:
            return out
        pkt = func._overloadpacket
        key = (ph, _device(out) if _numel(out) else _device(args))
        if pkt in flop_registry:
            self.mm[key] += int(flop_registry[pkt](*args, **kwargs, out_val=out))
            return out
        name = pkt.__name__
        if func.is_view or name in FREE:
            return out
        if name in WRITES:                                            # values written: the source operand
            src = kwargs.get("values", kwargs.get("src", kwargs.get("source")))
            if src is None:
                src = args[-1] if len(args) > 2 and isinstance(args[-1], torch.Tensor) else None
            n = src.numel() if isinstance(src, torch.Tensor) else max(_numel(out) or [0])
        elif name in READS_ALL:
            n = max(_numel((args, kwargs)) or [0])
        else:
            n = sum(_numel(out))
        self.other[key] += int(n)
        return out

    def table(self):
        keys = sorted(set(self.mm) | set(self.other))
        return [{"phase": p, "device": d, "mm_flops": self.mm.get((p, d), 0), "other_ops": self.other.get((p, d), 0)}
                for p, d in keys]
