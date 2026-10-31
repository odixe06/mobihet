"""Local post-epoch class probabilities. Only aggregated class tables leave a client."""
import torch

from core import grouped


@torch.no_grad()
def class_probability_sums(env, flat, bufs, max_rows=1 << 15):
    """One pass over each client's real train rows, final weights, eval BN and dropout off.

    No parameter, buffer or RNG updates. Return local class sums and counts before wire encoding.
    Batching follows the existing own-client evaluator; padded rows never enter the sums.
    """
    data, dev = env.train, env.dev
    n = data.off[1:] - data.off[:-1]
    sums = torch.zeros(env.K, 16, 16, device=dev)
    counts = torch.zeros(env.K, 16, device=dev)
    order = torch.argsort(n, descending=True)
    i = 0
    while i < env.K:
        b = min(int(n[order[i]]), max_rows)
        g = max(1, min(env.K - i, max_rows // max(b, 1)))
        idx = order[i:i + g]
        P, B = env.lay.params(flat[idx]), env.lay.bufs(bufs[idx])
        start, cnt = data.off[idx], n[idx]
        gi = torch.arange(g, device=dev)[:, None]
        for s in range(0, int(cnt.max()), b):
            j = s + torch.arange(b, device=dev)[None]
            valid = j < cnt[:, None]
            rows = start[:, None] + torch.where(valid, j, 0)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                logits, _ = grouped.EVAL(P, B, data.X[rows], env.cfg, train=False)
            prob = torch.where(valid[..., None], logits.float().softmax(-1), 0.0)
            y = data.Y[rows].long()
            key = (gi.expand_as(y), y)
            sg = torch.zeros(g, 16, 16, device=dev).index_put_(key, prob, accumulate=True)
            cg = torch.zeros(g, 16, device=dev).index_put_(key, valid.float(), accumulate=True)
            sums[idx] += sg
            counts[idx] += cg
        i += g
    return sums, counts
