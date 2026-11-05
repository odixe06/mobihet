"""Core codec feasibility: bootstrap bins and Q downlink only; no loss/optimizer change."""
import hashlib
import json
import os
from pathlib import Path
import zlib

import numpy as np
import torch
from methods.mobihet_wstudy import WLive
from methods.mobihet_next import signature, require_shared


def q8(q):
    body = torch.round(q * 255).to(torch.uint8)
    decoded = body.float()
    return body, decoded / decoded.sum(-1, keepdim=True).clamp_min(1)


class Method(WLive):
    def _copula_bootstrap(self, client, NB=256):
        nb = int(self.env.hp.get("compression_bins", 256))
        assert nb in (64, 256)
        super()._copula_bootstrap(client, nb)
        # Sparse histogram size at 64 bins has not been measured: use conservative dense uint32 accounting.
        if nb == 64:
            self.bootstrap_bytes["up"] = 64 + 4 * 16 * 77 * nb + 4 * 16 * (77 + 77 * 78 // 2)
        cdf = np.rint(self.cop_cdf.cpu().numpy().astype(np.float64)*65535).astype("<u2")
        raw = cdf.tobytes()
        packed = zlib.compress(raw, 6)
        assert zlib.decompress(packed) == raw
        path = Path(os.environ["MOBIHET_RECOVERY_DIR"]).parent / f"bootstrap_codec_rank{self.env.rank}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(bins=nb, cdf_bytes=len(raw), cdf_zlib_bytes=len(packed),
                                       cdf_sha256=hashlib.sha256(raw).hexdigest(), roundtrip=True,
                                       bootstrap_bytes=self.bootstrap_bytes), indent=2) + "\n")

    def round(self, r, lr):
        row = super().round(r, lr)
        if self.env.hp.get("compression_q8", False):
            original = self.Q.clone()
            body, self.Q = q8(original)
            assert torch.isfinite(self.Q).all() and (self.Q >= 0).all()
            row["codec_q8_max_abs"] = float((original-self.Q).abs().max())
            row["codec_q8_argmax_changes"] = int((original.argmax(-1) != self.Q.argmax(-1)).sum())
            row["bytes_down"] = body.numel() + 32
            current = signature(self.Q, self.bank_X, self.bank_target, self.variant)
            require_shared(self.env, current)
            row["teacher_produced_sha256"] = current
            path = Path(os.environ["MOBIHET_RECOVERY_DIR"]).parent / f"qcodec_rank{self.env.rank}_r{r:03d}.pt"
            torch.save(dict(original=original.cpu(), body=body.cpu(), decoded=self.Q.cpu()), path)
        return row
