"""Diagnostic Core variant: standard-normal copula latent, identical aggregated wire statistics."""
import json
import os
from pathlib import Path

import torch

from methods.mobihet_wstudy import WLive


def standard_latent(mu, factor):
    scale = factor.square().sum(-1, keepdim=True).sqrt()
    assert torch.isfinite(scale).all() and (scale > 0).all()
    return torch.zeros_like(mu), factor / scale


class Method(WLive):
    def _copula_bootstrap(self, client, NB=256):
        super()._copula_bootstrap(client, NB)
        self.cop_mu, self.cop_L = standard_latent(self.cop_mu, self.cop_L)
        folder = os.environ.get("MOBIHET_RECOVERY_DIR")
        if folder:
            path = Path(folder)
            path.mkdir(parents=True, exist_ok=True)
            record = {"rank": self.env.rank, "latent_mu_max_abs": float(self.cop_mu.abs().max()),
                      "latent_variance_max_error": float((self.cop_L.square().sum(-1)-1).abs().max()),
                      "bootstrap_bytes": self.bootstrap_bytes, "scope": "client-side postdecode, no new payload"}
            (path / f"standard_copula_rank{self.env.rank}.json").write_text(json.dumps(record, indent=2)+"\n")
