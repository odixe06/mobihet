"""Monotone marginal transport fixes the drift introduced by Gaussian conditional sampling."""
import numpy as np

from scripts.direct_generator import Generator as Base


class Generator(Base):
    @classmethod
    def fit(cls, real, pairs):
        model = super().fit(real, pairs)
        # Independent synthetic calibration, never validation rows. Store a fixed quantile grid, not data rows.
        reference = Base.sample(model, np.random.default_rng(9200), 32768, direct=True)
        model.grid = np.linspace(0, 1, 4097)
        model.source_quantiles = np.quantile(reference[:, model.rest], model.grid, axis=0)
        return model

    def sample(self, rng, n, direct=False):
        x = super().sample(rng, n, direct)
        if not direct:
            return x
        u = np.empty((n, len(self.rest)))
        for k, j in enumerate(self.rest):
            source = self.source_quantiles[:, k]
            # Collapsed values get midpoint CDF; no fabricated ordering inside ties.
            values, first, count = np.unique(source, return_index=True, return_counts=True)
            probability = (self.grid[first] + self.grid[first+count-1]) / 2
            u[:, k] = np.interp(x[:, j], values, probability, left=1e-6, right=1-1e-6)
        x[:, self.rest] = self.decode(np.clip(u, 1e-6, 1-1e-6), self.rest)
        return x
