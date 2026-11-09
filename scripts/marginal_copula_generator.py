"""Ablation: marginal transport on plain copula, no forest or pair-table sampling."""
import numpy as np

from scripts.direct_generator import Generator as Base


class Generator(Base):
    @classmethod
    def fit(cls, real, pairs):
        model = super().fit(real, pairs)
        reference = Base.sample(model, np.random.default_rng(9200), 32768)
        model.grid = np.linspace(0, 1, 4097)
        model.source_quantiles = np.quantile(reference, model.grid, axis=0)
        return model

    def sample(self, rng, n, direct=False):
        x = Base.sample(self, rng, n)
        if not direct:
            return x
        u = np.empty_like(x, dtype=np.float64)
        for j in range(x.shape[1]):
            values, first, count = np.unique(self.source_quantiles[:, j], return_index=True, return_counts=True)
            probability = (self.grid[first] + self.grid[first+count-1])/2
            u[:, j] = np.interp(x[:, j], values, probability, left=1e-6, right=1-1e-6)
        return self.decode(np.clip(u, 1e-6, 1-1e-6), range(x.shape[1]))
