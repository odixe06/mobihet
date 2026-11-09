"""Gaussian copula needs N(0, correlation) latent before the standard-normal CDF."""
import numpy as np
from scipy.special import ndtr

from scripts.direct_generator import Generator as Base


class Generator(Base):
    def sample(self, rng, n, direct=False):
        if not direct:
            return Base.sample(self, rng, n)
        scale = np.sqrt(np.square(self.chol).sum(1))
        factor = self.chol / scale[:, None]
        z = rng.standard_normal((n, len(self.mu))) @ factor.T
        return self.decode(np.clip(ndtr(z), 1e-6, 1-1e-6), range(len(self.mu)))
