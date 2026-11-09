import numpy as np

from scripts.standard_copula_generator import Generator
from scripts.probe_pair_generator import bins


def test_standard_copula_repairs_discrete_marginal():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(6000, 3)).astype(np.float32)
    x[:, 0] = (x[:, 0] > .4).astype(np.float32)
    model = Generator.fit(x, [(0, 1)])
    factor = model.chol/np.sqrt(np.square(model.chol).sum(1))[:, None]
    np.testing.assert_allclose(np.diag(factor@factor.T), 1, atol=1e-12)
    samples = model.sample(rng, 100000, direct=True)
    target = np.bincount(bins(x[:, 0]), minlength=16)/len(x)
    sampled = np.bincount(bins(samples[:, 0]), minlength=16)/len(samples)
    assert np.abs(target-sampled).sum()/2 < .015
