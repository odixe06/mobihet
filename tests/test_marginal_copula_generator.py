import numpy as np

from scripts.marginal_copula_generator import Generator
from scripts.direct_generator import Generator as Base
from scripts.probe_pair_generator import bins


def test_plain_transport_restores_marginals_and_control_stays_identical():
    rng = np.random.default_rng(73)
    x = rng.normal(size=(8000, 3)).astype(np.float32)
    x[:, 0] = (x[:, 0] > 0).astype(np.float32)
    model = Generator.fit(x, [(0, 1)])
    np.testing.assert_array_equal(model.sample(np.random.default_rng(1), 3000),
                                  Base.sample(model, np.random.default_rng(1), 3000))
    samples = model.sample(rng, 50000, direct=True)
    for j in range(3):
        target = np.bincount(bins(x[:, j]), minlength=16)/len(x)
        actual = np.bincount(bins(samples[:, j]), minlength=16)/len(samples)
        assert np.abs(target-actual).sum()/2 < .025
