import numpy as np

from scripts.direct_generator import Generator, conditional_parameters, counts, forest_order
from scripts.probe_pair_generator import bins


def test_counts_aggregate_and_forest_sampler_preserves_nonlinear_pair():
    rng = np.random.default_rng(42)
    u = rng.uniform(-2, 2, 8000)
    real = np.column_stack([u, u*u-2, rng.normal(size=len(u))]).astype(np.float32)
    full, joint = counts(real, [(0, 1)])
    a, aj = counts(real[:3000], [(0, 1)])
    b, bj = counts(real[3000:], [(0, 1)])
    np.testing.assert_array_equal(full, a+b)
    np.testing.assert_array_equal(joint[(0, 1)], aj[(0, 1)] + bj[(0, 1)])
    generator = Generator.fit(real, [(0, 1)])
    x = generator.sample(rng, 100000, direct=True)
    sample = bins(x[:, :2])
    empirical = np.bincount(sample[:, 0]*16+sample[:, 1], minlength=256).reshape(16, 16)/len(sample)
    target = joint[(0, 1)]/len(real)
    assert np.abs(empirical-target).sum()/2 < .02


def test_conditional_gaussian_recovers_joint_moments():
    rng = np.random.default_rng(52)
    cov = np.array([[2., .7, -.3], [.7, 1., .2], [-.3, .2, 1.]])
    mu = np.array([1., 2., -1.])
    rest, a, chol = conditional_parameters(mu, cov, [0])
    z = np.empty((100000, 3))
    z[:, 0] = rng.normal(mu[0], np.sqrt(cov[0, 0]), len(z))
    z[:, rest] = mu[rest] + (z[:, [0]]-mu[[0]])@a.T + rng.normal(size=(len(z), 2))@chol.T
    np.testing.assert_allclose(z.mean(0), mu, atol=.015)
    np.testing.assert_allclose(np.cov(z.T), cov, atol=.025)


def test_cycles_rejected():
    import pytest
    with pytest.raises(AssertionError, match="forest"):
        forest_order([(0, 1), (1, 2), (2, 0)])
