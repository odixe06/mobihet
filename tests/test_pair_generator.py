"""Marginal fitting recovers nonlinear dependence; unavailable support stays visible."""
import numpy as np

from scripts.probe_pair_generator import evaluate, fit_potentials


def test_pair_fit_recovers_dependence_without_changing_singletons():
    base = np.tile(np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), (100, 1))
    real = np.repeat(np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]]), [400, 100, 100, 400], axis=0)
    potentials, _ = fit_potentials(real, base, [(0, 1)])
    before, _ = evaluate(real, base, [(0, 1)])
    after, weights = evaluate(real, base.copy(), [(0, 1)], potentials)
    assert before["pair_tv"] > .29 and after["pair_tv"] < .001
    assert after["singleton_tv"] < .001
    assert after["ess_fraction"] > .7 and np.isclose(weights.sum(), 1)


def test_impossible_support_is_reported_instead_of_projected_away():
    real = np.tile(np.array([[-1., -1.], [1., 1.]]), (100, 1))
    base = np.tile(np.array([[-1., 1.], [1., -1.]]), (100, 1))
    potentials, _ = fit_potentials(real, base, [(0, 1)])
    result, weights = evaluate(real, base, [(0, 1)], potentials)
    assert result["unsupported_pair_mass_max"] == 1
    assert result["pair_tv"] == 1 and np.isfinite(weights).all()
