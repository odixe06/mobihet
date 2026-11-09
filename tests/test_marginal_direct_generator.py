import numpy as np

from scripts.marginal_direct_generator import Generator
from scripts.probe_pair_generator import bins


def test_transport_only_changes_remaining_dimensions_and_restores_marginal():
    rng = np.random.default_rng(81)
    x = rng.normal(size=(8000, 4)).astype(np.float32)
    x[:, 1] = x[:, 0]**2 - 1
    x[:, 2] = x[:, 0] + x[:, 1] + .05*x[:, 2]
    model = Generator.fit(x, [(0, 1)])
    from scripts.direct_generator import Generator as Base
    before = Base.sample(model, np.random.default_rng(4), 50000, direct=True)
    after = model.sample(np.random.default_rng(4), 50000, direct=True)
    np.testing.assert_array_equal(before[:, model.selected], after[:, model.selected])
    for j in model.rest:
        target = np.bincount(bins(x[:, j]), minlength=16)/len(x)
        sample = np.bincount(bins(after[:, j]), minlength=16)/len(after)
        assert np.abs(target-sample).sum()/2 < .025


def test_control_is_identical_to_untransported_gaussian():
    from scripts.direct_generator import Generator as Base
    x = np.random.default_rng(8).normal(size=(1000, 3)).astype(np.float32)
    model = Generator.fit(x, [(0, 1)])
    np.testing.assert_array_equal(model.sample(np.random.default_rng(1), 1000),
                                  Base.sample(model, np.random.default_rng(1), 1000))
