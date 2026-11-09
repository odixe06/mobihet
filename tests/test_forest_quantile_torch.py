import numpy as np
import torch

from methods.mobihet_forest_quantile import Sampler
from scripts.direct_generator import Generator as Base
from scripts.marginal_direct_generator import Generator
from scripts.probe_pair_generator import bins


def model():
    rng = np.random.default_rng(71)
    x = rng.normal(size=(8000, 4)).astype(np.float32)
    x[:, 1] = x[:, 0]**2 - 1
    x[:, 2] = x[:, 0] + x[:, 1] + .05*x[:, 2]
    return x, Generator.fit(x, [(0,1)])


def test_torch_transport_matches_numpy_including_ties_and_endpoints():
    _, m = model()
    # Force duplicate quantiles to exercise midpoint CDF, including the first/last group.
    m.source_quantiles = np.round(m.source_quantiles, 1)
    sampler = Sampler([m], "cpu")
    x = np.array([[-10, -10], [-.5, 0], [0, .5], [10,10], m.source_quantiles[0], m.source_quantiles[-1]], np.float32)
    u = []
    grid = np.linspace(0,1,len(m.source_quantiles))
    for k in range(2):
        values, first, count = np.unique(m.source_quantiles[:,k],return_index=True,return_counts=True)
        probability = (grid[first]+grid[first+count-1])/2
        u.append(np.interp(x[:,k],values,probability,left=1e-6,right=1-1e-6))
    expected = m.decode(np.clip(np.array(u).T,1e-6,1-1e-6),m.rest)
    actual = sampler.transport(torch.tensor(x)[None])[0].numpy()
    np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=0)


def test_torch_sampler_preserves_pair_and_marginals_and_rng_resume():
    x,m=model(); sampler=Sampler([m,m],"cpu")
    gen=torch.Generator().manual_seed(1)
    a=sampler.sample(50000,gen).numpy()
    assert np.isfinite(a).all() and a.shape==(2,50000,4)
    for c in range(2):
        target=m.joint[(0,1)]/len(x)
        b=bins(a[c,:,:2]);sample=np.bincount(b[:,0]*16+b[:,1],minlength=256).reshape(16,16)/len(b)
        assert np.abs(sample-target).sum()/2 < .025
        for j in m.rest:
            target=np.bincount(bins(x[:,j]),minlength=16)/len(x)
            sample=np.bincount(bins(a[c,:,j]),minlength=16)/len(b)
            assert np.abs(sample-target).sum()/2 < .03
    state=gen.get_state();b=sampler.sample(50,gen)
    gen.set_state(state)
    assert torch.equal(b,sampler.sample(50,gen))


def test_sampler_does_not_modify_calibrated_numpy_statistics():
    _,m=model();before={k:v.copy() for k,v in vars(m).items() if isinstance(v,np.ndarray)}
    sampler=Sampler([m],"cpu")
    sampler.sample(100,torch.Generator().manual_seed(43))
    assert all(np.array_equal(vars(m)[k],v) for k,v in before.items())
    assert np.array_equal(m.sample(np.random.default_rng(2),100),Base.sample(m,np.random.default_rng(2),100))
