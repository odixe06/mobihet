import torch

from methods.mobihet_copula_standard import standard_latent


def test_normalization_preserves_latent_correlation_without_changing_inputs():
    gen = torch.Generator().manual_seed(4)
    mu = torch.randn(16, 7, generator=gen)
    factor = torch.randn(16, 7, 7, generator=gen).tril()+torch.eye(7)
    old_mu, old_factor = mu.clone(), factor.clone()
    normalized_mu, normalized = standard_latent(mu, factor)
    cov = factor@factor.transpose(-1,-2)
    scale = cov.diagonal(dim1=-2, dim2=-1).sqrt()
    correlation = cov/scale[..., :, None]/scale[..., None, :]
    torch.testing.assert_close(normalized@normalized.transpose(-1,-2), correlation)
    torch.testing.assert_close(normalized.square().sum(-1), torch.ones_like(mu))
    assert normalized_mu.count_nonzero() == 0
    assert torch.equal(mu, old_mu) and torch.equal(factor, old_factor)


def test_zero_variance_rejected():
    import pytest
    with pytest.raises(AssertionError):
        standard_latent(torch.zeros(2,3),torch.zeros(2,3,3))
