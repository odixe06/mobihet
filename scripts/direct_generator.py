"""CPU sufficient-statistic forest/Gaussian sampler for feasibility, not FL production."""
import numpy as np
from scipy.special import ndtr, ndtri

from scripts.probe_pair_generator import bins


def forest_order(pairs):
    graph = {}
    for a, b in pairs:
        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)
    seen, roots, edges = set(), [], []
    for root in sorted(graph):
        if root in seen:
            continue
        roots.append(root); seen.add(root)
        queue = [(root, None)]
        while queue:
            node, parent = queue.pop(0)
            for child in sorted(graph[node]):
                if child == parent:
                    continue
                assert child not in seen, "Pair graph must be a forest"
                seen.add(child); edges.append((node, child)); queue.append((child, node))
    return sorted(graph), roots, edges


def conditional_parameters(mu, covariance, selected):
    rest = np.array(sorted(set(range(len(mu))) - set(selected)), dtype=int)
    selected = np.array(selected, dtype=int)
    a = np.linalg.solve(covariance[np.ix_(selected, selected)], covariance[np.ix_(selected, rest)]).T
    conditional = covariance[np.ix_(rest, rest)] - a @ covariance[np.ix_(selected, rest)]
    conditional = (conditional + conditional.T) / 2
    return rest, a, np.linalg.cholesky(conditional)


def counts(real, pairs):
    b = bins(real, 256)
    single = np.stack([np.bincount(b[:, j], minlength=256) for j in range(real.shape[1])])
    coarse = b // 16
    joint = {(i, j): np.bincount(coarse[:, i]*16 + coarse[:, j], minlength=256).reshape(16, 16)
             for i, j in pairs}
    return single, joint


class Generator:
    def __init__(self, single, joint, mu, chol, pairs):
        self.mass = single / single.sum(1, keepdims=True)
        original_cdf = self.mass.cumsum(1)
        self.cdf = np.rint(original_cdf * 65535) / 65535
        self.cdf[:, -1] = 1
        self.decoded_mass = np.diff(self.cdf, axis=1, prepend=np.zeros((len(single), 1)))
        self.mu, self.chol = mu, chol
        self.selected, self.roots, self.edges = forest_order(pairs)
        self.rest, self.a, self.conditional_chol = conditional_parameters(mu, chol @ chol.T, self.selected)
        self.single, self.joint = single, joint

    @classmethod
    def fit(cls, real, pairs):
        single, joint = counts(real, pairs)
        mass = single / single.sum(1, keepdims=True)
        score = ndtri(np.clip(mass.cumsum(1) - mass / 2, 1e-4, 1-1e-4)).astype(np.float32)
        z = score[np.arange(real.shape[1]), bins(real, 256)].astype(np.float64)
        mu = z.mean(0)
        centered = z - mu
        cov = centered.T @ centered / len(z) + np.eye(real.shape[1]) * 1e-4
        chol = np.linalg.cholesky(cov).astype(np.float16).astype(np.float64)
        return cls(single, joint, mu.astype(np.float16).astype(np.float64), chol, pairs)

    def decode(self, u, columns):
        result = np.empty_like(u, dtype=np.float32)
        for k, j in enumerate(columns):
            ix = np.searchsorted(self.cdf[j], u[:, k]).clip(0, 255)
            mass = self.decoded_mass[j, ix]
            fraction = (u[:, k] - self.cdf[j, ix] + mass) / np.maximum(mass, 1e-12)
            result[:, k] = -4 + (ix + fraction) / 32
        return result

    def sample(self, rng, n, direct=False):
        d = len(self.mu)
        if not direct:
            z = rng.standard_normal((n, d)) @ self.chol.T + self.mu
            return self.decode(np.clip(ndtr(z), 1e-6, 1-1e-6), range(d))
        coarse = np.full((n, d), -1, dtype=int)
        for root in self.roots:
            probability = self.single[root].reshape(16, 16).sum(1).astype(float)
            probability /= probability.sum()
            coarse[:, root] = np.searchsorted(probability.cumsum(), rng.random(n)).clip(0, 15)
        for parent, child in self.edges:
            table = self.joint.get((parent, child))
            if table is None:
                table = self.joint[(child, parent)].T
            denominator = table.sum(1, keepdims=True)
            assert (denominator[coarse[:, parent]] > 0).all(), "Sampled unsupported parent"
            conditional = table / np.maximum(denominator, 1)
            coarse[:, child] = (rng.random(n)[:, None] > conditional.cumsum(1)[coarse[:, parent]]).sum(1).clip(0, 15)
        x = np.empty((n, d), dtype=np.float32)
        u_selected = np.empty((n, len(self.selected)))
        for k, j in enumerate(self.selected):
            group = coarse[:, j]
            mass = self.single[j].reshape(16, 16).astype(float)
            mass /= np.maximum(mass.sum(1, keepdims=True), 1)
            local_fine = (rng.random(n)[:, None] > mass.cumsum(1)[group]).sum(1).clip(0, 15)
            fine = group * 16 + local_fine
            fraction = rng.random(n)
            x[:, j] = -4 + (fine + fraction) / 32
            u_selected[:, k] = self.cdf[j, fine] - self.decoded_mass[j, fine] * (1-fraction)
        z_selected = ndtri(np.clip(u_selected, 1e-6, 1-1e-6))
        mean = self.mu[self.rest] + (z_selected - self.mu[self.selected]) @ self.a.T
        z_rest = mean + rng.standard_normal((n, len(self.rest))) @ self.conditional_chol.T
        x[:, self.rest] = self.decode(np.clip(ndtr(z_rest), 1e-6, 1-1e-6), self.rest)
        assert np.isfinite(x).all()
        return x
