"""Repair sketch-space overshoot using a decoded-patch residual line search."""
import torch
from mh.v2_methods import Method as Original, finite
from mh.v2_math import pack_patch, unpack

ARMS = ['local', 'a_own', 'a_dense', 'a_small_noef', 'a_small_contract_ef',
        'a_medium_noef', 'a_medium_contract_ef']


def contract_patch(sketch, accumulated, k, base):
    _, proposal = sketch.patch(accumulated, k, base)
    encoded = sketch.encode(proposal)
    denominator = encoded.double().square().sum()
    gamma = float((accumulated.double()*encoded.double()).sum()/denominator) if denominator else 0.
    gamma = min(1., max(0., gamma))
    indices = torch.nonzero(proposal, as_tuple=False).flatten()
    before = float(accumulated.double().square().sum())
    # Quantization is inside the acceptance test. Zero is the guaranteed safe fallback.
    for attempt in range(17):
        scale = gamma*(.5**attempt) if attempt < 16 else 0.
        wire = pack_patch(indices, proposal[indices]*scale, base, base+1)
        restored = unpack(wire, accumulated.device, base)
        delta = torch.zeros_like(proposal)
        delta[restored['indices']] = restored['values']
        residual = accumulated-sketch.encode(delta)
        if float(residual.double().square().sum()) <= before:
            return wire, delta, residual, scale
    raise AssertionError('zero patch must preserve residual exactly')


class Method(Original):
    def aggregate(self, r):
        if self.sketch is None or '_contract_' not in self.arm:
            return super().aggregate(r)
        total, n = self.aggregate_sum.finish()
        self.last_mean = (total/n).detach().clone()
        accumulated = self.last_mean+self.residual
        wire, delta, residual, gamma = contract_patch(self.sketch, accumulated, self.topk, self.version)
        self.global_vector += delta
        self.residual = residual
        assert torch.isfinite(self.global_vector).all()
        self.latest_packet = wire
        self.diag = dict(residual_norm=float(residual.norm()), update_norm=float(delta.norm()),
                         accumulated_norm=float(accumulated.norm()), gamma=gamma,
                         contraction=float(residual.norm()/accumulated.norm()) if accumulated.norm() else 0.)
        for k in range(self.R.K):
            self.record(wire, r, k, 'down', 'proxy_update')
        self.version += 1
