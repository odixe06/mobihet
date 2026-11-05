"""Regression against the stored P15r5 CPU/CUDA fp16 rounding boundary."""
from pathlib import Path

import pytest
import torch

from methods.mobihet_study import decode_probability
from scripts.audit_p20 import check_class_mean


@pytest.fixture
def saved():
    path = Path(__file__).resolve().parents[1] / 'artifacts/p20/mh-p15-bank-input-s43/mh-p15-bank-input-s43/models/r005.pt'
    if not path.exists(): pytest.skip('P15r5 artifact not restored')
    return torch.load(path, map_location='cpu', weights_only=True)


def test_stored_cuda_rounding_boundary(saved):
    table = torch.zeros(16, 16).index_add_(0, saved['bank_y'], decode_probability(saved['bank_body'])) / 64
    assert (saved['Q'] - decode_probability(table.half())).abs().max() > 2e-7
    assert check_class_mean(saved['Q'], saved['bank_body'], saved['bank_y'])['ambiguous_fp16_bins'] > 0


def test_rejects_changed_class_identity(saved):
    with pytest.raises(AssertionError, match='outside'):
        check_class_mean(saved['Q'].roll(1, 1), saved['bank_body'], saved['bank_y'])


def test_rejects_corrupted_probability(saved):
    q = saved['Q'].clone()
    q[7, 7] += .01
    q[7] /= q[7].sum()
    with pytest.raises(AssertionError, match='outside'):
        check_class_mean(q, saved['bank_body'], saved['bank_y'])
