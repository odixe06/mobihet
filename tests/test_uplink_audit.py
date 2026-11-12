"""Captured real LG packet regression; corruption must still fail component bounds."""
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.audit_uplink import check_aggregate


def captured():
    # Class 6, real local smoke r1: fp16 log-score cells plus small Q8 cells.
    with np.load(Path(__file__).parent / "fixtures/uplink-fallback-r1.npz") as z:
        counts = z["counts"]
        values = torch.from_numpy(z["values"].astype(np.float32))
        small = torch.from_numpy(z["small"].copy()).float() / 255
        logp = small.clamp_min(1e-6).log()
        logp -= logp.mean(-1, keepdim=True)
        values[counts < 8] = logp[counts < 8]
        total = np.zeros((16, 16), np.float64)
        stored = total.copy()
        total[6] = (values.double().numpy() * counts[:, None]).sum(0)
        stored[6] = z["stored_sum"]
        n = np.ones(16, np.float64)
        n[6] = counts.sum()
        fallback = np.zeros(16, np.int64)
        fallback[6] = counts[counts < 8].sum()
    return total, stored, n, fallback


def test_captured_hybrid_packet_is_not_pure_fp16_replay():
    total, stored, count, fallback = captured()
    error = check_aggregate(total, stored, count, fallback, "lg16")
    assert 1e-10 < error < 1e-8  # reproduce the actual CPU/GPU fallback-log discrepancy


@pytest.mark.parametrize("label", [0, 6])
def test_corrupt_aggregate_is_still_rejected(label):
    total, stored, count, fallback = captured()
    stored[label, 0] += count[label] * (1e-8 if label == 0 else 1e-4)
    with pytest.raises(AssertionError):
        check_aggregate(total, stored, count, fallback, "lg16")
