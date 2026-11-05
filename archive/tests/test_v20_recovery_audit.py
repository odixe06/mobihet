"""Telemetry audit rejects mismatched steps, untrained clients and forged finite summaries."""
import copy
import numpy as np
import pytest
from scripts.audit_v20_recovery import check_telemetry


def fixture():
    counts = np.zeros((8, 16), dtype=np.int64); counts[3, 0] = 65; counts[7, 0] = 128
    clients = [{"client_id": 3, "applied": 1, "skipped": 1}, {"client_id": 7, "applied": 2, "skipped": 0}]
    record = {"rank": 0, "accepted_ranks": 2, "world": 2,
        "guard_policy": "common_global_skip_le_0.01_finite_positive_applied_per_client", "clients": clients,
        "groups": [{"client_ids": [3, 7], "applied": [1, 2], "skipped": [1, 0], "weights_finite": True,
                    "buffers_finite": True, "loss_finite": True}],
        "all_finite": True, "applied": 3, "skipped": 1, "zero_applied_client_ids": [],
        "global_info": {"skipped_frac": .001, "train_loss": 1.}}
    return record, counts


def test_valid_telemetry():
    record, counts = fixture(); assert check_telemetry(record, 0, np.array([3, 7]), counts) == (3, 1)


@pytest.mark.parametrize("change", ["finite", "steps", "group", "zero"])
def test_corrupt_telemetry_rejected(change):
    record, counts = fixture(); record = copy.deepcopy(record)
    if change == "finite": record["groups"][0]["weights_finite"] = False
    if change == "steps": counts[3, 0] += 128
    if change == "group": record["groups"][0]["applied"][0] = 2
    if change == "zero":
        record["clients"][0].update(applied=0, skipped=2)
        record["groups"][0].update(applied=[0, 2], skipped=[2, 0])
        record.update(applied=2, skipped=2)
    with pytest.raises(AssertionError): check_telemetry(record, 0, np.array([3, 7]), counts)
