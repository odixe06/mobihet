"""Recovery tolerates rare masked skips but cannot admit all-skipped or nonfinite clients."""
import pytest
from methods.mobihet_vrecover import admissible


def test_rare_skip_is_admissible():
    assert admissible({"all_finite": True, "zero_applied_client_ids": []},
                      {"skipped_frac": 3 / 513368, "train_loss": 1.})


@pytest.mark.parametrize("observation,info", [
    ({"all_finite": True, "zero_applied_client_ids": [716]}, {"skipped_frac": 1e-6, "train_loss": 1.}),
    ({"all_finite": False, "zero_applied_client_ids": []}, {"skipped_frac": 0., "train_loss": 1.}),
    ({"all_finite": True, "zero_applied_client_ids": []}, {"skipped_frac": .011, "train_loss": 1.}),
    ({"all_finite": True, "zero_applied_client_ids": []}, {"skipped_frac": float("nan"), "train_loss": 1.}),
    ({"all_finite": True, "zero_applied_client_ids": []}, {"skipped_frac": 0., "train_loss": float("inf")}),
])
def test_failed_client_or_nonfinite_evidence_is_rejected(observation, info):
    assert not admissible(observation, info)
