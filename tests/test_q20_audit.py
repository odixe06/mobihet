"""Stored Q20 normalization boundary must pass; altered target content must fail."""
from pathlib import Path

import pytest
import torch

from methods.mobihet_study import decode_probability
from scripts.audit_q20 import check_decoded_probability

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def saved():
    path = ROOT / "artifacts/q20/mh-q5-mixed-class-s43/mh-q5-mixed-class-s43/models/r017.pt"
    if not path.exists(): pytest.skip("Q5r17 artifact not restored")
    return torch.load(path, weights_only=True, map_location="cpu")


@pytest.mark.parametrize("slug,r", [("mh-q5-mixed-class-s43", 17), ("mh-q6-mixed-input-s42", 14),
                                   ("mh-q9-input-sqrt-s43", 19)])
def test_stored_normalization_boundary(slug, r):
    path = ROOT / "artifacts/q20" / slug / slug / "models" / f"r{r:03}.pt"
    if not path.exists(): pytest.skip(f"{slug}r{r} artifact not restored")
    s = torch.load(path, weights_only=True, map_location="cpu")
    assert (s["bank_target"] - decode_probability(s["bank_body"])).abs().max() > 2e-7
    assert check_decoded_probability(s["bank_target"], s["bank_body"])["max_cpu_difference"] > 2e-7


@pytest.mark.parametrize("change", ["class_identity", "rescale", "small_probability"])
def test_rejects_altered_target(saved, change):
    target = saved["bank_target"].clone()
    if change == "class_identity": target = target.roll(1, 1)
    elif change == "rescale": target[718] *= 1.0001
    else:
        target[718, 0] += 1e-5
        target[718] /= target[718].sum()
    with pytest.raises(AssertionError, match="outside fp32 normalization interval"):
        check_decoded_probability(target, saved["bank_body"])


def test_zero_body_entry_requires_exact_zero_target():
    body = torch.eye(16, dtype=torch.float16)
    target = body.float()
    check_decoded_probability(target, body)
    target[0, 1] = 1e-12
    with pytest.raises(AssertionError, match="outside fp32 normalization interval"):
        check_decoded_probability(target, body)
