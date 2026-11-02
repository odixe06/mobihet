"""X20 cells, locked before launch (design: docs/research/screening-x20.md).

core/bank combine the W20 components that passed their gates (step floor, compact bootstrap) without re-testing them;
live_real_nckd_local changes ONE component of W2/W3 (non-target set restricted to the vehicle's classes); full is read
only if live_real_nckd_local passes its gate.
"""
from scripts.w20_common import SOURCES as W_SOURCES

PASSED = {"synth_step_floor": 16, "w_boot": "compact"}   # W20 gates: step_floor PASS, boot_compact within spread
CELLS = {
    "core": {"mode": "stream", "r_kind": "live", "w": dict(PASSED)},                              # vs R0-R3, W6-W9
    "bank": {"mode": "mixed_input", "r_kind": "live", "w": dict(PASSED)},                         # vs Q6/Q7, X core
    "live_real_nckd_local": {"mode": "real", "r_kind": "live", "w": {"w_real": "nckd_local"}},    # vs W2/W3, Q6/Q7
    "full": {"mode": "real", "r_kind": "live", "w": {"w_real": "nckd_local", **PASSED}},          # vs X bank
}

SOURCES = [*W_SOURCES, "scripts/x20_common.py"]
