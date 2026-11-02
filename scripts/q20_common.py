"""Predeclared Q20 factors, no test-dependent configuration."""
from methods.mobihet_next import VARIANTS
from scripts.p20_common import settings as parent_settings

CELLS = VARIANTS


def settings(mode):
    assert mode in CELLS
    return parent_settings("class") | {"next_mode": mode, "real_kd": mode in {"real", "margin", "strength"},
                                     "real_kd_gate": mode in {"real", "margin", "strength"}}
