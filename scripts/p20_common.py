"""Predeclared P20 cells; no metric-dependent selection or hidden architecture changes."""
from methods.mobihet_study import MODES

CELLS = MODES


def settings(mode):
    no = mode in {"real", "real_no_la", "augment", "project_augment", "bank_hard"}
    real = mode in {"gate", "margin", "strength"}
    return dict(arm="none", synth="off" if mode in {"real", "real_no_la"} else "copula", synth_rows=64,
                synth_target="soft", kd_weight="fixed", q_codec="mean_fp16", q_collect="online", q_agg="count",
                kd_support="off", anchor="off", kd_bins=1, no_kd=no, real_kd=real,
                real_kd_gate=real, study_mode=mode, teacher_rows=1024, proxy_steps=200)
