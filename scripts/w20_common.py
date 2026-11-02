"""W20 cells, locked before launch. One component changed per cell against an audited Q/R control (see screening-w20)."""

# step floor 16 = real steps of a 1024-row client at batch 64; fixed a priori, not chosen on any evaluation split
CELLS = {
    "replay_real_nckd": {"mode": "real", "r_kind": "replay", "w": {"w_real": "nckd"}},   # vs R4/R5 noReal, R6/R7 full KD
    "live_real_nckd": {"mode": "real", "r_kind": "live", "w": {"w_real": "nckd"}},       # vs Q6/Q7 noReal, Q10/Q11 full KD
    "synth_nckd": {"mode": "stream", "r_kind": "live", "w": {"w_synth": "nckd"}},        # vs Q0/R0/R2, Q1/R1/R3
    "step_floor": {"mode": "stream", "r_kind": "live", "w": {"synth_step_floor": 16}},   # vs same stream sessions
    "boot_compact": {"mode": "stream", "r_kind": "live", "w": {"w_boot": "compact"}},    # vs same stream sessions
}

SOURCES = ["core/local.py", "core/fl.py", "core/grouped.py", "core/optim.py", "methods/mobihet.py", "methods/mobihet_study.py",
           "methods/mobihet_study_wire.py", "methods/mobihet_next.py", "methods/mobihet_rstudy.py",
           "methods/mobihet_rstudy_data.py", "methods/mobihet_wstudy.py", "scripts/run_p21_recovery.py", "scripts/run_fl.py",
           "scripts/run_q20.py", "scripts/q20_common.py", "scripts/w20_common.py", "scripts/preflight_w20.py",
           "tests/test_wstudy.py"]
