"""Byte budget per contact for the tiers in docs/research/de-xuat-huong-moi.md §3 (archive simulator, no training time).
    python scripts/budget_tiers.py mh"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import mobility as M
sim = {"rsu_radius_m": 800.0, "speed_min_mps": 5.0, "speed_max_mps": 20.0, "rate_min_Bps": 20000.0, "rate_max_Bps": 80000.0,
       "pos_noise_m": 8.0, "vel_noise_mps": 1.5, "t_ref_s": 10.0, "dev_min": 0.5, "dev_max": 2.0, "train_jitter": 0.1, "guard_s": 2.0, "sim_seed": 42}
rows = [1] * 20
taus, rates = [], []
for r in range(1, 51):
    for st in M.round_states(sim, r, rows):
        taus.append(st["tau_true_s"]); rates.append(st["rate_actual_Bps"])
taus, rates = np.array(taus), np.array(rates)
print("tau_true s: median %.1f p10 %.1f p90 %.1f" % tuple(np.percentile(taus, [50, 10, 90])))
for cs, rs, cap in [(0.01, 0.2, 32768), (0.1, 0.2, None), (0.1, 1.0, None), (1.0, 1.0, None)]:
    guard = 0.01 if cs < 1 else 2.0
    b = rates * rs * np.maximum(0, taus * cs - guard)
    if cap: b = np.minimum(b, cap)
    print(f"contact_scale {cs} rate_scale {rs} cap {cap}: bytes/contact median {np.median(b)/1e3:.1f} kB p10 {np.percentile(b,10)/1e3:.1f} p90 {np.percentile(b,90)/1e3:.1f} mean {b.mean()/1e3:.1f}")
