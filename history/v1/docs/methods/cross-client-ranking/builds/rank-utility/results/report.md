# CrossClientRank — terminal audited utility

Pairedr20: False; paired utility PASS: None; scale eligible: None.

| seed | arm | logical round | group gp |
|---|---|---:|---:|
| 42 | cr_classweight | 20 | 0.464933 |
| 42 | cr_dml | 16 | 0.633531 |
| 42 | cr_fixed | 20 | 0.464976 |
| 42 | cr_fl | 20 | 0.466075 |
| 42 | cr_gaussian | 20 | 0.571137 |
| 42 | cr_local | 20 | 0.464340 |
| 42 | cr_own | 20 | 0.466336 |
| 42 | cr_shuffle | 20 | 0.464880 |
| 43 | cr_fl | 3 | 0.462332 |
| 43 | cr_local | 3 | 0.463121 |

[10metrics](metrics-all.csv) · [Per-class](per-class-f1.csv) · [Gates/provenance](gates.json).
Sáu forks chỉ báo newly measuredr3–20; r1/r2 gốc không được gọi gp mới. References báo actualr1–20.
R3 cost preflight không thay pairedr20 utility hoặc post-warm cost. Không đổi headline/HP/round sau FAIL.
NNLS control clip[0,4], không exact-gradient-matched; probe không là calibration-equivalence proof.
Audit nguồn/state/BN/wire/predictions→10metrics; không independent saved-weight inference hoặc NNLS refit.
