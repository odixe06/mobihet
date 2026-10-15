# CountSketch repair — audited utility and partial feasibility

Decoded residual contraction/numerical stability không chứng minh proxy quality. Primary group/private/gp giữ nguyên.

| seed | arm | rounds/20 | private gp | private head diagnostic |
|---|---|---:|---:|---:|
| 42 | a_dense | 20/20 | 0.661926 | 0.670272 |
| 42 | a_medium_contract_ef | 17/20 | 0.214137 | 0.461266 |
| 42 | a_medium_noef | 20/20 | 0.270082 | 0.498471 |
| 42 | a_own | 20/20 | 0.469440 | 0.446580 |
| 42 | a_small_contract_ef | 20/20 | 0.213280 | 0.459551 |
| 42 | a_small_noef | 20/20 | 0.210803 | 0.459570 |
| 42 | local | 20/20 | 0.464340 | 0.442355 |
| 43 | a_dense | 20/20 | 0.645405 | 0.654578 |
| 43 | a_medium_contract_ef | 14/20 | 0.213502 | 0.467340 |
| 43 | a_medium_noef | 20/20 | 0.240748 | 0.484031 |
| 43 | a_own | 20/20 | 0.467270 | 0.443451 |
| 43 | a_small_contract_ef | 20/20 | 0.197996 | 0.455346 |
| 43 | a_small_noef | 20/20 | 0.205401 | 0.443381 |
| 43 | local | 20/20 | 0.460972 | 0.438842 |

[Toàn10metrics](metrics-all.csv) · [Per-class](per-class-f1.csv) · [Fixed-r20 gates/partial curves](gates.json).

Các arm chưa20 không được gọi strict-r20 FAIL. Không đổi gp sang head để cứu gate.
Sparse patch coverage/ties chỉ mô tả carrier; không cô lập raw-update error hoặc bác cả họ CountSketch/EF.
Không có official test hoặc inference từ saved weights trong audit này.
