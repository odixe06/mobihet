# ClassTail — audited results

Paired r2 mechanism PASS; calibration không chứng minh utility. R20 chỉ chốt từ đủ hai seed đã audit.

| seed | classes tail10–40% | pooled/own membership diff |
|---|---:|---:|
| 42 | 16/16 | 0.145720 |
| 43 | 16/16 | 0.136078 |

Audited evaluation rows đủ10metrics: 2080; paired r20: True.
[10metrics mọi round](metrics-all.csv) · [Per-class F1](per-class-f1.csv) · [Locked gates](gates.json).

| seed | arm | rounds | group gp primary | group head diagnostic |
|---|---|---:|---:|---:|
| 42 | ct_gaussian | 20 | 0.541142 | 0.525944 |
| 42 | ct_dml | 20 | 0.658373 | 0.590910 |
| 42 | ct_fixed | 20 | 0.463862 | 0.441320 |
| 42 | ct_shuffle | 20 | 0.463154 | 0.443040 |
| 42 | ct_own | 20 | 0.463289 | 0.441457 |
| 42 | ct_fl | 20 | 0.461645 | 0.442246 |
| 42 | ct_classmass | 20 | 0.462358 | 0.443132 |
| 42 | ct_local | 20 | 0.464328 | 0.442355 |
| 43 | ct_fixed | 20 | 0.461292 | 0.438619 |
| 43 | ct_own | 20 | 0.461293 | 0.438663 |
| 43 | ct_fl | 20 | 0.458453 | 0.438404 |
| 43 | ct_dml | 20 | 0.650172 | 0.585647 |
| 43 | ct_classmass | 20 | 0.458617 | 0.438806 |
| 43 | ct_local | 20 | 0.460985 | 0.438842 |
| 43 | ct_shuffle | 20 | 0.459487 | 0.438979 |
| 43 | ct_gaussian | 20 | 0.545060 | 0.529681 |

Paired utility PASS: False; scale eligible: False.
Pooled/class-mass difference≤0,005 cả hai seed: True.
[Failure diagnostics](failure-analysis.json): late thresholds ở bin floor và tail nhỏ mô tả lượng tử hóa;
không cô lập nguyên nhân duy nhất. Pooled không hơn own/shuffle/class-mass thì chưa chứng minh tri thức hard-example peer.
Không kéo dài hoặc đổi alpha/lambda/bin sau FAIL; không suy cả họ tail-risk không thể hữu ích.

Benign guard so local và reference Gaussian/DML mạnh nhất; không chọn best round.
Cost báo cả algorithm và wall round r3–20; baseline CE không scan ở các round đó.
Class-mass CE ngang pooled trong≤0,005 cả hai seed thì chưa quy lợi cho hard-example mechanism.
R1/contact/DP chưa kiểm. Source/fixture/state/predictions đã audit; không sampled saved-weight inference.
