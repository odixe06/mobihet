# BoundaryCells — audited fixed-r20 utility

Primary private/group/gp target-mass đã khóa. Không dùng official test, best round hoặc đổi headline.

| seed | candidate | rounds | group gp | delta strongest reference | gate |
|---|---|---:|---:|---:|---|
| 42 | bc_direct2 | 20/20 | 0.504056 | -0.126833 | FAIL necessary reference/benign gate; no extension needed |
| 42 | bc_direct4 | 20/20 | 0.512693 | -0.118196 | FAIL necessary reference/benign gate; no extension needed |
| 42 | bc_teacher2 | 20/20 | 0.501386 | -0.129503 | FAIL necessary reference/benign gate; no extension needed |
| 42 | bc_teacher4 | 20/20 | 0.506774 | -0.124115 | FAIL necessary reference/benign gate; no extension needed |
| 43 | bc_direct2 | 20/20 | 0.509389 | -0.131181 | FAIL necessary reference/benign gate; no extension needed |
| 43 | bc_direct4 | 20/20 | 0.516368 | -0.124203 | FAIL necessary reference/benign gate; no extension needed |
| 43 | bc_teacher2 | 20/20 | 0.505897 | -0.134674 | FAIL necessary reference/benign gate; no extension needed |
| 43 | bc_teacher4 | 20/20 | 0.510874 | -0.129697 | FAIL necessary reference/benign gate; no extension needed |

[10metrics mọi round](utility-metrics-all.csv) · [Per-class F1](utility-per-class-f1.csv) · [Gates/provenance](utility-gates.json).

Thiếu controls không được gọi full paired-r20 completion. Candidate đã r20 và trượt necessary strong-reference gate
thì không cần huấn luyện nốt controls để xác nhận không đủ điều kiện tăng quy mô.
Cost algorithm và full round so plain CE r3–20; không dummy-inflate denominator.
Audit gồm source/state/payload/predictions→10metrics; strict reconstruction client0 mỗi round, không sampled saved-weight inference.
[Teacher/private diagnostics](utility-failure-analysis.json) và direct-table forensic mô tả target/teacher bottleneck;
teacher failure chưa được quy cho một nguyên nhân duy nhất; không mặc định thêm round sửa target tĩnh.
