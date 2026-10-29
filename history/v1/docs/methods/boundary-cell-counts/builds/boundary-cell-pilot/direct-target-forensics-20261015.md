# BoundaryCells direct P2 — chẩn đoán payload thật

P2 direct r20 FAIL cả hai seed: group/private/gp0,504056/0,509389, kém Gaussian0,126833/0,131181.
Δown0,009266/0,011509; seed42 trượt own gate. Không nối adaptation P2 hoặc đổi headline để cứu.
[Terminal audit và round đầy đủ](../../../../archive/operations/continuation-2026-10-15.md).

Đọc target packet từ actual audited P2-FL r20 và P4-FL r2/r12; CRC/dictionary identity/payload SHA/source
và group fixture SHA đều kiểm. Group probe balanced128/class=2.048record, seed1729; không SGD/official test/HP search.
Hai seed có cùng payload SHA: counts/dictionary cố định không phụ thuộc training seed, nên đây là **một**
chẩn đoán carrier, không là hai probe thống kê độc lập.

| payload global từ full T | P2 | P4 |
|---|---:|---:|
| Eligible cells | 16/32 | 33/64 |
| Coverage group sample | 95,65% | 100% |
| Mean normalized target entropy, 1=uniform | 0,9146 | 0,9614 |
| Mean P(true class), uniform=0,0625 | 0,1110 | 0,1160 |
| Mean max target probability | 0,1502 | 0,1393 |
| Table-only diagnostic F1 macro | 0,12949 | 0,28017 |

[Artifact đủ10metric](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/probes/direct-target-forensics-20261015.json),
[script](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/scripts/diagnose_direct_targets.py).
Table-only score không là DAGSNet private/teacher score; soft targets không cần argmax đúng để có ích.

Payload P2 phủ phần lớn group nhưng chỉ mang phân bố lớp mờ từ hai cặp feature/cell thô. Kết hợp hard CE
với auxiliary soft CE có thể chủ yếu regularize/smooth logits, trong khi thiếu tín hiệu để giữ gain Gaussian.
Đây là giải thích được hỗ trợ bởi payload và contrast, chưa cô lập riêng bins/axes/teacher/gp; không chứng minh
toàn họ cell summaries thất bại. Counts và target tĩnh, thêm round không tạo thêm tri thức RSU trong carrier này;
SGD vẫn có thể thay private score, nhưng không cứu được locked r20 decision đã FAIL.

P4 tăng table discrimination/coverage dù target entropy cao hơn; teachers có thể dùng backbone66feature để fit
thay lookup trực tiếp. Đây là lý do còn giữ **các arm P4/teacher đã chốt**, không thêm tuning hoặc phiên P2.
Utility/cost từng arm và hai seed vẫn quyết định từ artifact; probe này không mở quota hay đổi gate.

Cập nhật terminal chiều15/10: các contrast P4direct/P2-P4teacher đã đủr20 cả hai seed và FAIL
necessary reference gate. Lý do kiểm thêm ở trên là rationale trước trial, không còn là việc pending.
[Utility và learned-teacher diagnostics](results/utility-report.md); không nối hoặc tune cell-size để cứu adaptation.
