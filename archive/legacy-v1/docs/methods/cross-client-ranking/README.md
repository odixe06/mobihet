# CrossClientRank — score moments giữa private models

Owner chốt15/10: probe train-only hai seed42/43 trên checkpoint CE-r2 đã audit; không SGD mới.
RSU nhận positive/negative score moments theo lớp; mỗi xe kiểm gradient ranking trên real T của mình.
Không teacher/sampler/book mới, không feature-boundary claim hoặc guarantee AUC của shared model.

[Nguồn và objective](../../archive/research/new-directions-2026-10-15-afternoon.md) ·
[Contract, provenance, kết quả](builds/rank-probe/README.md).
[Utility r20 đã owner chốt/push](builds/rank-utility/README.md); cost gate trước tiếp lịch20.

Primary utility đã khóa: private/group/gp theo true-class prior cố định mọi arm/reference;
head diagnostic. Ranking không có CE-target-mass prior hợp thức. Probe PASS chưa là utility PASS;
probe FAIL thì dừng adaptation, không đổi score/basis/gate để tìm thắng.
