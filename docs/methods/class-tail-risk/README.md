# ClassTail — histogram CE theo lớp, objective tail-risk local

Chủ dự án chốt 15/10: α=0,25, λ=0,25, seed42/43, lịch20; hai round CE warm-up.
32 bin log + overflow, quantile mép trên; probe FAIL thì không mở r20.

[Nguồn, cơ chế và phản biện](../../research/new-directions-2026-10-15.md) ·
[Contract và trạng thái](builds/class-tail-pilot/README.md).

Không teacher/sampler/book mới; mọi model DAGSNet. Shared threshold điều phối độ ưu tiên local,
chưa chứng minh truyền ranh giới peer hoặc cải thiện F1. Không kế thừa theorem FedAvg/FedCVaR.
