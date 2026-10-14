# ResponseRisk — consistency chống receiver shortcut

FL mới: local DAGSNet đo JS dưới swap receiver noise/profile; RSU gộp response moments theo lớp,
phát trọng số consistency. Own real/perturbed records đều ở xe.
Chủ dự án cho swap đồng bộ9feature cùng lớp/cùng client/khác receiver, qua fidelity gate.
Feature stress này chưa tự là causal intervention đúng vật lý.

**Quyết định mới nhất 13/10:** chỉ kiểm local augmentation trước; shared/own response weights
chưa được mở pilot. Giữ λJS=0,1, swap đồng bộ 9 feature cùng lớp/cùng client/khác receiver,
probe receiver-group g%20=1 loại khỏi SGD ở mọi arm, holdout cũ giữ nguyên.
Đợt local là kiểm thành phần; chưa chứng minh hoặc thay thế phương pháp FL.

- [Cơ chế và controls](../../docs/research/new-fl-primary-sources-2026-10-13.md).
- [Dự báo](../../docs/research/performance-feasibility-2026-10-13.md).

**Dự báo thấp/vừa:** receiver shortcut có bằng chứng; lợi pooled ngoài uniform/own chưa biết.
Trạng thái bounded fidelity probe xong; chưa pilot. Transform đầu chỉ một block đã được phép;
không dùng dự toán G4 của ghi chú cho byte G1. Response stats trên subset chưa train shadow teacher;
split nội bộ và controls cần khóa trước train.

[Probe fidelity](../../papers/receiver-response/fidelity-probe.json):4toy cases gồm missing/nonfinite
đều giữ extractor closure;40.951/40.960sample row có donor cùng lớp/client khác hashedreceiver.
55feature ngoài block giữ nguyên; không upload record.9row thiếu donor phải skip transform và báo.
PASS này không bảo đảm vật lý/causal hoặc IDS utility; dự báo vẫn thấp/vừa.
