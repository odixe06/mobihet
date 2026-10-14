# BoundaryCells — đếm nhãn theo ô input

Hướng FL mới: additive label counts trong các ô feature cố định → tổng cohort tại RSU → bảng soft targets.
Private DAGSNet học trên **record của chính xe**. Cả direct loss và teacher DAGSNet trung gian được phép.
Không noise anchors, Gaussian synthesis, learned rule network hoặc trao đổi record.

- [Thiết kế và cổng](design.md).
- [Dự báo trước kiểm](../../docs/research/performance-feasibility-2026-10-13.md).
- [Nguồn primary](../../docs/research/new-fl-primary-sources-2026-10-13.md).
- [Probe đã đo](../../papers/boundary-cells/bounded-probe.json); [source](../../papers/boundary-cells/proj/probe.py).

**Trạng thái:** bounded probe xong; chưa DAGSNet utility pilot, chưa R1/contact/R2 proof.
4 cặp có dấu hiệu tri thức hơn own/shuffle nhưng target vẫn rất mềm. Không gọi table diagnostic là IDS model.
