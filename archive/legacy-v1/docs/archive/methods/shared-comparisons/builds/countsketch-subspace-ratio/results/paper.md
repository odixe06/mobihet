# MobiHet V2 — các adaptation FL để kiểm khả thi

Đề xuất dự án, không reproduction nguyên một paper. Thiết kế và nguồn: [index V2](../../../../../research/de-xuat-v2/README.md),
[nguồn gốc](../../../../../research/de-xuat-v2/nguon-nghien-cuu.md). Cấu hình/cổng khóa: [pilot V2](../../../../../research/pilot-v2.md).

A: CountSketch tuyến tính của weighted local-epoch delta; residual tại RSU; decoded sparse patch.
Khác FetchSGD stochastic-gradient; không mang bảo đảm hội tụ sang local-epoch AdamW/BN/codec.
B: fixed scalar basis θ=θ₀+Za, first-order coefficient FedAvg. Lấy ý tưởng seed accumulator và projected-gradient,
khác changing-basis matrix accumulators của FedKRSO; DAGSNet từ đầu, không pretrained LLM.
C: logistic data/noise16task trên DAGSNet để học p_y^τ/q_y, dùng clipped class-normalized synthetic weights.
Không raw density p/q, không JEM/SGLD, không teacher soft-label.

Mọi learned model DAGSNet66→16. R2 chỉ thiết kế qua KD/weighted CE. R3 local record/perturbed record không rời xe;
thống kê/cập nhật chỉ cohort-total, secure aggregation mô phỏng, chưa DP/cryptography.
