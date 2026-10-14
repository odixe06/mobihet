# Residual hàm từ nhãn local trên anchor

[Danh mục](../README.md) · [Bài học](lessons.md)

Nhãn lịch sử: K2/V9. Teacher-first M64/512 cả hai seed đã audit26arm-seed×20round:
cost PASS, quality paired FAIL. Không mở private KD/contact/full hoặc rerun y nguyên.
Đồng bộprobe/teacher cótínhbyte. K2b V5 làcơ chếkhác đãpilot.

## Cơ chế FL

Tính residual/gradient hàm trên anchor synthetic chung, gộp thống kê để cập nhật teacher DAGSNet.

## Tài liệu liên quan

- [v3-knowledge-sources.md](../../docs/research/v3-knowledge-sources.md)

Mọi learned model giữ DAGSNet; không raw record rời xe. Chưa có bằng chứng cải thiện F1/R1/R2.

## Triển khai V9

Teacher-first M64/512 đãchốt; [manifest](../../docs/research/pilot-v9-residual.md).
[Source](../../mh/v9_anchor.py), [driver](../../mh/v9_run.py),
[verifier](../../scripts/verify_teacher.py). Local13arms/3exactresumes đãPASS;
finalsource/W&B-cost-controller PASS trướcpush16:48UTC+7 ngày2026-10-12.
[Ledger](../../papers/v9/rebuild.md), [expansion](../../docs/operations/expansion-2026-10-12.md).
Khôngutility/R1/R2claim trướcterminalaudit.
[Report hai seed](../../papers/v9/report.md), [cổng](../../papers/v9/gates.json).
