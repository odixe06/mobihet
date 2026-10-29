# Residual hàm từ nhãn local trên anchor — căn cứ và cơ chế

## Căn cứ xây dựng

V9 M64/512 không qua teacher quality; không privateKD/full y nguyên. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/anchor-function-residual.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

Tính residual/gradient hàm trên anchor synthetic chung, gộp thống kê để cập nhật teacher DAGSNet.

## Thiết kế gốc, code và bằng chứng

- [v3-knowledge-sources.md](../../research/v3-knowledge-sources.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
