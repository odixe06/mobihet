# Teacher từ kernel soft target trên anchor chung — căn cứ và cơ chế

## Căn cứ xây dựng

V5-H coverage cao đi cùng target gần uniform; teacher gate FAIL. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/kernel-anchor-distillation.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

Xe tính kernel sum theo lớp trên Gaussian anchors chung; SecureSum tạo bảng soft target; teacher DAGSNet train bảng, không classifier bảng triển khai.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [pilot-v5-anchors.md](../../research/pilot-v5-anchors.md)
- [v3-knowledge-sources.md](../../research/v3-knowledge-sources.md)
- [v5_anchor.py](../../../../mh/v5_anchor.py)
- [report.md](../synthetic-knowledge/builds/anchor-synthetic-books/results/report.md)
- [rebuild.md](../synthetic-knowledge/builds/anchor-synthetic-books/results/rebuild.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
