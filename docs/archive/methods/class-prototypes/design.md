# Prototype trung bình lớp — căn cứ và cơ chế

## Căn cứ xây dựng

Mean prototype không chở đủ ranh giới; không ưu tiên biến thể mean. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/class-prototypes.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

Gộp vector trung bình theo lớp, dùng regularization/prototype loss ở xe.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [de-xuat-huong-moi.md](../../research/de-xuat-huong-moi.md)
- [pilot-p1.md](../../research/probes/pilot-p1.md)
- [pilot.py](../../../../core/pilot_runtime.py)
- [report.md](../shared-baselines/builds/pilot-baselines/results/report.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
