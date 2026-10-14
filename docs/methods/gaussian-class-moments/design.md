# Gaussian từ moment lớp gộp — căn cứ và cơ chế

## Căn cứ xây dựng

P1 covariance/quần thể có lợi; full r9–r10 đã audit, chưa r50. Bằng chứng và giới hạn tại [bài học phương pháp](lessons.md) và [tổng hợp bằng chứng](../../archive/overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

Xe gửi tổng theo lớp; RSU gộp moment, phát Gaussian lớp; xe train CE real + CE synthetic. Moment là thống kê, không learned model.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [de-xuat-huong-moi.md](../../archive/research/de-xuat-huong-moi.md)
- [h2-input-stats.md](../../archive/research/probes/h2-input-stats.md)
- [pilot-p1.md](../../archive/research/probes/pilot-p1.md)
- [pilot-p2.md](../../archive/research/probes/pilot-p2.md)
- [pilot.py](../../../mh/pilot.py)
- [report.md](../../archive/methods/shared-baselines/builds/pilot-baselines/results/report.md)
- [report.md](../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/results/report.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
