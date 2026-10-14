# DAGSNet học tỷ số mật độ cho Gaussian — căn cứ và cơ chế

## Căn cứ xây dựng

V2-C ESS thấp, không hơn diag/own/shuffle theo cổng. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/gaussian-density-ratio.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

DAGSNet NCE real làm trơn local tau0.05 so Gaussian; weight synthetic theo ratio clip10, chuẩn hóa lớp. Tham số tỷ số gộp, không gửi record.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [c-density-ratio.md](../../research/de-xuat-v2/c-density-ratio.md)
- [pilot-v2.md](../../research/pilot-v2.md)
- [pilot-v2.md](../../research/probes/pilot-v2.md)
- [report.md](../shared-comparisons/builds/countsketch-subspace-ratio/results/report.md)
- [rebuild.md](../shared-comparisons/builds/countsketch-subspace-ratio/results/rebuild.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
