# Gaussian được vận chuyển bằng MALA — căn cứ và cơ chế

## Căn cứ xây dựng

V4-M không qua utility; clip floor làm score correction bằng0 ở nhiều lớp. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/gaussian-mala-transport.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

DAGSNet ratio gộp hướng dẫn MALA tại xe; Gaussian base, eta0.01,20burn-in rồi một transition/batch. Không gửi chain/record.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [v3-knowledge-sources.md](../../research/v3-knowledge-sources.md)
- [pilot-v4-mala.md](../../research/pilot-v4-mala.md)
- [pilot-v4.md](../../research/probes/pilot-v4.md)
- [v4_mala.py](../../../../mh/v4_mala.py)
- [report.md](../gaussian-sampling/builds/support-mala/results/report.md)
- [rebuild.md](../gaussian-sampling/builds/support-mala/results/rebuild.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
