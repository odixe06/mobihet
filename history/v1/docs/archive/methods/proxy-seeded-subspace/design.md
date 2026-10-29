# Proxy trong cơ sở sinh từ seed — căn cứ và cơ chế

## Căn cứ xây dựng

V2-B teacher gate FAIL; không KD/ZO/contact. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/proxy-seeded-subspace.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

DAGSNet proxy theta0+Za; trao đổi hệ số FO K512/K1024, không trao đổi record. Teacher qua mới mở private KD/ZO.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [b-proxy-seed.md](../../research/de-xuat-v2/b-proxy-seed.md)
- [pilot-v2.md](../../research/pilot-v2.md)
- [pilot-v2.md](../../research/probes/pilot-v2.md)
- [v2_methods.py](../../../../core/shared_prior_and_proxy.py)
- [report.md](../shared-comparisons/builds/countsketch-subspace-ratio/results/report.md)
- [rebuild.md](../shared-comparisons/builds/countsketch-subspace-ratio/results/rebuild.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
