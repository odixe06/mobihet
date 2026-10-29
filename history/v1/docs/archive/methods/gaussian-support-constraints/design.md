# Gaussian giữ atom, profile và first-session — căn cứ và cơ chế

## Căn cứ xây dựng

V4-S fidelity PASS không dẫn tới utility hơn diag. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/gaussian-support-constraints.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

Moment gộp + atom probabilities; sampler analytic giữ binary/profile categorical và first-session zero-imputation. DAGSNet vẫn nhận66feature.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [pilot-v4-support.md](../../research/pilot-v4-support.md)
- [pilot-v4.md](../../research/probes/pilot-v4.md)
- [v4_support.py](../../../../mh/v4_support.py)
- [v4_schema.py](../../../../core/feature_support_schema.py)
- [report.md](../gaussian-sampling/builds/support-mala/results/report.md)
- [rebuild.md](../gaussian-sampling/builds/support-mala/results/rebuild.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
