# Nén update proxy bằng CountSketch — căn cứ và cơ chế

## Căn cứ xây dựng

V2-A sketch residual khuếch đại collision; không rerun y nguyên. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/proxy-countsketch.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

Sketch update proxy, gộp rồi decode patch; so kích thước nhỏ/vừa, có/không error feedback. Private DAGSNet nhận KD.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [a-proxy-sketch.md](../../research/de-xuat-v2/a-proxy-sketch.md)
- [pilot-v2.md](../../research/pilot-v2.md)
- [pilot-v2.md](../../research/probes/pilot-v2.md)
- [v2_math.py](../../../../core/vector_codecs.py)
- [v2_methods.py](../../../../core/shared_prior_and_proxy.py)
- [report.md](../shared-comparisons/builds/countsketch-subspace-ratio/results/report.md)
- [rebuild.md](../shared-comparisons/builds/countsketch-subspace-ratio/results/rebuild.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
