# Proxy: PowerSGD hai lượt — căn cứ và cơ chế

## Căn cứ xây dựng

V3-P adaptive basis có đóng góp nhưng compression utility FAIL. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/proxy-powersgd.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

RSU/xe gộp hai lượt low-rank rank8; broadcast basis giữa lượt; residual chỉ commit khi cohort đủ; fp32, private+KD.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế gốc, code và bằng chứng

- [v3-carriers-sources.md](../../research/v3-carriers-sources.md)
- [pilot-v3.md](../../research/pilot-v3.md)
- [v3_math.py](../../../../mh/v3_math.py)
- [v3_methods.py](../../../../mh/v3_methods.py)
- [report.md](../proxy-compression/builds/proxy-carriers/results/report.md)
- [rebuild.md](../proxy-compression/builds/proxy-carriers/results/rebuild.md)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
