# Proxy: projection trực giao đổi seed — căn cứ và cơ chế

## Căn cứ xây dựng

V3-O small/medium không giữ đủ lợi dense32. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/proxy-orthogonal-projection.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

Update dạng ma trận được chiếu lên cơ sở trực giao fresh rank1/rank32; EF parameter-space; control rank8 so PowerSGD.

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
