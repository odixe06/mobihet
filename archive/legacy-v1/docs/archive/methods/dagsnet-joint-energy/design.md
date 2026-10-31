# Joint-energy bằng DAGSNet — căn cứ và cơ chế

## Căn cứ xây dựng

V10 Gaussian-base+MH hợp lệ về base nhưng không qua cổng runtime/sampler. Bằng chứng và giới hạn tại [bài học phương pháp](../../../lessons/dagsnet-joint-energy.md) và [tổng hợp bằng chứng](../../overview/evidence-summary.md). Không suy codec/probe PASS thành utility PASS.

## Cách thực hiện

DAGSNet tạo năng lượng chung feature/label; sinh input synthetic tại xe, tham số model được gộp.

## Thiết kế gốc, code và bằng chứng

- [v3-knowledge-sources.md](../../research/v3-knowledge-sources.md)
- [probe_v3_energy.py](../../../../archive/methods/proxy-compression/builds/proxy-carriers/scripts/probe_proxy_carrier_energy.py)
- [energy-probe.json](../../../../archive/methods/proxy-compression/builds/proxy-carriers/probes/energy-probe.json)

Không raw/perturbed record rời xe. Mọi learned model hiện DAGSNet; R2 khác kiến trúc chỉ ở thiết kế. Pilot StageU không chứng minh contact hữu hạn, secure aggregation hoặc DP.

[Kết quả và các build](README.md).
