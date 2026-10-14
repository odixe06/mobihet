# hybrid-full

Nhãn lịch sử `hybrid-full`; Pilot hai seed PASS; full đang nối tới r50.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](protocol.md) · [Danh mục đề xuất](../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Source đã chạy | [source/](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/source) |
| Probe và proof | [probes/](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/notebooks) |
| Artifact gồm smoke/checkpoint/prediction | [artifacts/](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/artifacts) |
| Scripts riêng | [scripts/](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/scripts) |
| Logs | [full-50rounds/](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/logs) |

## Kết quả đã có

- [feasibility-20261014.md](results/feasibility-20261014.md)
- [latest-rounds.md](results/latest-rounds.md)
- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [kaggle.dvc](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/kaggle.dvc)
- [smoke.dvc](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/smoke.dvc)
- [diagnostics.dvc](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/diagnostics.dvc)
- [input.dvc](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/input.dvc)
- [resume_reference.dvc](../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../operations/project-layout.md).
