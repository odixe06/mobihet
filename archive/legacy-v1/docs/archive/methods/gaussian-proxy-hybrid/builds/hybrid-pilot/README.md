# hybrid-pilot

Nhãn lịch sử `v8`; Pilot hai seed PASS; full đang nối tới r50.

[Căn cứ/cơ chế](../../../../../methods/gaussian-proxy-hybrid/design.md) · [Cấu hình và cổng](../../../../research/pilot-v8-hybrid.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Probe và proof | [probes/](../../../../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/notebooks) |
| Artifact gồm smoke/checkpoint/prediction | [artifacts/](../../../../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/artifacts) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/scripts) |
| Logs | [v8-pilot/](../../../../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/logs) |

## Kết quả đã có

- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [kaggle.dvc](../../../../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/kaggle.dvc)
- [smoke.dvc](../../../../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/smoke.dvc)
- [input.dvc](../../../../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/input.dvc)
- [resume_reference.dvc](../../../../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).

[Module riêng](../../../../../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/source/mh/README.md): source canonical, byte/hash giữ nguyên; import cũ qua alias.
