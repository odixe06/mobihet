# proxy-carriers

Nhãn lịch sử `v3`; Build so sánh nhiều phương pháp; trạng thái từng arm tại danh mục đề xuất.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/pilot-v3.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Source đã chạy | [source/](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/source) |
| Probe và proof | [probes/](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/notebooks) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/scripts) |
| Logs | [v3-carrier-comparison/](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/logs) |

## Kết quả đã có

- [figures.md](results/figures.md)
- [paper.md](results/paper.md)
- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [runs.dvc](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/runs.dvc)
- [kaggle.dvc](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/kaggle.dvc)
- [smoke.dvc](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/smoke.dvc)
- [input.dvc](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/input.dvc)
- [resume_reference.dvc](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).

[Module riêng](../../../../../../archive/methods/proxy-compression/builds/proxy-carriers/source/mh/README.md): source canonical, byte/hash giữ nguyên; import cũ qua alias.
