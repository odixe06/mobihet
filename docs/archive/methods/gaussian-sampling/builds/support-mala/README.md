# support-mala

Nhãn lịch sử `v4`; Build so sánh nhiều phương pháp; trạng thái từng arm tại danh mục đề xuất.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/pilot-v4-support.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Probe và proof | [probes/](../../../../../../archive/methods/gaussian-sampling/builds/support-mala/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/gaussian-sampling/builds/support-mala/notebooks) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/gaussian-sampling/builds/support-mala/scripts) |
| Logs | [v4-support-mala/](../../../../../../archive/methods/gaussian-sampling/builds/support-mala/logs) |

## Kết quả đã có

- [figures.md](results/figures.md)
- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [runs.dvc](../../../../../../archive/methods/gaussian-sampling/builds/support-mala/runs.dvc)
- [kaggle.dvc](../../../../../../archive/methods/gaussian-sampling/builds/support-mala/kaggle.dvc)
- [smoke.dvc](../../../../../../archive/methods/gaussian-sampling/builds/support-mala/smoke.dvc)
- [input.dvc](../../../../../../archive/methods/gaussian-sampling/builds/support-mala/input.dvc)
- [resume_reference.dvc](../../../../../../archive/methods/gaussian-sampling/builds/support-mala/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).

[Module riêng](../../../../../../archive/methods/gaussian-sampling/builds/support-mala/source/mh/README.md): source canonical, byte/hash giữ nguyên; import cũ qua alias.
