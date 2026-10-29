# countsketch-subspace-ratio

Nhãn lịch sử `v2`; Build so sánh nhiều phương pháp; trạng thái từng arm tại danh mục đề xuất.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/pilot-v2.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Probe và proof | [probes/](../../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/notebooks) |
| Artifact gồm smoke/checkpoint/prediction | [artifacts/](../../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/artifacts) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/scripts) |
| Logs | [v2-sketch-subspace-ratio/](../../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/logs) |

## Kết quả đã có

- [paper.md](results/paper.md)
- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [runs.dvc](../../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/runs.dvc)
- [kaggle.dvc](../../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/kaggle.dvc)
- [smoke.dvc](../../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/smoke.dvc)
- [input.dvc](../../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/input.dvc)
- [resume_reference.dvc](../../../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).
