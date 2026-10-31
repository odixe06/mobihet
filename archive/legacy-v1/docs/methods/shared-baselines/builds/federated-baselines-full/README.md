# federated-baselines-full

Nhãn lịch sử `full-validation`; Build so sánh nhiều phương pháp; trạng thái từng arm tại danh mục đề xuất.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](protocol.md) · [Danh mục đề xuất](../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Source đã chạy | [source/](../../../../../methods/shared-baselines/builds/federated-baselines-full/source) |
| Probe và proof | [probes/](../../../../../methods/shared-baselines/builds/federated-baselines-full/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../methods/shared-baselines/builds/federated-baselines-full/notebooks) |
| Artifact gồm smoke/checkpoint/prediction | [artifacts/](../../../../../methods/shared-baselines/builds/federated-baselines-full/artifacts) |
| Scripts riêng | [scripts/](../../../../../methods/shared-baselines/builds/federated-baselines-full/scripts) |
| Logs | [full-h1-h2-50rounds/](../../../../../methods/shared-baselines/builds/federated-baselines-full/logs) |

## Kết quả đã có

- [latest-rounds.md](results/latest-rounds.md)
- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [mount-output.dvc](../../../../../methods/shared-baselines/builds/federated-baselines-full/mount-output.dvc)
- [kaggle.dvc](../../../../../methods/shared-baselines/builds/federated-baselines-full/kaggle.dvc)
- [smoke.dvc](../../../../../methods/shared-baselines/builds/federated-baselines-full/smoke.dvc)
- [input.dvc](../../../../../methods/shared-baselines/builds/federated-baselines-full/input.dvc)
- [resume_reference.dvc](../../../../../methods/shared-baselines/builds/federated-baselines-full/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../operations/project-layout.md).
