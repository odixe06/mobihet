# boundary-cell-pilot

Nhãn lịch sử `boundary-cells`; Calibration cost PASS; utility pilot đã push.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/README.md) · [Danh mục đề xuất](../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Source đã chạy | [source/](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/source) |
| Probe và proof | [probes/](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/notebooks) |
| Artifact gồm smoke/checkpoint/prediction | [artifacts/](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/artifacts) |
| Scripts riêng | [scripts/](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/scripts) |
| Logs | [cell-counts-v1/](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/logs) |

## Kết quả đã có

- [calibration-report.md](results/calibration-report.md)
- [contract.md](results/contract.md)
- [rebuild.md](results/rebuild.md)

## Khôi phục artifact

- [kaggle.dvc](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/kaggle.dvc)
- [smoke.dvc](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/smoke.dvc)
- [input.dvc](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/input.dvc)
- [controller-fixture.dvc](../../../../../methods/boundary-cell-counts/builds/boundary-cell-pilot/controller-fixture.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../operations/project-layout.md).
