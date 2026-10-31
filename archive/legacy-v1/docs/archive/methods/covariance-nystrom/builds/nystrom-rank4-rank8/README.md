# nystrom-rank4-rank8

Nhãn lịch sử `v7`; Pilot hai seed quality FAIL; compression PASS.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/pilot-v7-nystrom.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Probe và proof | [probes/](../../../../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/notebooks) |
| Artifact gồm smoke/checkpoint/prediction | [artifacts/](../../../../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/artifacts) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/scripts) |
| Logs | [v7-rank4-rank8/](../../../../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/logs) |

## Kết quả đã có

- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [kaggle.dvc](../../../../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/kaggle.dvc)
- [smoke.dvc](../../../../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/smoke.dvc)
- [input.dvc](../../../../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/input.dvc)
- [resume_reference.dvc](../../../../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).

[Module riêng](../../../../../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/source/mh/README.md): source canonical, byte/hash giữ nguyên; import cũ qua alias.
