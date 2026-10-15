# anchor-residual

Nhãn lịch sử `v9`; Pilot hai seed quality FAIL; cost PASS.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/pilot-v9-residual.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Probe và proof | [probes/](../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/notebooks) |
| Artifact gồm smoke/checkpoint/prediction | [artifacts/](../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/artifacts) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/scripts) |
| Logs | [v9-teacher-residual/](../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/logs) |

## Kết quả đã có

- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)
- [Forensics saved target hai seed15/10](results/forensics-20261015.md): kernel rộng, residual ít tín hiệu không gian; không nối nguyên trạng.

## Khôi phục artifact

- [kaggle.dvc](../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/kaggle.dvc)
- [smoke.dvc](../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/smoke.dvc)
- [input.dvc](../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/input.dvc)
- [resume_reference.dvc](../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).

[Module riêng](../../../../../../archive/methods/anchor-function-residual/builds/anchor-residual/source/mh/README.md): source canonical, byte/hash giữ nguyên; import cũ qua alias.
