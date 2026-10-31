# geometry-g59-g61

Nhãn lịch sử `v6`; Pilot FAIL: G59 lỗi số, G61 quality/byte FAIL.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/pilot-v6-geometry.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Probe và proof | [probes/](../../../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/notebooks) |
| Artifact gồm smoke/checkpoint/prediction | [artifacts/](../../../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/artifacts) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/scripts) |
| Logs | [v6-g59-g61/](../../../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/logs) |

## Kết quả đã có

- [figures.md](results/figures.md)
- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [kaggle.dvc](../../../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/kaggle.dvc)
- [smoke.dvc](../../../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/smoke.dvc)
- [input.dvc](../../../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/input.dvc)
- [resume_reference.dvc](../../../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).

[Module riêng](../../../../../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/source/mh/README.md): source canonical, byte/hash giữ nguyên; import cũ qua alias.
