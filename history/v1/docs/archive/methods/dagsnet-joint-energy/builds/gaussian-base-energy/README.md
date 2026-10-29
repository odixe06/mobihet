# gaussian-base-energy

Nhãn lịch sử `v10`; Cost/sampler FAIL; candidate r20 chưa đo.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/pilot-v10-energy.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Probe và proof | [probes/](../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/notebooks) |
| Artifact gồm smoke/checkpoint/prediction | [artifacts/](../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/artifacts) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/scripts) |
| Logs | [v10-gaussian-base-mh/](../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/logs) |

## Kết quả đã có

- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [kaggle.dvc](../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/kaggle.dvc)
- [smoke.dvc](../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/smoke.dvc)
- [input.dvc](../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/input.dvc)
- [resume_reference.dvc](../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).

[Module riêng](../../../../../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/source/mh/README.md): source canonical, byte/hash giữ nguyên; import cũ qua alias.
