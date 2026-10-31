# fragmented-snapshots

Nhãn lịch sử `p2`; Đã kiểm; hạ tầng, còn giới hạn.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/pilot-plan.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Probe và proof | [probes/](../../../../../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/notebooks) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/scripts) |
| Logs | [p2-deadline/](../../../../../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/logs) |

## Kết quả đã có

- [report.md](results/report.md)

## Khôi phục artifact

- [runs.dvc](../../../../../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/runs.dvc)
- [kaggle.dvc](../../../../../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/kaggle.dvc)
- [source_audit.dvc](../../../../../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/source_audit.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).

[Module riêng](../../../../../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/source/mh/README.md): source canonical, byte/hash giữ nguyên; import cũ qua alias.
