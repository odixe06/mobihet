# static-synthetic-books

Nhãn lịch sử `new-fl`; Build so sánh nhiều phương pháp; trạng thái từng arm tại danh mục đề xuất.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/pilot-new-fl.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Probe và proof | [probes/](../../../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/notebooks) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/scripts) |
| Logs | [sa-static-books/](../../../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/logs) |

## Kết quả đã có

- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [runs.dvc](../../../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/runs.dvc)
- [kaggle.dvc](../../../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/kaggle.dvc)
- [smoke.dvc](../../../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/smoke.dvc)
- [input.dvc](../../../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/input.dvc)
- [resume_reference.dvc](../../../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).

[Module riêng](../../../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/source/mh/README.md): source canonical, byte/hash giữ nguyên; import cũ qua alias.
