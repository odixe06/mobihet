# anchor-synthetic-books

Nhãn lịch sử `v5`; Build so sánh nhiều phương pháp; trạng thái từng arm tại danh mục đề xuất.

[Căn cứ/cơ chế](../../design.md) · [Cấu hình và cổng](../../../../research/pilot-v5-books.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Probe và proof | [probes/](../../../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/probes) |
| Calibration/pilot/continuation notebooks | [notebooks/](../../../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/notebooks) |
| Scripts riêng | [scripts/](../../../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/scripts) |
| Logs | [v5-anchor-gradient-books/](../../../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/logs) |

## Kết quả đã có

- [figures.md](results/figures.md)
- [paper.md](results/paper.md)
- [rebuild.md](results/rebuild.md)
- [report.md](results/report.md)

## Khôi phục artifact

- [runs.dvc](../../../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/runs.dvc)
- [kaggle.dvc](../../../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/kaggle.dvc)
- [smoke.dvc](../../../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/smoke.dvc)
- [input.dvc](../../../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/input.dvc)
- [resume_reference.dvc](../../../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/resume_reference.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).

[Module riêng](../../../../../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/source/mh/README.md): source canonical, byte/hash giữ nguyên; import cũ qua alias.
