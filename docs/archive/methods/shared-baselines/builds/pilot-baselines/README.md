# pilot-baselines

Nhãn lịch sử `pilot`; Build so sánh nhiều phương pháp; trạng thái từng arm tại danh mục đề xuất.

[Căn cứ/cơ chế](../../../../../methods/shared-baselines/design.md) · [Cấu hình và cổng](../../../../research/pilot-plan.md) · [Danh mục đề xuất](../../../../../proposals.md)

## Giai đoạn trong cùng build

| phần | đường dẫn |
|---|---|
| Scripts riêng | [scripts/](../../../../../../archive/methods/shared-baselines/builds/pilot-baselines/scripts) |
| Logs | [p1-pilot/](../../../../../../archive/methods/shared-baselines/builds/pilot-baselines/logs) |

## Kết quả đã có

- [report.md](results/report.md)

## Khôi phục artifact

- [runs.dvc](../../../../../../archive/methods/shared-baselines/builds/pilot-baselines/runs.dvc)
- [fixture.dvc](../../../../../../archive/methods/shared-baselines/builds/pilot-baselines/fixture.dvc)

DVC có thể chỉ có metadata ở local sau cleanup. Hash remote giữ nguyên; restore đúng target cần dùng. Source/manifest/notebook đã PUSHED bất biến; source snapshot dùng đường dẫn runtime gốc trong notebook. [Quy tắc cấu trúc và alias](../../../../../operations/project-layout.md).
