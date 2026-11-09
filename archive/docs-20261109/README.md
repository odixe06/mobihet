# Archive 09/11/2026 — hồ sơ nghiên cứu đã đóng

Snapshot, **không phải trạng thái sống**. Live state ở [CONTEXT](../../CONTEXT.md) → [continue](../../docs/operations/continue.md).
Verdict/bài học còn dùng đã chép vào [LESSONS](../../LESSONS.md), [catalog generator](../../docs/research/generators.md) và
[results-y50](../../docs/research/results-y50.md). Chỉ mở khi cần tra lại số đo hoặc luật cũ cụ thể.

| file | nội dung | trạng thái |
|---|---|---|
| [screening-generator-utility50](research/screening-generator-utility50.md) | lịch utility50 standard/forest | hủy trước push |
| [screening-standard-t4](research/screening-standard-t4.md), [screening-forest-t4](research/screening-forest-t4.md), [results-generator-runtime](research/results-generator-runtime.md) | gate runtime T4 standard latent / forest+quantile | runtime PASS, generator đã đóng |
| [screening-generator-pilot20](research/screening-generator-pilot20.md), [results-generator-pilot20](research/results-generator-pilot20.md) | pilot20 standard/forest, đủ 10 metric | utility FAIL hai seed, dừng r20 |
| [copula-marginal-diagnosis](research/copula-marginal-diagnosis.md) | tách fitting error / temporal shift của standard latent | đóng cùng standard |
| [results-km-pilot](research/results-km-pilot.md), [screening-km-pilot50](research/screening-km-pilot50.md) | KM16 fp32 (w0 2D) full-scale r20 → r50 | r20 FAIL absent s42; r50 FAIL benign (−.0018/−.0016); thay bằng KM50 w0 .5 int8 |
| [screening-y50](research/screening-y50.md) | thiết kế Y50 (oracle/prior/noq/nofloor) | đóng; kết quả ở results-y50 |
| `*-launch.json` (c10, forest-t4, r20-repeat, standard-t4, w20, x20, y50-wave1/2) | receipt launch của study đã đóng | chỉ lịch sử |
