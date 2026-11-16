# MobiHet/T3 — DAGSNet r50,3seed41/42/43

Audit CM/source/HP/support/10metric PASS; seed41 bổ sung full state/RNG hai rank PASS. Không prediction→CM replay;3seed không tự chứng minh significance.

| Metric | Seed41 | Seed42 | Seed43 | Mean ± sample SD |
|---|---:|---:|---:|---:|
|accuracy|0.912844675|0.912556903|0.913067297|0.912822959 ± 0.000255889|
|precision_macro|0.908332279|0.908030522|0.909126650|0.908496484 ± 0.000566212|
|precision_micro|0.912844675|0.912556903|0.913067297|0.912822959 ± 0.000255889|
|precision_weighted|0.908265991|0.907963402|0.908511483|0.908246958 ± 0.000274536|
|recall_macro|0.908772643|0.908693170|0.908936271|0.908800694 ± 0.000123954|
|recall_micro|0.912844675|0.912556903|0.913067297|0.912822959 ± 0.000255889|
|recall_weighted|0.912844675|0.912556903|0.913067297|0.912822959 ± 0.000255889|
|f1_macro|0.907866023|0.907678902|0.908324032|0.907956319 ± 0.000331909|
|f1_micro|0.912844675|0.912556903|0.913067297|0.912822959 ± 0.000255889|
|f1_weighted|0.910012175|0.909724615|0.910254489|0.909997093 ± 0.000265259|
|client_f1_macro_p10|0.778256751|0.777777778|0.778254007|0.778096179 ± 0.000275747|
|client_f1_macro_min|0.155555556|0.219410219|0.261796043|0.212253939 ± 0.053480553|

Nguồn: `runs/t3-s41/audit.json` (T341), `runs/b21-r50/audit.json` (T3P/T343). CBB là đối chứng, không dùng kết quả CBB thay T3.

T3: S64→64/M64→32/L48→16 từr26,floor16. Luật/code: [method](method.md), [B21](screening-dose-b21.md). Ranking hai kiến trúc và cost đang tổng hợp sau audit BERT; trạng thái [continue](../operations/continue.md).
