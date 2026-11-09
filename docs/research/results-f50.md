# F50 — phương pháp đề xuất 50 round

06/11/2026. F0 bank s42, F1 bank s43, F2 core s42, F3 core s43: đủ 50 round, audit PASS 4/4 (`runs/f-study/closure.json`, `runs/f-study-s43/closure.json`).
Thiết kế và luật đọc khóa trước: [screening-f50](../../archive/docs-20261107/research/screening-f50.md). Số tính bằng `scripts/analyse_f50.py` từ CM own-test, ghi `runs/f-study/analysis/analysis.json`.
F3 chạy một session (~10,2 h), không cần `-c2`. Rà toàn pipeline sau F50: [pipeline-review](../../archive/docs-20261107/research/pipeline-review.md).

## Kết luận

1. **Bank không qua luật giữ tầng, đã chốt.** Luật cần bank − core vượt spread (low256 .003 hoặc p10 .006) ở cả hai seed. Tại r50: seed 42 +.0013 và +.0002, seed 43 +.0011 và +.0001. Phương pháp = **core** (1.054.839 B/xe/50 round; bản lossless [L50](results-l50.md) 691.466 B).
2. **Lợi của bank thu hẹp theo round ở cả hai seed.** Seed 42 ở r20 bank hơn core +.0097 p10, +.0101 low256; tới r50 gần như hết, trong khi macro F1 kém −.0016 và tốn gấp ~4,1 lần byte (4.334.839 B).
3. **So với APFed r50 (baseline seed 42):** core trung bình hai seed có client mean .8640 so với .8663, p10 .7564 so với .7533, nhưng thấp hơn ở pooled: accuracy .9096 so với .9278, f1_macro .8995 so với .9119. Byte ~151 lần ít hơn (1,05 MB so với ≈159,5 MB). Baseline một seed ở repo khác. SCAFFOLD (.7738) và CANPerFL (.7684) có p10 cao hơn core.
4. **timeDelay vẫn là lớp yếu nhất** (F1 ≈ .34 ở cả bốn run); 50 round không giải quyết.

## 10 metric pooled r50

| metric | F0 bank s42 | F1 bank s43 | F2 core s42 | F3 core s43 | APFed client s42 |
|---|---:|---:|---:|---:|---:|
| accuracy | .908743 | .909278 | .909355 | .909745 | .9278 |
| precision_macro | .896672 | .898260 | .898246 | .899947 | .9377 |
| precision_micro | .908743 | .909278 | .909355 | .909745 | .9278 |
| precision_weighted | .902198 | .902703 | .903132 | .903386 | .9217 |
| recall_macro | .900779 | .901095 | .902222 | .902059 | .9032 |
| recall_micro | .908743 | .909278 | .909355 | .909745 | .9278 |
| recall_weighted | .908743 | .909278 | .909355 | .909745 | .9278 |
| f1_macro | .897511 | .898478 | .899160 | .899845 | .9119 |
| f1_micro | .908743 | .909278 | .909355 | .909745 | .9278 |
| f1_weighted | .904538 | .905056 | .905430 | .905707 | .9169 |

## Theo xe r50

| chỉ số | F0 | F1 | F2 | F3 | APFed |
|---|---:|---:|---:|---:|---:|
| client mean | .863534 | .864928 | .863357 | .864697 | .8663 |
| p10 | .756430 | .756649 | .756214 | .756547 | .7533 |
| min | .162393 | .311966 | .153846 | .214379 | .1500 |
| low256 | .843511 | .846934 | .842172 | .845799 | — |
| absent (xe nhỏ) | .721543 | .713183 | .709325 | .710611 | — |
| rare1_16 (xe nhỏ) | .842471 | .842881 | .839812 | .839914 | — |
| timeDelay F1 | .339013 | .338937 | .342007 | .340645 | — |

## Quỹ đạo bank − core, seed 42

| round | f1_macro | p10 | low256 | client mean | timeDelay |
|---:|---:|---:|---:|---:|---:|
| 20 | +.00023 | +.00974 | +.01006 | +.00288 | −.00134 |
| 30 | −.00027 | +.00168 | +.00316 | +.00172 | −.00257 |
| 40 | −.00126 | +.00264 | +.00191 | +.00041 | −.00329 |
| 50 | −.00165 | +.00022 | +.00134 | +.00018 | −.00299 |

## Quỹ đạo bank − core, seed 43

| round | f1_macro | p10 | low256 | client mean | timeDelay |
|---:|---:|---:|---:|---:|---:|
| 20 | −.00013 | +.00566 | −.00126 | +.00155 | −.00235 |
| 30 | −.00054 | −.00064 | +.00285 | +.00119 | −.00349 |
| 40 | −.00106 | −.00106 | +.00075 | +.00001 | −.00251 |
| 50 | −.00137 | +.00010 | +.00114 | +.00023 | −.00171 |

## Giới hạn

Official test đã đọc nhiều batch, không phải kiểm định độc lập. Hai seed không phải significance. Audit F3 là CM/source/config/teacher, chưa prediction replay. Chưa có DP/secure aggregation thực thi.
Model `.pt` F0–F2 đã rời local, khôi phục qua DVC ([backup](../operations/backup.md)); F3 còn local, chưa DVC.
