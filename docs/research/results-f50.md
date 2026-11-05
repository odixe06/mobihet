# F50 — phương pháp đề xuất 50 round

05/11/2026. F0 bank s42, F1 bank s43, F2 core s42: đủ 50 round, audit PASS 3/3 (`runs/f-study/closure.json`).
Thiết kế và luật đọc khóa trước: [screening-f50](screening-f50.md). Số tính bằng `scripts/analyse_f50.py` từ CM own-test, ghi `runs/f-study/analysis/analysis.json`.
F3 core s43 đang chạy; xem [bàn giao](../operations/continue.md).

## Kết luận

1. **Bank không qua luật giữ tầng.** Luật cần bank − core vượt spread (low256 .003 hoặc p10 .006) ở cả hai seed. Seed 42 tại r50 chỉ +.0013 và +.0002, nên F3 không thể đổi kết quả này. Theo luật, phương pháp = **core** (1.054.839 B/xe/50 round; bản lossless [L50](results-l50.md) 691.466 B). Xác nhận chính thức sau F3 theo đúng trình tự đã khóa.
2. **Lợi của bank thu hẹp theo round.** Ở r20 bank hơn core +.0097 p10, +.0101 low256; tới r50 gần như hết, trong khi macro F1 kém −.0016 và tốn gấp ~4,1 lần byte (4.334.839 B).
3. **So với APFed r50 (seed 42):** core ngang client mean (.8634 so với .8663 kém −.0029) và nhỉnh p10 (.7562 so với .7533), nhưng thấp hơn ở pooled: accuracy .9094 so với .9278, f1_macro .8992 so với .9119. Byte ~151 lần ít hơn (1,05 MB so với ≈159,5 MB). Chưa phải so sánh đầy đủ: core mới một seed, baseline ở repo khác.
4. **timeDelay vẫn là lớp yếu nhất** (F1 ≈ .34 ở cả ba run); 50 round không giải quyết.

## 10 metric pooled r50

| metric | F0 bank s42 | F1 bank s43 | F2 core s42 | APFed client s42 |
|---|---:|---:|---:|---:|
| accuracy | .908743 | .909278 | .909355 | .9278 |
| precision_macro | .896672 | .898260 | .898246 | .9377 |
| precision_micro | .908743 | .909278 | .909355 | .9278 |
| precision_weighted | .902198 | .902703 | .903132 | .9217 |
| recall_macro | .900779 | .901095 | .902222 | .9032 |
| recall_micro | .908743 | .909278 | .909355 | .9278 |
| recall_weighted | .908743 | .909278 | .909355 | .9278 |
| f1_macro | .897511 | .898478 | .899160 | .9119 |
| f1_micro | .908743 | .909278 | .909355 | .9278 |
| f1_weighted | .904538 | .905056 | .905430 | .9169 |

## Theo xe r50

| chỉ số | F0 | F1 | F2 | APFed |
|---|---:|---:|---:|---:|
| client mean | .863534 | .864928 | .863357 | .8663 |
| p10 | .756430 | .756649 | .756214 | .7533 |
| min | .162393 | .311966 | .153846 | .1500 |
| low256 | .843511 | .846934 | .842172 | — |
| absent (xe nhỏ) | .721543 | .713183 | .709325 | — |
| rare1_16 (xe nhỏ) | .842471 | .842881 | .839812 | — |
| timeDelay F1 | .339013 | .338937 | .342007 | — |

## Quỹ đạo bank − core, seed 42

| round | f1_macro | p10 | low256 | client mean | timeDelay |
|---:|---:|---:|---:|---:|---:|
| 20 | +.00023 | +.00974 | +.01006 | +.00288 | −.00134 |
| 30 | −.00027 | +.00168 | +.00316 | +.00172 | −.00257 |
| 40 | −.00126 | +.00264 | +.00191 | +.00041 | −.00329 |
| 50 | −.00165 | +.00022 | +.00134 | +.00018 | −.00299 |

## Giới hạn

Official test đã đọc nhiều batch, không phải kiểm định độc lập. Bank có hai seed, core một seed. Chưa có DP/secure aggregation thực thi.
Model `.pt` đã rời local, khôi phục qua DVC ([backup](../operations/backup.md)).
