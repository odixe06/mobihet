# R20/V20 — lặp control, replay tiếp nhận, validation sạch (02/11/2026)

**Kết luận chính:** realKD (binary/margin/strength) có hại trực tiếp lên timeDelay/lớp vắng ngay cả khi teacher cố định;
lợi thế của margin trong live đến từ việc bảo vệ teacher (feedback), không từ tiếp nhận tốt hơn. Trên validation sạch,
lợi pooled của margin biến mất trong khi tác hại vẫn còn → loại realKD khỏi phương pháp mặc định.
Augmentation copula là nguồn lợi lớn nhất và tổng quát hóa thật; input-target nâng xe yếu trên validation sạch.

Phạm vi: DAGSNet W1, full 5343 xe, 20 round, cosine50, seed 42/43. Thiết kế khóa trước: [R20](screening-r20.md), [V20](screening-v20.md).
R đọc official own-test (đã đọc lặp lại nhiều batch, exploratory); V đọc foldval của nhánh fit foldtrain (val chưa fit).
Hai seed không phải CI/significance. Tái tạo: `python scripts/analyse_r20_v20.py` → `runs/r-study/analysis/{analysis.json,endpoint.csv}`.

## Trạng thái artifact

| nhóm | run | trạng thái |
|---|---|---|
| repeat R0–R3 | stream A/B × seed42/43 | 4/4 COMPLETE, audit PASS 20 round |
| replay R4–R11 | mixed_input noReal/real/margin/strength × 2 seed, teacher Q6/Q7 đóng băng | 7/8 PASS; R8 budget-stop r19 → nối phiên `takids1909/mh-r8-replay-margin-s42-resume1/1` từ full checkpoint 2 rank, audit PASS r20 |
| validation V2–V7 | augment/mixed_input/margin × 2 seed | 6/6 PASS |
| validation V0/V1 | CE+LA | gốc ERROR (guard zero-skip trước r1); V0R/V1R chạy lại fresh với guard common≤1%, PASS; skip 1–10/513k bước mỗi round |

R8 r20 đến từ phiên resume: exact-resume RNG chưa được chứng minh, đọc R8 như run có một điểm nối.

## 1. Độ ồn giữa các phiên (cùng công thức, cùng seed)

Ba phiên stream mỗi seed (Q0/R0/R2 và Q1/R1/R3), khoảng max−min tại r20:

| chỉ số | seed42 | seed43 |
|---|---:|---:|
| pooled F1 | .00050 | .00025 |
| p10 | .0059 | .0033 |
| mean xe<256 | .0030 | .0015 |
| timeDelay F1 | .0030 | .0034 |
| absent acc xe<256 | .018 | .0064 |

Giữa hai phiên cùng công thức, **~250 xe giảm >.05 và ~250 xe tăng >.05** mỗi seed. Hệ quả:
mọi đếm "số xe giảm" ở P/Q (vd 75 xe margin vs 29 xe control) nằm trong độ ồn phiên; chênh pooled <~.0005 hoặc p10 <~.006
không phân biệt được với lặp lại. Các chênh đã báo trước đây cần đọc lại theo ngưỡng này: sqrt input weight (−.0008/−.0006),
mixed support S1 (+.0001/−.0010), margin−strength pooled (+.0003/+.0005) đều ở biên hoặc trong độ ồn.

## 2. Replay control đạt tripwire

R4/R5 (replay noReal) − Q6/Q7 (live nguồn): pooled +.0003/−.0003, timeDelay −.0009/−.0044, p10 −.0003/−.0038,
low256 −.0027/−.0002 → PASS cả hai seed. Teacher đóng băng tái tạo được trajectory nguồn trong độ ồn.

## 3. Tiếp nhận realKD khi teacher cố định

Hiệu so R4/R5 (cùng teacher, không realKD):

| cell | pooled | other15 | timeDelay | absent acc | absent acc xe<256 | low256 mean |
|---|---|---|---|---|---|---|
| binary real (R6/R7) | +.0017/+.0020 | +.0051/+.0054 | −.049/−.050 | −.055/−.046 | −.055/−.038 | +.0072/+.0026 |
| margin (R8/R9) | +.0010/+.0014 | +.0049/+.0054 | −.059/−.058 | −.049/−.050 | −.051/−.035 | +.0052/+.0043 |
| strength (R10/R11) | +.0016/+.0021 | +.0051/+.0055 | −.050/−.048 | −.056/−.042 | −.046/−.036 | +.0065/+.0000 |

- **Margin không hơn binary/strength khi teacher cố định:** margin−binary pooled −.0008/−.0006, timeDelay −.010/−.008;
  margin−strength pooled −.0007/−.0007. Dấu hiệu bác bỏ đã khóa trong R20 xảy ra: không coi margin là luật tiếp nhận bảo vệ xe.
- Tác hại timeDelay/lớp vắng tồn tại **không cần feedback**: realKD trên hàng thật kéo dự đoán theo hàng teacher Q[y].
- Theo từng xe, vượt độ ồn phiên của chính xe đó (|Δ| > max(spread, .01)): binary 876/826 xe giảm, 1185/1211 xe tăng.

### Live so với replay: feedback khuếch đại tác hại

| cell | timeDelay live (Q−Q6/Q7) | timeDelay replay | pooled live | pooled replay |
|---|---|---|---|---|
| binary | −.092/−.093 | −.049/−.050 | +.0008/+.0005 | +.0017/+.0020 |
| margin | −.071/−.071 | −.059/−.058 | +.0016/+.0014 | +.0010/+.0014 |
| strength | −.082/−.083 | −.050/−.048 | +.0013/+.0009 | +.0016/+.0021 |

Cơ chế đo trực tiếp trên teacher live seed42 (Q[13,13] / Q[13,benign] tại r20): noReal .739/.250, stream .721/.267,
binary .581/.397, strength .601/.379, margin .656/.327. realKD làm học sinh nhầm timeDelay→benign nhiều hơn, uplink mang
nhầm lẫn đó vào Q, Q kéo học sinh tiếp — vòng phản hồi. Margin hạ trọng số realKD ở hàng có margin thấp nên giữ teacher
sạch hơn; đó là nguồn lợi live của nó. Hai kênh hại: (a) tiếp nhận trực tiếp ≈ −.05 timeDelay, (b) feedback thêm ≈ −.04 (binary).

Gốc rễ là thành phần target-class của KD theo lớp: Q[13] = (.74 timeDelay, .25 benign), KL(Q[y]‖p) kéo xác suất lớp thật
về .74 và đẩy .25 sang benign trên mọi hàng timeDelay thật — đúng cặp nhầm khó nhất của dataset.
Phần non-target (quan hệ giữa các lớp sai) là phần O20 đã chứng minh có utility. → giả thuyết W20 (decoupled KD).

## 4. Validation sạch (foldtrain → foldval)

| cell (V) | pooled | mean | p10 | low256 | timeDelay | absent acc | absent xe<256 |
|---|---|---|---|---|---|---|---|
| CE+LA (V0R/V1R) | .8749/.8753 | .7845/.7852 | .579/.574 | .736/.742 | .351/.350 | 0 | 0 |
| augment (V2/V3) | .8974/.8979 | .8433/.8438 | .693/.701 | .784/.781 | .351/.351 | .758/.757 | .373/.406 |
| mixed_input (V4/V5) | .8999/.9004 | .8493/.8498 | .708/.710 | .789/.788 | .340/.338 | .767/.771 | .365/.367 |
| margin (V6/V7) | .9005/.9008 | .8523/.8524 | .714/.711 | .791/.791 | .264/.262 | .696/.710 | .335/.321 |

- **Augmentation − CE:** pooled +.0226 cả hai seed, p10 +.114/+.127, lớp vắng 0 → .76. Nguồn lợi lớn nhất,
  tổng quát hóa trên val chưa fit — không phải hiệu ứng chọn trên test.
- **Mixed_input − augment:** pooled +.0025/+.0024, p10 +.014/+.009, low256 +.004/+.007 (vượt độ ồn);
  timeDelay −.011/−.014. V không có ô stream classQ nên chưa tách input-target khỏi classQ trên val.
- **Margin − mixed_input:** pooled +.0006/+.0005 (≈ độ ồn), timeDelay −.077/−.076, absent −.071/−.061,
  absent xe<256 −.030/−.046. Trên val sạch realKD không còn lợi pooled đáng kể, tác hại giữ nguyên.
- **Gap timeDelay có cả ở CE:** xe ≥4096 recall train .886 → val .267 (V0R). Generalization của timeDelay là vấn đề
  dữ liệu/đặc trưng (benign↔timeDelay gần như không để lại mâu thuẫn), không do KD gây ra; KD chỉ làm nặng thêm.

## 5. Chưa bão hòa

Mọi ô r15→20: pooled +.002–.004, mean xe<256 +.026–.050, p10 +.017–.033. Xe nhỏ chỉ có 2–4 bước/round
(1 epoch, batch 64) nên còn thiếu tối ưu. 20 round không đại diện plateau 50 round.

## 6. So benchmark cùng horizon (seed42 r20, [tham chiếu](../../../docs/research/benchmark-r20-reference.md))

| phương pháp | pooled | mean | p10 | timeDelay | byte/xe/20r |
|---|---|---|---|---|---:|
| APFed client (trao đổi trọng số) | .9030 | .8703 | .7610 | .254 | ~63,8 MB |
| S0 stream (ta) | .8936 | .8345 | .6925 | .324 | 3,33 MB |
| mixed_input (ta) | .8935 | .8375 | .6986 | .321 | 4,65 MB |

Thiếu hụt chủ yếu ở xe yếu (p10 −.07, mean −.035); ta đã hơn ở timeDelay. Ta dùng ~19× ít byte hơn APFed,
không gửi trọng số nên hỗ trợ kiến trúc khác nhau. Byte APFed là ước lượng dense fp32 395.408 tham số × 2 chiều × 20 round.

## 7. Byte bootstrap — đo offline

Bootstrap copula 1.655.872 B/chiều = histogram biên 256 bin fp32 (1.261.568, 76%) + hiệp phương sai normal-score (197.120)
+ moment Gaussian (197.184, **không dùng** trên đường copula). Đo trên toàn train+val (`scripts/measure_bootstrap_codec.py`,
`runs/bootstrap-codec/uplink-hist.json`): histogram mỗi xe mã hóa thưa (delta-index, count varint) **không mất mát**
trung bình 73.847 B (trung vị 54.531, p90 163.184, max 210.439; xe<256 trung bình 12.242) thay vì 1.261.568 B.
Tổng server giữ nguyên nên training không đổi. Uplink bootstrap ≈ 271 KB (−84%) không cần chạy lại.
Downlink nén lossy (CDF uint16 + fp16 Cholesky ≈ 729 KB) cần kiểm utility: ô `boot_compact` của W20.

## Hệ quả cho phương pháp

1. Giữ: copula bootstrap + streaming classQ trên synthetic (544 B/round) làm lõi; input-target bank là tầng tùy chọn khi
   băng thông cho phép (+33 KB/round, lợi xe yếu trên val).
2. Loại khỏi mặc định: realKD binary/margin/strength, sqrt input weight, proxy server/model downlink.
3. Kiểm tiếp (W20, [thiết kế](../../../docs/research/screening-w20.md)): decoupled KD (bỏ phần target-class), step floor synthetic cho xe ít bước,
   downlink bootstrap nén. Mọi so sánh dùng độ ồn phiên ở mục 1 làm ngưỡng đọc.
