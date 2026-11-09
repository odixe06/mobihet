# Kết quả W20 — decoupled KD, step floor synthetic, bootstrap nén

02/11/2026. 10/10 run COMPLETE, audit 20 round PASS (source/CM/aggregate/teacher chain/byte/npy cache verify), skip ≤ 1,7e-6.
Thiết kế và dấu hiệu bác bỏ: [screening-w20](screening-w20.md). Độ ồn phiên: [R20/V20 §1](../../docs/research/results-r20-v20.md#1-độ-ồn-giữa-các-phiên-cùng-công-thức-cùng-seed).
Số liệu: `runs/r-study/analysis/{analysis.json,endpoint.csv}` (`scripts/analyse_r20_v20.py`), cohort/lớp/teacher
`runs/w-study/analysis.json` (`scripts/analyse_w20.py`). Official own-test, đọc lặp nhiều lần, 2 seed, không CI.

## 1. Endpoint r20 (seed 42/43)

| ô | pooled | mean | p10 | low256 | timeDelay | other15 | absent | absent xe<256 |
|---|---|---|---|---|---|---|---|---|
| stream R0/R1 | .8931/.8944 | .8347/.8369 | .6956/.6951 | .7706/.7765 | .3205/.3206 | .9313/.9326 | .834/.837 | .534/.533 |
| stream R2/R3 | .8933/.8943 | .8349/.8371 | .6984/.6984 | .7676/.7754 | .3208/.3184 | .9314/.9327 | .833/.838 | .516/.534 |
| mixed_input Q6/Q7 | .8935/.8945 | .8375/.8401 | .6986/.7048 | .7743/.7845 | .3214/.3228 | .9316/.9326 | .830/.832 | .530/.536 |
| realKD live Q10/Q11 | .8943/.8950 | .8386/.8416 | .6980/.7043 | .7729/.7862 | .2293/.2296 | .9386/.9394 | .771/.784 | .474/.503 |
| realKD replay R6/R7 | .8955/.8962 | .8398/.8424 | .6972/.7055 | .7788/.7868 | .2716/.2687 | .9371/.9380 | .775/.786 | .470/.502 |
| **replay_real_nckd** W0/W1 | .8959/.8968 | .8393/.8414 | .6996/.6986 | .7750/.7839 | .3100/.3128 | .9349/.9358 | .798/.804 | .511/.522 |
| **live_real_nckd** W2/W3 | .8960/.8967 | .8396/.8415 | .7020/.6984 | .7740/.7834 | .3120/.3133 | .9349/.9356 | .795/.801 | .500/.521 |
| **synth_nckd** W4/W5 | .8933/.8947 | .8335/.8362 | .6912/.6950 | .7680/.7757 | .3436/.3449 | .9299/.9313 | .834/.841 | .527/.538 |
| **step_floor** W6/W7 | .8933/.8940 | .8412/.8408 | **.7105/.7060** | **.7946/.7944** | .3208/.3224 | .9315/.9321 | .848/.853 | **.656/.661** |
| **boot_compact** W8/W9 | .8932/.8944 | .8344/.8373 | .6951/.6978 | .7672/.7776 | .3209/.3211 | .9314/.9326 | .834/.833 | .542/.536 |

Byte/xe/20 round: stream, synth_nckd, step_floor 3.333.504; boot_compact **1.022.199**; real_nckd 4.645.504.
Thời gian train: step_floor +2–4% so stream; các ô khác trong dao động.

## 2. Gate đã khóa

| ô | gate | kết quả | quyết định |
|---|---|---|---|
| replay_real_nckd | timeDelay không giảm >.02 so R4/R5; giữ phần lớn lợi other15 của R6/R7 | timeDelay −.010/−.006; other15 +.0029/+.0032 (R6/R7 +.0051/+.0054) | **PASS**, nhưng lớp vắng −.032/−.028 |
| live_real_nckd | Q[13,13] r20 không thấp như Q10 (.58); timeDelay không giảm >.02 | Q[13,13] **.742/.740** (noReal .739); timeDelay −.009/−.010 | **PASS**, lớp vắng −.035/−.032 |
| synth_nckd | timeDelay tăng > spread .0035, pooled không giảm >.0005 | timeDelay +.020–.027; pooled −.0003…+.0004 | PASS hình thức, **không nhận** (§3.2) |
| step_floor | low256/p10 tăng > spread .003/.006 | low256 +.018–.027, p10 +.008–.018 mọi control | **PASS** |
| boot_compact | mọi chỉ số trong spread | pooled/mean/p10/low256/timeDelay trong spread; absent seed43 −.005 (seed42 ±.001) | **chấp nhận** |

## 3. Cơ chế

### 3.1 Non-target KD trên dòng thật

- Bỏ phần target-class của KD **cắt vòng feedback**: teacher live giữ Q[13,13] .742, Q[13,benign] .242, ngang noReal.
  Live = replay (pooled +.0025/+.0022 so Q6/Q7 vs +.0021/+.0026 so R4/R5): không còn khuếch đại.
- Pooled cao nhất từ trước tới nay ở 20 round (.8960/.8967), other15 +.003, mean +.002, gain tập trung ở xe ≥961 dòng
  (cohort mean +.002–.003; xe 64–256 ≈0).
- timeDelay còn −.006…−.010 (> spread .0035): phần non-target vẫn dạy "timeDelay giống benign" trên hàng thật.
- **Hại lớp vắng còn lại** (−.028…−.035, xe ≥4096 −.065…−.119). Dòng test của lớp vắng bị đoán sang lớp xe *đang có*
  tăng .153 → .192 (Q6 → W2), sang lớp vắng khác giảm .017 → .013 → cả nhóm lớp vắng bị đè.
  Giải thích: NCKD khớp phân bố giữa các lớp sai theo Q[y]; lớp vắng của xe thường có khối lượng teacher rất nhỏ trên hàng
  của các lớp xe có, nên logit của chúng bị ép xuống trên mọi input local. → kiểm ở X20 (`nckd_local`).

### 3.2 synth_nckd chỉ dời ngưỡng timeDelay/benign

timeDelay recall .239 → .303 nhưng precision .487 → .400; benign bị đoán timeDelay .051 → .094 (gần gấp đôi báo động giả),
benign F1 −.018, other15 −.0013…−.0017; Q[13,13] **giảm** .721 → .665. Pooled đứng yên vì +.02 timeDelay bù −.0015 × 15 lớp.
Không có phân tách tốt hơn, chỉ đổi điểm làm việc → không đưa vào phương pháp. Củng cố kết luận: timeDelay bị giới hạn bởi đặc trưng.

### 3.3 Step floor

| cohort train rows | xe | Δ F1/xe (so R0, R2, Q0 / R1, R3, Q1) | absent |
|---|---:|---|---|
| 64–256 | 848 | +.024–.027 / +.018–.019 | +.12–.14 |
| 256–960 | 1.378 | +.007–.009 / +.004–.006 | +.02 |
| 961–4.095 (không chạm floor) | 1.594 | +.001 / −.001 | ≈0 |
| ≥4.096 (không chạm) | 1.523 | ≈0 | −.01…−.03 |

- Lợi đúng nơi có bước thêm; xe không chạm floor trung bình ≈0. Số xe vượt spread lặp tăng ở *mọi* cohort, cả lên lẫn xuống,
  vì floor đổi độ rộng permutation → lệch dòng RNG: sd Δ của xe không chạm (.035/.014) bằng sd khi đổi seed (.035/.014),
  còn boot_compact giữ dòng RNG chỉ .025/.011. Không phải tác hại; per-car "vượt repeat" vô hiệu khi ô đổi lượng RNG tiêu thụ.
- p10 .708 trung bình so APFed .761: khoảng cách giảm từ ~.066 xuống ~.053; low256 r15→20 vẫn +.025–.030 (chưa bão hòa).

### 3.4 Bootstrap nén

Downlink CDF uint16 + fp16 mean/Cholesky, uplink thưa không mất mát: tổng 3,31 MB → 1,02 MB/xe/20 round, utility trong spread.
Lớp vắng seed43 −.005 so ba phiên (seed42 không) — theo dõi ở combo, chưa đủ để bác bỏ.

## 4. Bài học

- Một ô "PASS gate" có thể chỉ dời ngưỡng giữa hai lớp nhầm; với cặp nhầm khó phải đọc precision/recall và nhầm chéo.
- Đếm xe vượt độ ồn lặp chỉ hợp lệ khi ô giữ dòng RNG; ô đổi số bước/lượng mẫu phải so với dao động đổi seed.
- Bỏ target-class KD giải quyết feedback và timeDelay phần lớn, nhưng KD trên dòng thật còn một kênh hại riêng cho lớp vắng.
