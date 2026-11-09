# Kết quả X20 — combo floor + nén, input-bank, non-target KD giới hạn trong lớp xe có

03/11/2026. 8/8 run COMPLETE, audit 20 round PASS (`runs/x-study/closure.json`), skip 0.
Thiết kế và gate khóa trước: [screening-x20](screening-x20.md). Độ ồn phiên: [R20/V20 §1](../../docs/research/results-r20-v20.md#1-độ-ồn-giữa-các-phiên-cùng-công-thức-cùng-seed);
spread dùng như W20: low256 .003, p10 .006, timeDelay .0035.
Số liệu: `runs/r-study/analysis/analysis.json` (`scripts/analyse_r20_v20.py`), cohort/lớp/teacher `runs/w-study/analysis.json`
(`scripts/analyse_w20.py`). Official own-test, đọc lặp nhiều lần, 2 seed, không CI.

## 1. Endpoint r20 (seed 42/43)

| ô | pooled | mean | p10 | low256 | ≥4096 | timeDelay | other15 | absent | absent xe<256 | rare |
|---|---|---|---|---|---|---|---|---|---|---|
| stream R0/R1 | .8931/.8944 | .8347/.8369 | .6956/.6951 | .7706/.7765 | .8787/.8802 | .3205/.3206 | .9313/.9326 | .834/.837 | .534/.533 | .811/.808 |
| mixed_input Q6/Q7 | .8935/.8945 | .8375/.8401 | .6986/.7048 | .7743/.7845 | .8791/.8805 | .3214/.3228 | .9316/.9326 | .830/.832 | .530/.536 | .811/.816 |
| step_floor W6/W7 | .8933/.8940 | .8412/.8408 | .7105/.7060 | .7946/.7944 | .8790/.8802 | .3208/.3224 | .9315/.9321 | .848/.853 | .656/.661 | .823/.821 |
| live_real_nckd W2/W3 | .8960/.8967 | .8396/.8415 | .7020/.6984 | .7740/.7834 | .8821/.8830 | .3120/.3133 | .9349/.9356 | .795/.801 | .500/.521 | .805/.809 |
| **core** X0/X1 | .8932/.8940 | .8413/.8413 | .7114/.7091 | .7973/.7943 | .8790/.8804 | .3216/.3212 | .9313/.9322 | .847/.851 | .657/.644 | .823/.822 |
| **bank** X2/X3 | .8936/.8943 | .8437/.8445 | **.7167/.7168** | **.8010/.8015** | .8793/.8807 | .3203/.3212 | .9318/.9326 | .847/.855 | .648/.661 | **.830/.826** |
| **live_real_nckd_local** X4/X5 | .8958/.8969 | .8386/.8417 | .6980/.7004 | .7730/.7831 | .8821/.8833 | .3104/.3143 | .9348/.9358 | **.834/.837** | .529/.530 | .801/.807 |
| **full** X6/X7 | **.8958/.8966** | **.8460/.8454** | .7149/.7174 | .8014/.7983 | .8820/.8830 | .3110/.3134 | .9348/.9355 | .855/.852 | .636/.659 | .823/.819 |

Byte/xe/20 round: core 1.022.199; bank, full 2.334.199; live_real_nckd_local 4.645.504 (bootstrap cũ).
Thời gian train trong dao động (13.050–13.550 s).
Tham chiếu r20 (seed42, [benchmark](../../../docs/research/benchmark-r20-reference.md)): FedAvg pooled .8946/mean .8411/p10 .7019/timeDelay .085;
APFed client .9118/.8703/.7610/.254, ≈63,8 MB.

## 2. Gate đã khóa

| ô | gate | kết quả (seed 42/43) | quyết định |
|---|---|---|---|
| core | low256/p10 trong spread W6/W7; pooled/timeDelay trong spread stream | so W6/W7: low256 +.0027/−.0001, p10 +.0008/+.0031, pooled −.0001/.0000, timeDelay +.0007/−.0012 | **PASS** — floor và nén cộng được |
| bank | p10 > core +.006 **hoặc** low256 > core +.003 ở cả hai seed | p10 +.0053/+.0077; low256 **+.0037**/+.0072 | **PASS** (seed42 nhờ low256, sát ngưỡng) |
| live_real_nckd_local | xác nhận nếu absent ≥ Q6/Q7 −.01 và pooled ≥ Q6/Q7 +.0015 | absent **+.0046/+.0051**; pooled +.0023/+.0025 | **XÁC NHẬN** giả thuyết W20 |
| full | pooled > bank +.001; absent ≥ bank −.01; low256/p10 không dưới bank − spread | pooled +.0022/+.0023 ✓; absent +.008/−.002 ✓; p10 −.0018/+.0006 ✓; low256 +.0004/**−.0032** | **KHÔNG QUA** — low256 seed43 vượt spread .003 |

full trượt sát ngưỡng (.0002). Không nới gate sau khi xem kết quả; thêm vào đó full có hai chi phí ngoài gate nhất quán ở cả hai
seed: timeDelay −.0094/−.0078 (> spread .0035) và rare −.0073/−.0072. → realKD không vào phương pháp mặc định (§3.3).

## 3. Cơ chế

### 3.1 core: floor + nén cộng thẳng

X0/X1 trùng W6/W7 ở mọi chỉ số và mọi cohort (Δ cohort ≤ .003, absent xe<256 +.001/−.016). Số xe đổi > spread lặp
1.052–1.068 mỗi chiều, đối xứng — đúng mức lệch dòng RNG đã thấy ở W20, không phải tác hại. Cặp byte thấp nhất giữ lợi floor.

### 3.2 bank: tăng tốc hội tụ cho xe nhỏ/vừa, lợi đang thu hẹp

| cohort train rows | xe | Δ F1/xe bank − core (s42/s43) |
|---|---:|---|
| 64–256 | 848 | +.0037/+.0072 |
| 256–960 | 1.378 | +.0044/+.0050 |
| 961–4.095 | 1.594 | +.0020/+.0024 |
| ≥4.096 | 1.523 | +.0003/+.0003 |

- Xe lên > spread lặp nhiều hơn xe xuống: 1.388 vs 1.204 / 1.387 vs 1.205 (chênh +184/+182, ở core−W6 chỉ ±14…60).
- rare +.007/+.004, mean +.0024/+.0032; teacher Q[13,13] .721 → .739, timeDelay precision +.012/+.008 (recall −.004/−.002).
- **Lợi thu hẹp theo round:** low256 bank − core r10 +.020/+.011 → r15 +.009/+.007 → r20 +.004/+.007; p10 +.020/+.013 → +.013/+.005
  → +.005/+.008. Bank chủ yếu tăng tốc; ở 50 round lợi có thể ≈0 trong khi tốn +1,31 MB/20 round (2,3× byte lõi).

### 3.3 nckd_local: xác nhận cơ chế hại lớp vắng; phần còn lại là trade-off timeDelay

- Giới hạn non-target vào lớp xe có **xóa hết hại lớp vắng**: absent so W2/W3 +.039/+.037, xe ≥4096 +.119/+.150; so Q6/Q7 +.005.
  Pooled, other15, ≥4096 giữ nguyên W2/W3 (Δ ≤ .0003). → W20 đúng: NCKD đè logit lớp vắng trên input local.
- Teacher Q[13,13] .742/.740 = W2/W3 (không feedback).
- Kênh còn lại là **timeDelay**: −.011/−.009 so Q6/Q7, td→benign +.020/+.017, recall −.013/−.011, precision +.007/+.009.
  Cơ chế: NCKD trên hàng timeDelay thật dạy phân bố lớp sai ≈ benign (Q[13,benign] .24) → đẩy logit benign lên trên input
  timeDelay. Không đổi được bằng giới hạn tập lớp vì benign có ở mọi xe.
- Lợi tập trung ở xe lớn: cohort ≥961 +.002–.003, 64–960 ≈0 (−.0014…+.0004); rare −.009…−.011.

### 3.4 full = bank + nckd_local

Trùng phép cộng: pooled +.0022/+.0023 so bank (= nckd_local so mixed_input), cohort ≥961 +.0023–.0029, timeDelay −.009/−.008,
xe 64–256 +.0004/−.0032. realKD không giúp xe yếu (mục tiêu chính), đổi +.002 pooled ở xe lớn lấy −.008 timeDelay và
−.007 rare. p10 .715–.717 đạt nhờ bank, không nhờ realKD.

## 4. Vị trí so baseline r20 (seed42, khác phiên, chỉ định vị)

bank X2: pooled .8936 ≈ FedAvg .8946; mean .8437 > .8411; p10 .7167 > .7019; timeDelay .320 > .085 (APFed client .254),
ở 2,33 MB thay vì ≈63,8 MB. Khoảng cách tới APFed client: pooled −.018, mean −.027, **p10 −.044** (W20 −.053, trước floor −.066).

## 5. Bài học

- Gate "hoặc" cho tầng tùy chọn phải kèm đọc quỹ đạo: bank qua gate r20 nhưng lợi giảm đều r10 → r20 (hiệu ứng tốc độ).
- Giới hạn KD vào lớp có ở xe là sửa đúng cho hại lớp vắng; nhưng cặp nhầm có lớp mặt ở mọi xe (benign/timeDelay)
  thì không giới hạn được — mọi KD trên dòng thật dạy sự nhầm của teacher.
- Gain pooled ở xe lớn không chuyển thành gain xe yếu; đọc cohort trước khi nhận một thành phần "tốt nhất pooled".
