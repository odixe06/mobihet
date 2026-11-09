# KM-lite codec + thống kê toàn mạng (sim b9) — luật khóa 09/11/2026 (trước push)

Bối cảnh: KM16 w0 .5 int8 full-scale r50 Δmacro +.0085/+.0086 so Core (FAIL guard benign −.0014, luật [screening-km50](screening-km50.md));
pooled macro .9077/.9085 sau APFed .9119 và ≈ CAN-PerFL .9088, nhưng **tổng byte 50 round ≈3,67 MB/xe** (Core bootstrap 1,00 MB +
KM lên 0,88 MB + KM xuống 1,74 MB + Q 54 KB) so FedProto/FedTGP 1,64 MB, TinyProto 0,17 MB. Ràng buộc owner giữ: compute ≤ Core, không trao model.

## Bằng chứng trước thiết kế (không train, không val/test chính thức)

- `scripts/probe_km_codec{,2}.py` → `runs/km-codec-probe/probe{,2}.json`: LL held-out fold val trong normal-score, 200 xe sim.
  Thước đo chỉ so trong cùng họ nhiễu; tham chiếu: up int8 (b8, utility trung tính +.0067 vs +.0070) mất 3,15 nat.
  down int8 −0,09, down int6 −0,68 (≪ 3,15) → chấp nhận được; down int4 −28,8, up int6/int4 −17/−55 → loại;
  class moments int8 ≈ 0; PPCA-R tự cơ sở: phổ phẳng (99% var cần ~50/77 chiều), R48 −15 nat → loại.
  Lloyd2 −1,1 nat (không dùng). Lấy mẫu 10% xe trong sim 200 xe −17,6 nat — nhưng sim chỉ 1,5 M dòng; full-scale 10% ≈ 3,7 M dòng.
- Độ thưa: xe có trung bình 128/256 ô (lớp, cụm) → sparse uplink giảm ≈ 50% chính xác (lossless); zlib-6 trên mã int8 lên ×0,43–0,47, xuống ×0,66.
- Phân rã own-test r50 (đọc lại CM đã audit, không chọn HP): lợi KM nằm ở precision lớp1 (+.09); gap còn lại so APFed là
  precision lớp 7/6/1 (FP từ benign) và benign recall (dự đoán timeDelay 223k vs 50k). Trần oracle sim còn dư chủ yếu lớp 7
  (oracle +.038 F1, KM16 +.004–.007, K24/K32 +.010) → hướng tăng K có cơ sở nhưng sim 200 xe có ít dòng/cụm hơn full-scale ≈25×.
- `scripts/check_km_lite.py`: knobs tắt == source frozen KM50/b8 (`2e92f1d`) bit-for-bit (km_p/km_mu/km_L/bytes); lite8/lite6/part.1 hữu hạn,
  đường chéo Cholesky > 0. Smoke local 200 xe lite6+part.1 r1: compile gate PASS, skip 0.

## Cơ chế và thay đổi (`methods/mobihet_sim.py`, mặc định tắt = KM50)

- `cop_kmeans_lite=8|6`: class moments int8/xe (khởi tạo + đích co), tâm broadcast int8, Cholesky xuống đường chéo fp16 + phần dưới `bits`
  một scale fp16/hàng + zlib, mean fp16; bỏ downlink mean/Cholesky lớp (client chỉ sinh từ mixture); CDF lossless zlib (L50);
  uplink sparse + zlib. Byte đo trong run (`KM_LITE_BYTES`). Compute tại xe = KM50 (không thêm bước/dòng).
- `cop_kmeans_part=p`: chỉ phần p xe (hash cố định theo id) gửi moment bậc hai cuối; mọi xe vẫn gửi count/mean.
- `cop_pop=-1` (chỉ simulator): thống kê generator từ fold-train **toàn 5.343 xe**; train/LA/Q/đánh giá vẫn 200 xe sim, fold val.
  Lý do: lượng dòng sau mỗi thống kê như full-scale, để hỏi đúng câu K32/lấy mẫu xe. Không đọc fold val/official test cho thống kê.

| arm | thay đổi (trên KM16 w0 .5 int8) | so với |
|---|---|---|
| LT8 | lite8, seed42 | KQ8 b8 seed42 |
| LT6 | lite6, seed42 | KQ8 b8 seed42 |
| LT6S43 | lite6, seed43 | KQ8S43 b8 |
| PK16 | lite6 + pop toàn mạng, K16 | (đối chứng của PK32/PP10; so LT6 chỉ báo) |
| PK32 | lite6 + pop, K32 | PK16 |
| PP10 | lite6 + pop, part 0,1 | PK16 |

## Luật (fold val, r20, cosine20, như screening-sim-foldval; không chọn round)

Δ = arm − đối chứng ghi trên. Guard chung: Δp10 ≥ −.006, ΔtimeDelay ≥ −.0035, Δbenign ≥ −.0005, Δabsent ≥ −.01 khi ≥2.000 dòng.
low256 chỉ báo (nhiễu S1−S0 −.045 ở ~20 xe, b2).

1. **Codec (không thua):** LT* PASS khi Δmacro ≥ −.0015 + guard. Chọn **lite6** nếu LT6 và LT6S43 cùng PASS; không thì lite8 nếu LT8 PASS;
   không thì không đổi codec.
2. **K32 (tăng):** PK32 PASS khi Δmacro ≥ +.002 + guard.
3. **Lấy mẫu 10% xe (không thua):** PP10 PASS khi Δmacro ≥ −.0015 + guard.

Một seed cho PK/PP, panel fold val đã dùng nhiều lần → không là xác nhận độc lập. Không tune bits/K/p sau khi xem số.

## Bước sau (chỉ khi audit PASS; full-scale cần owner duyệt quota)

Full-scale pilot r20 cosine50 seed42, 5.343 xe, official own-test, so KM50 Q42 cùng r20 (cùng seed/lịch):
codec chọn (+part 0,1 nếu PP10 PASS) phải Δmacro ≥ −.0015 + guard pilot20; K32 (nếu PK32 PASS) phải Δmacro ≥ +.002 + guard.
Byte báo đủ hai chiều + bootstrap + Q theo `KM_LITE_BYTES`.

## Pilot full-scale lite6 — luật khóa 09/11/2026 (trước push, owner chọn "pilot r20 seed42, luật cũ")

Run `scripts/run_km_lite_pilot20.py` → slug `mh-k-km16lite6-s42-p20`, 5.343 xe, official own-test, cosine50 dừng r20 (`--rounds 50 --until 20`),
save mỗi 5 round (state r20 hai rank để nối r50), cap 7,5 h, không W&B. Hp = **đúng hp KM50 Q42 + `cop_kmeans_lite=6`** (assert trong launcher);
overlay `methods/mobihet_sim.py` sha `683aa2b3…` (= b10; `check_km_lite.py` PASS 09/11: knob tắt khớp frozen KM50 bit-for-bit).
Đối chứng: KM50 Q42 `catbaochau/mh-k-km16q8-s42-r50/1` tại r20 (`cms_r020`, cùng seed/lịch).

**PASS (non-inferiority codec)** khi tại r20: Δf1_macro ≥ −.0015 và guard pilot20 Δp10 ≥ −.006, Δlow256 ≥ −.003, ΔtimeDelay ≥ −.0035,
Δbenign ≥ −.0005, Δabsent ≥ −.01. Báo thêm (không gate): Δabsent trung bình r16–r20 (guard 1.555 dòng nhiễu ±.05 giữa round liền),
byte `KM_LITE_BYTES` hai chiều + bootstrap + Q, thời gian bootstrap. Một seed, test đã đọc lặp → không là significance.
PASS → tự nối r50 từ state r20 + push seed43 r50 (owner 09/11, thang ở CONTEXT); luật r50 = luật codec trên tại r50 vs KM50 Q42/Q43 r50, cả hai seed. FAIL do riêng absent r20 trong khi mean r16–r20 qua → báo owner, không đổi luật.

### Kết quả pilot r20 (09/11 13:27, `minhtran0601/mh-k-km16lite6-s42-p20/1`, audit PASS, skip 0) — **FAIL → nhánh lite6 dừng**

`runs/km-lite-pilot20/{audit,verdict-r20}.json`. Δ so KM50 Q42 r20: macro −.0001 ✓, p10 −.0023 ✓, **low256 −.0063 ✗** (guard −.003),
timeDelay −.0015 ✓, benign −.0001 ✓, **absent −.0186 ✗** (guard −.01; mean r16–r20 −.0082). Hai guard trượt → không phải "chỉ absent", không ngoại lệ.
So Core F2 r20: macro +.0073, p10 +.0151, low256 +.0087, absent −.0302. Byte bootstrap: up 259 KB, down 871 KB/xe.
Đọc: macro lệch ≤ .0004 ở mọi round r10–r20 (ghép cặp cùng seed). low256/absent dao động quanh 0 (r16–r20 trung bình −.0029/−.0082);
chênh **giữa seed** KM50 (Q43−Q42) cùng round đã ±.005–.01 low256, ±.02–.037 absent → FAIL r20 nằm trong vùng nhiễu, **chưa chứng minh lite6 hại**.
Verdict giữ (luật khóa); không chấm lại lite6 bằng luật mới — muốn kiểm lại cần run mới với luật khóa trước.
Hệ quả: generator đóng băng = **KM16 w0 .5 int8, codec gốc (≈3,67 MB/xe/50r)**; lite6 không vào nền. Không thử lite8/biến thể codec khi chưa có owner duyệt.

| r20 | accuracy | P macro | P micro | P weighted | R macro | R micro | R weighted | F1 macro | F1 micro | F1 weighted |
|---|---|---|---|---|---|---|---|---|---|---|
| lite6 s42 | .9153 | .9054 | .9153 | .9071 | .9011 | .9153 | .9153 | .9006 | .9153 | .9087 |

## Đợt b10 — cải tiến theo module (luật khóa 09/11, trước push)

Owner 09/11 duyệt: probe → sim b10 song song b9, pilot full-scale sau verdict. Nền mọi arm = KQ8 (KM16 w0 .5 int8, thống kê sim),
đối chứng b8 KQ8 seed42 / KQ8S43. Compute tại xe = Core (EM: một lượt bootstrap thêm, 16 Gaussian/dòng ≪ 1 epoch DAGSNet).

Probe `scripts/probe_km_em.py` → `runs/km-codec-probe/probe_em.json` (LL fold val, mean16 so KM16): **EM1 +6,7 nat, EM2 +9,6**
(lớp7 +6,5/+9,2; K8→K16 chỉ +4), whitened k-means +2,3, soft k-means ≈0 (bỏ). Preflight `check_km_lite.py` PASS mọi knob;
smoke local 200 xe 3 round gộp mọi knob: compile PASS, skip 0, hữu hạn (đã sửa NaN từ dòng synthetic bị mask trong Q1).

| arm | module | knob | cơ chế | byte thêm |
|---|---|---|---|---|
| E1 / E1S43 | generator | `cop_kmeans_em=1` | gán cứng cắt covariance theo ô Voronoi → thiếu vùng ranh giới mode | 1 mixture xuống + 1 moment lên |
| E2 | generator | `cop_kmeans_em=2` | như E1 | ×2 |
| WH | generator | `cop_kmeans_white=true` | Euclid bị 24 feature p4_std tương quan chi phối | 0 (Cholesky lớp đã có) |
| SB | synthetic | `synth_benign=19` (3/lớp tấn công) | FP lớp 7/6/1 từ benign tương lai chưa thấy; benign toàn mạng đa dạng hơn | 0 |
| ST | synthetic | `cop_kmeans_temp=0.5` | phủ mode hiếm | 0 |
| GF | synthetic | `synth_gapfill=true` | model khớp mode local; ưu tiên cụm xe thiếu | 0 |
| QS | teacher | `q_source=synth` | Q đo in-sample quá tự tin (Q[13,13] .94 vs recall .29); synthetic là out-of-sample | 0 |

**Luật r20 fold val:** PASS khi Δmacro ≥ +.002 so KQ8 cùng seed + guard chung (p10, timeDelay, benign, absent như trên).
EM được chọn khi E1 và E1S43 cùng PASS; E2 chỉ thay E1 nếu E2 − E1 ≥ +.001 (gấp đôi byte EM). Arm một seed PASS → lặp seed43
trước full-scale. Không tune giá trị knob sau khi xem số; không cộng gain các arm để dự báo combo.

## Kết quả r20 fold val (kiểm 09/11/2026, 14/14 kernel audit PASS)

Receipt: `runs/sim-kaggle-b9/verdict-r20.json`, `runs/sim-kaggle-b10/verdict-r20.json`.

### Tóm tắt phán quyết

1. **Codec (b9): `lite6` PASS cả 2 seed** (LT6 Δmacro −.0002, LT6S43 Δmacro −.0000; đều đạt gate ≥ −.0015 và mọi guard). Dung lượng giảm từ ≈3,67 MB (KM50 gốc) xuống **1,20 MB/xe/50r** (up 323 KB, down 881 KB). LT8 cũng PASS (−.0001) nhưng 1,35 MB. Theo luật §1: **chọn `lite6`** cho pilot full-scale.
2. **K32 (b9): `PK32` FAIL** vs đối chứng PK16 (Δmacro −.0011, Δp10 −.0178, Δbenign −.0009; trượt gate ≥ +.002). K=32 không cải thiện, tăng byte lên 1,82 MB. **Loại K32**.
3. **Lấy mẫu 10% xe (b9): `PP10` FAIL** vs PK16 (Δp10 −.0282, ΔtimeDelay −.0044, Δbenign −.0018; trượt guard p10/timeDelay/benign). 10% xe làm suy giảm tail client. **Loại part 0.1**.
4. **Module variants (b10): Cả 8/8 arm đều FAIL utility gate** (yêu cầu Δmacro ≥ +.002 so KQ8 cùng seed):
   - **EM1 / EM2:** E1 Δmacro −.0005 (FAIL macro); E1S43 Δmacro −.0007 (FAIL macro, timeDelay, benign); E2 Δmacro −.0011 (FAIL macro, timeDelay). `EM selected: False`. **Loại EM**.
   - **Whitening (WH):** Δmacro −.0028 (FAIL macro, p10, timeDelay, benign).
   - **Benign share 19% (SB):** benign F1 tăng +.0113 nhưng timeDelay tụt −.0452, Δmacro −.0011 (FAIL macro, p10, timeDelay).
   - **Nhiệt độ cụm 0.5 (ST):** Δmacro −.0007 (FAIL macro, timeDelay).
   - **Gap-fill (GF):** Δmacro −.0045 (FAIL macro, timeDelay, benign).
   - **Synthetic Q (QS):** sụt giảm nghiêm trọng Δmacro −.0258, absent_acc −.1361.

### Bảng kết quả so sánh b9 và b10

| arm | batch | vs | f1_macro | Δmacro | p10 | low256 | timeDelay | benign | absent_acc | verdict | byte/xe (up + down = tổng) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LT8** | b9 | KQ8 | 0.9147 | −0.0001 | 0.8017 | 0.8678 | 0.3784 | 0.8714 | 0.7120 | **PASS** | 323.314 + 1.028.051 = 1.351.365 |
| **LT6** | b9 | KQ8 | 0.9146 | −0.0002 | 0.7977 | 0.8676 | 0.3709 | 0.8727 | 0.6963 | **PASS** | 323.314 + 881.346 = 1.204.660 |
| **LT6S43** | b9 | KQ8S43 | 0.9142 | −0.0000 | 0.7942 | 0.8331 | 0.3711 | 0.8715 | 0.6754 | **PASS** | 323.314 + 881.346 = 1.204.660 |
| PK32 | b9 | PK16 | 0.9126 | −0.0011 | 0.7912 | 0.8603 | 0.3714 | 0.8706 | 0.6806 | **FAIL** [macro, p10, benign] | 327.697 + 1.489.559 = 1.817.256 |
| PP10 | b9 | PK16 | 0.9123 | −0.0014 | 0.7808 | 0.8314 | 0.3676 | 0.8698 | 0.7068 | **FAIL** [p10, timeDelay, benign] | 150.986 + 882.831 = 1.033.817 |
| E1 | b10 | KQ8 | 0.9143 | −0.0005 | 0.7917 | 0.8536 | 0.3711 | 0.8720 | 0.6963 | **FAIL** [macro] | — |
| E1S43 | b10 | KQ8S43 | 0.9136 | −0.0007 | 0.7841 | 0.8197 | 0.3637 | 0.8707 | 0.7016 | **FAIL** [macro, timeDelay, benign] | — |
| E2 | b10 | KQ8 | 0.9138 | −0.0011 | 0.7968 | 0.8547 | 0.3696 | 0.8723 | 0.7173 | **FAIL** [macro, timeDelay] | — |
| WH | b10 | KQ8 | 0.9120 | −0.0028 | 0.7873 | 0.8561 | 0.3630 | 0.8709 | 0.6806 | **FAIL** [macro, p10, timeDelay, benign] | — |
| SB | b10 | KQ8 | 0.9138 | −0.0011 | 0.7847 | 0.8384 | 0.3291 | 0.8828 | 0.6806 | **FAIL** [macro, p10, timeDelay] | — |
| ST | b10 | KQ8 | 0.9142 | −0.0007 | 0.8017 | 0.8535 | 0.3685 | 0.8715 | 0.6702 | **FAIL** [macro, timeDelay] | — |
| GF | b10 | KQ8 | 0.9103 | −0.0045 | 0.7959 | 0.8248 | 0.3690 | 0.8702 | 0.7487 | **FAIL** [macro, timeDelay, benign] | — |
| QS | b10 | KQ8 | 0.8890 | −0.0258 | 0.7761 | 0.8332 | 0.3505 | 0.8686 | 0.5288 | **FAIL** [macro, p10, timeDelay, benign] | — |

### Phân tích bổ sung (09/11, đọc lại CM fold val đã audit, không đổi verdict)

Script tái tạo: `scripts/analyse_km_lite_detail.py` → in ra stdout. Lớp: 0 benign, 7 positionMirroring, 11 suddenConstantSpeed,
13 timeDelay, 15 zeroSpeedReport.

- Ổn định: Δmacro r16/r18/r20 cùng dấu, không arm nào ≥ +.002 ở bất kỳ round nào. Nhiễu: KQ8 − KQ8S43 = +.0006;
  độ lệch chuẩn chênh round r16–r20 .0003–.0005 (KQ8, KQ8S43, LT6). FAIL không do nhiễu.
- PK16 − LT6 (pop 37 M dòng vs sim 1,5 M, chỉ báo): macro −.0010, p10 +.011, timeDelay +.001.
- Lớp nhạy nhất: c11 giảm ở PK32 −.011, PP10 −.009, E1S43 −.008, WH −.014, GF −.017, QS −.193; c13 giảm ở WH −.011, SB −.045.
- SB: benign P −.013 R +.043; timeDelay P +.138 R −.070; dự đoán benign +3.630, timeDelay −2.397 → dời ngưỡng.
- QS: c11 R −.278, c11→c15 130 → 436 dòng; absent −.136.
- GF: absent +.084; c11 P −.047 (FP), c1 F1 −.013.
- Absent giữa hai arm không hiệu ứng: LT6 +.031, LT6S43 −.026.

Kết luận áp dụng: đóng băng generator **KM16 w0 .5 + codec lite6** (owner 09/11); không thêm biến thể K/EM/whitening/share/temp/gap-fill/Q synth.

### Đủ 10 metric phân loại

| arm | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted |
|---|---|---|---|---|---|---|---|---|---|---|
| LT8 | 0.9232 | 0.9240 | 0.9232 | 0.9166 | 0.9107 | 0.9232 | 0.9232 | 0.9147 | 0.9232 | 0.9175 |
| LT6 | 0.9236 | 0.9251 | 0.9236 | 0.9167 | 0.9099 | 0.9236 | 0.9236 | 0.9146 | 0.9236 | 0.9175 |
| LT6S43 | 0.9230 | 0.9243 | 0.9230 | 0.9162 | 0.9100 | 0.9230 | 0.9230 | 0.9142 | 0.9230 | 0.9169 |
| PK32 | 0.9224 | 0.9204 | 0.9224 | 0.9155 | 0.9098 | 0.9224 | 0.9224 | 0.9126 | 0.9224 | 0.9166 |
| PP10 | 0.9217 | 0.9216 | 0.9217 | 0.9147 | 0.9081 | 0.9217 | 0.9217 | 0.9123 | 0.9217 | 0.9159 |
| E1 | 0.9235 | 0.9237 | 0.9235 | 0.9165 | 0.9107 | 0.9235 | 0.9235 | 0.9143 | 0.9235 | 0.9173 |
| E1S43 | 0.9228 | 0.9238 | 0.9228 | 0.9157 | 0.9091 | 0.9228 | 0.9228 | 0.9136 | 0.9228 | 0.9166 |
| E2 | 0.9235 | 0.9236 | 0.9235 | 0.9165 | 0.9098 | 0.9235 | 0.9235 | 0.9138 | 0.9235 | 0.9173 |
| WH | 0.9222 | 0.9226 | 0.9222 | 0.9151 | 0.9076 | 0.9222 | 0.9222 | 0.9120 | 0.9222 | 0.9158 |
| SB | 0.9281 | 0.9388 | 0.9281 | 0.9239 | 0.9048 | 0.9281 | 0.9281 | 0.9138 | 0.9281 | 0.9184 |
| ST | 0.9229 | 0.9244 | 0.9229 | 0.9161 | 0.9096 | 0.9229 | 0.9229 | 0.9142 | 0.9229 | 0.9168 |
| GF | 0.9214 | 0.9187 | 0.9214 | 0.9144 | 0.9077 | 0.9214 | 0.9214 | 0.9103 | 0.9214 | 0.9151 |
| QS | 0.9162 | 0.9189 | 0.9162 | 0.9109 | 0.8760 | 0.9162 | 0.9162 | 0.8890 | 0.9162 | 0.9078 |
