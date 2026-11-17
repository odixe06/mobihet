# Lite2 — bootstrap v2 (km2) và liều 8 dòng trên T3 (khóa 16/11/2026, trước push)

Owner 16/11: đích xây lại seed 42, đơn model — tổng byte ≤ 0,40 MB/xe/50r, compute ≤ 1,37× FedAvg, Macro-F1 r50 DAGSNet > .8998,
BERT ≥ .8951; mở lại knob chi phí cũ như adaptation mới với control ghép cặp và gate mới; owner duyệt toàn bộ arm + gate dưới đây.
**Yêu cầu mới 17/11:** compute đích1× FedAvg và top5; riêng compute bằng FedAvg là đạt, không cần thấp hơn; gate≤1,37× dưới đây chỉ là luật Lite2 đã khóa, PASS không đồng nghĩa đạt mục tiêu mới. Verdict cũ (K8, G1, T5…) giữ nguyên theo gate lúc đó. Control = T3 ([results-t3](results-t3.md)); yêu cầu ở [task brief](../../task_brief.md).

## Bằng chứng → cơ chế → lợi kỳ vọng → dấu hiệu bác bỏ

- **Bằng chứng byte** (probe local không train, panel sim 200 xe, `runs/lite-byte-probe/`): bootstrap = 98,9 % tổng byte T3
  (1,151 MB full-scale; online 13–17 KB/50r) ⇒ top 5 communication BERT (< .4128 MB) chỉ đạt được bằng cắt bootstrap ≈ 65 %.
  Phần lớn byte là Cholesky cụm (downlink) và moment bậc hai theo ô xe×lớp×cụm (uplink); 51 % ô có < 16 dòng nhưng chỉ giữ 5 % dòng (K16).
  CDF mã hóa mass + coder tốt nhất 258,7 → 157,8 KB không mất thông tin.
- **Bằng chứng compute:** T3 ≈ 1,652× FedAvg; 600 xe ≥ 16.384 dòng giữ 63 % bước synthetic, mất macro của knob giảm liều nằm ở xe nhỏ
  ([B21](screening-dose-b21.md), [LESSONS](../../LESSONS.md)).
- **Cơ chế:** (byte) bớt thành phần mixture (K 6/8), bảng biên 64 bin, Lloyd + moment lớp trên 25 % xe (lượt cuối mọi xe),
  ô < τ dòng chỉ gửi count + mean int8, Cholesky 5/6 bit, coder lossless tốt nhất — giữ nguyên sampling/Q/loss/lịch T3.
  (compute) tầng M/L nhận 32/16 và 16/8 dòng synthetic/bước (D8), hoặc cắt cả tầng S nửa sau lịch (U1); 8 dòng xoay lớp 0–7/8–15 theo bước.
- **Khác study đã đóng:** K8 (b9, −.002) và T5/G1 FAIL dưới gate non-inferiority −.0015; nay owner chấp nhận đổi tối đa .015 macro r50
  lấy chi phí ⇒ gate mới dưới đây. km2 (lấy mẫu xe, ngưỡng ô τ, coder) chưa từng train.
- **Lợi kỳ vọng:** byte 1,196 MB → 0,36–0,38 MB (preflight); compute 1,652× → ≈ 1,36×; macro giảm trong ngân sách gate. Không kỳ vọng tăng macro.
- **Bác bỏ:** arm trượt gate sim; hoặc byte bootstrap thật full-scale > ≈ 382 KB; hoặc pilot/full trượt luật dưới. FAIL → dừng arm, không tune/kéo dài.

## Siêu tham số giữ nguyên

[Protocol](protocol.md) không đổi (cosine 1e-3 → 1e-5, 1 epoch dòng thật, batch 64, AdamW, clip, AMP, W1, θ⁰, mọi xe mỗi round, cách chấm).
hp nền = T3 sim (`runs/lite-byte-probe/t3_sim_hp.txt`, `synth_late_from=10`); mỗi arm chỉ thêm knob trong bảng.
Code: `methods/mobihet_lite.py` (km2 tắt = T3 bit-exact), tier 8 dòng trong `methods/mobihet_compute.py`; `tests/test_mobihet_lite.py` 8/8.

## Arm — sim fold val 200 xe, r20 cosine20, DAGSNet seed 42 (`scripts/run_sim_lite2.py`, 12 kernel, CAP 2,75 h)

Bootstrap: byte/xe trên panel sim (preflight không train, `runs/lite-byte-probe/boot_preflight{,2}.json`); full-scale đọc lại ở round 0.
Liều (S/M/L, đầu lịch/sau mốc): T3 `[[64,64],[64,32],[48,16]]` 1,652× · D8 `[[64,64],[32,16],[16,8]]` 1,361× · U1 `[[64,32],[32,16],[16,16]]` 1,364×.

| arm | bootstrap | liều | knob | byte bootstrap sim |
|---|---|---|---|---:|
| S0 | T3 (KM16 lite6) | T3 | control, `mobihet_compute` | 1 196 036 |
| S1 | **B**: K6, 64 bin, frac .25, τ16, 6-bit, mean fp16 | T3 | | 362 363 |
| S2 | **A32**: K8, 64 bin, frac .25, τ32, 5-bit, mean int8 | T3 | thay A (τ16 392 019 vượt ngân sách) | 372 355 |
| S3 | T3 | D8 | | 1 196 036 |
| S4 | B | D8 | | 362 363 |
| S5 | A32 | D8 | | 372 355 |
| S6 | T3 | U1 | | 1 196 036 |
| S7 | B | U1 | | 362 363 |
| S8 | A24: như A32, τ24 | T3 | sát ngân sách | 380 664 |
| I1 | km2, chỉ 64 bin | T3 | cô lập | 914 323 |
| I2 | km2, chỉ K6 | T3 | cô lập | 590 185 |
| I3 | km2, chỉ τ16 + frac .25 | T3 | cô lập | 895 129 |

## Luật khóa (macro-first; guard p10/low256/TD/benign/absent report-only)

- **Sim** (so S0 cùng round, artifact audit PASS trước): PASS nếu Δf1_macro TB r16–20 ≥ −.005 **và** Δf1_macro r20 ≥ −.006.
  I1–I3 chỉ báo cáo (định vị nguồn mất mát), không lên bậc. Cả B và A PASS → chọn bootstrap byte thấp hơn; liều: arm PASS compute thấp nhất
  trên bootstrap được chọn (combo phải tự PASS, không cộng PASS từng knob). Một seed: screening, không là significance.
- **Pilot** full-scale 5.343 xe, r1–20 cosine50, DAGSNet + BERT, so T3 seed 42 cùng round: Δmacro DAGSNet ≥ −.006, BERT ≥ −.010,
  áp cho **cả** TB r16–20 **và** r20 (khóa 17/11 trước push pilot: đọc chặt, thỏa cả hai cách hiểu; chỉ một trong hai đạt → FAIL theo luật,
  báo owner); byte bootstrap thật ≤ 382 000 B/xe (`config.json`; 0,40 MB − online BERT .0174 MB). Audit artifact PASS trước.
  Port BERT = copy byte-identical hai file method (overlay SHA phải bằng bản sim đã audit).
- **Full r50** (nối checkpoint, cosine50): DAGSNet > .8998, BERT ≥ .8951; tổng ≤ 0,40 MB/xe/50r; compute ≤ 1,37× FedAvg; báo hạng performance/communication.
- PASS tự lên bậc kế; FAIL dừng arm. Mixed DAGSNet+BERT chỉ sau khi đơn model PASS. Báo đủ 10 metric, per-class, cohort cỡ xe, byte, FLOPs, `t_train`.

## Kết quả

### Sim r20 (17/11, 12/12 audit PASS: `runs/sim-kaggle-lite2/{audit,verdict-r20}.json`, guard [report.txt](../../runs/sim-kaggle-lite2/report.txt))

Control S0: macro r20 .9138, TB r16–20 .9134, p10 .8046, skip 0 mọi arm. Δ so S0 cùng round; tầng = macro gộp theo cohort dòng fold train
(S < 4.096: 131 xe · M: 44 · L ≥ 16.384: 25); byte bootstrap = đo trong run (2 rank), lệch preflight ≤ 37 B.

| arm | bootstrap + liều | byte bootstrap | Δ TB r16–20 | Δ r20 | Δp10 | Δ tầng S / M / L | Δ benign F1 | `t_train` h | verdict |
|---|---|---:|---:|---:|---:|---|---:|---:|---|
| S1 | B | 362 351 | −.0021 | −.0022 | −.0157 | −.0080 / +.0005 / −.0021 | −.0006 | .840 | PASS |
| S2 | A32 | 372 392 | −.0023 | −.0027 | −.0066 | −.0076 / −.0001 / −.0031 | −.0016 | .854 | PASS |
| S8 | A24 | 380 652 | −.0026 | −.0030 | −.0092 | −.0087 / −.0026 / −.0022 | +.0000 | .841 | PASS |
| S3 | T3 + D8 | 1 196 036 | −.0032 | −.0031 | −.0056 | −.0039 / −.0037 / −.0026 | −.0046 | .899 | PASS |
| S6 | T3 + U1 | 1 196 036 | −.0028 | −.0026 | −.0045 | −.0081 / −.0026 / −.0013 | −.0017 | .875 | PASS |
| S4 | B + D8 | 362 351 | **−.0051** | −.0053 | −.0221 | −.0084 / −.0070 / −.0037 | −.0028 | .834 | **FAIL** (TB) |
| S7 | **B + U1** | 362 351 | −.0047 | −.0048 | −.0246 | −.0101 / −.0047 / −.0036 | −.0010 | .780 | PASS |
| S5 | A32 + D8 | 372 392 | −.0040 | −.0042 | −.0175 | −.0071 / −.0028 / −.0046 | −.0022 | .821 | PASS |
| I1 | chỉ 64 bin | 914 355 | −.0012 | −.0009 | −.0062 | −.0036 / −.0009 / +.0000 | +.0010 | .835 | báo cáo |
| I2 | chỉ K6 | 590 189 | −.0023 | −.0023 | −.0163 | −.0094 / −.0020 / −.0009 | −.0017 | .859 | báo cáo |
| I3 | chỉ τ16 + frac .25 | 895 149 | −.0012 | −.0011 | −.0026 | −.0069 / −.0008 / −.0006 | −.0011 | .888 | báo cáo |

- **Theo luật khóa:** B và A32 cùng PASS → bootstrap **B** (byte thấp hơn); liều trên B: D8 FAIL (TB −.0051, lệch ngưỡng .0001), U1 PASS
  ⇒ lên pilot **B + U1** (bootstrap sim 362 KB, compute 1,364×). A32 + D8 cũng PASS nhưng không được chọn (bootstrap byte cao hơn).
- Biên mỏng: B + U1 cách ngưỡng TB .0003, một seed; mất mát gần cộng tính (B −.0021 + U1 −.0028 = −.0049 vs đo −.0047;
  B + D8 −.0053 vs −.0051; A32 + D8 −.0055 vs −.0040) ⇒ nhiễu cỡ .001, pilot mới là phép thử.
- Mất tập trung ở lớp 11 (−.020…−.031 mọi arm kể cả chỉ đổi liều), lớp 13 (−.015…−.023 ở arm 64 bin: I1 −.018, I2/I3 ≈ 0) và lớp 1
  (K6/liều: −.015…−.018); F1 lớp 13 của control chỉ .375.
- p10 (report-only): K6 là nguồn chính (I2 −.016; I1 −.006; I3 −.003); B + liều −.022…−.025, A32 −.007. Tầng S chịu gần hết mất mát bootstrap
  (−.007…−.010), M/L ≈ 0. Cohort < 256 dòng chỉ 9 xe: nhiễu, không kết luận. absent_fp không đổi (≤ +.0002).
- Byte round r1–20 ≈ 6,1–6,4 KB/xe mọi arm (U1/D8 +3 %).

### Pilot / full

Pilot DAGSNet **B + U1** seed 42 (`scripts/run_lite2_full.py`, `runs/lite2-full/`): hp = T3 pilot seed 42 (`runs/b21-full/batch.json`, T3P)
+ knob B + U1, `synth_late_from=25` ⇒ r1–20 chỉ chạy cột đầu của U1 (S64/M32/L16); cắt tầng S 64 → 32 từ r26 chỉ được thử ở r50.
Control = T3P r1–20 đã audit (macro r20 .8995, TB r16–20 .8980, `t_train` 3,77 h); source core/methods của control = HEAD.

Pilot FeatureBERT **B + U1** seed 42 (`../fl_veremi_bert/scripts/run_lite2_pilot.py`, `runs/lite2-pilot/` ở repo BERT): hp = T3 BERT seed 42
(`runs/t3/batch.json`) + knob B + U1 — cùng tập hp với pilot DAGSNet; base BERT `b93198a` (core/methods = source control) + overlay hai file method.
Control = T3 BERT seed 42 r1–20 (macro r20 .9029, TB r16–20 .9009, p10 .7375, `t_train` 5,95 h). Trước push: `check_t3_port.py` PASS,
smoke local 100 xe 1 round (compile gate PASS, skip 0, KM2 96/96 cụm giữ).

**Pilot DAGSNet B + U1 — PASS (17/11, `runs/lite2-full/audit.json`; 12/12 kiểm artifact PASS: round 1..20, source/HP, LR, skip 0,
CM support = control, 10 metric từ CM).** So T3P cùng round:

| | B + U1 | T3P | Δ | ngưỡng |
|---|---:|---:|---:|---:|
| Macro-F1 r20 | .8968 | .8995 | −.0028 | ≥ −.006 |
| Macro-F1 TB r16–20 | | | −.0028 | ≥ −.006 |
| byte bootstrap/xe (up 98 943 + down 241 720) | 340 663 | 1 151 278 | −70,4 % | ≤ 382 000 |
| byte round r1–20/xe | 6 470 | 6 417 | +0,8 % | |
| client F1 p10 / mean / median | .7231 / .8474 / .8665 | .7260 / .8513 / .8711 | −.0028 / −.0039 / −.0047 | report |
| accuracy / precision macro / recall macro | .9135 / .9072 / .8930 | .9150 / .9045 / .9003 | −.0015 / +.0028 / −.0073 | report |
| `t_train` r1–20 | 3,64 h | 3,77 h | −3,5 % | |

- Δmacro theo round: −.0168 (r1) → −.0101 (r3) → −.0053 (r8) → −.0038 (r12) → −.0026…−.0032 (r14–20): khoảng cách thu hẹp rồi phẳng.
- Per-class r20 (Δ F1): lớp 1 −.010, 13 −.008, 11 −.007, 2/4 −.005, 6 +.008; còn lại |Δ| ≤ .004. Mất chủ yếu ở recall.
- Bootstrap full-scale 340 663 B < panel sim 362 351 B (trung bình uplink/xe thấp hơn khi đủ 5 343 xe). Ước tổng 50r ≈ 0,357 MB.
- Theo luật: PASS → nối r21–50 (`dinhan25/mh-lite2-bu1-s42-r50/1`, parent = state 2 rank r20 của kernel pilot, cùng hp/source).
  Verdict full (`run_lite2_full.py audit r50`): macro r50 > .8998, bootstrap + mọi round ≤ 400 000 B, compute 1,364× ≤ 1,37×,
  r1–20 phải trùng byte với pilot. Control r50 T3 seed 42 = .9077 ⇒ dư địa Δ ≈ −.0079; cột sau của U1 (S 64 → 32 từ r26) chưa được pilot thử.

**Pilot BERT B+U1 PASS** (17/11, watcher COMPLETE 09:18 +07, pull108file/0thiếu, audit rc0):
macro r20 .898173076 vs T3 .902919201 (Δ−.004746125); Δ TB r16–20 −.005172594, cả hai≥−.010.
Bootstrap340663,237B/xe (up98942,875 + down241720,362); online20round7575,610B/xe;
`t_train`5,8427h vs5,9499h (−1,8%); p10 .734552 (Δ−.002985).
Audit bounded r1–20/CM-support/10metric/LR/HP/source/finite PASS, mỗi client có appliedstep;
đã chạy lại bounded auditor độc lập trong phiên tổng hợp brief. Pull không có state2rank, không claim fullresumeaudit.
[Receipt BERT](../../../fl_veremi_bert/runs/lite2-pilot/audit.json); [10metric hai model](../../task_brief.md) §13.

**Full DAGSNet B+U1 seed42 r50 — PASS theo gate Lite2 cũ** (17/11, thu output và audit sau khôi phục watcher).
[Receipt audit + 10 metric](lite2-dagsnet-r50-audit.json): đủ r1–50, prefix r1–20 trùng pilot; source/HP/LR/CM-support/finite/applied-step PASS.
Macro .905876977 vs T3 .907678902 (Δ−.001801925); Δ TB r46–50 −.001820567.
Bootstrap340663,237B + online50round13453,067B = **354116,304B/xe (0,354116 MB)**.
Compute estimator **1,364× ≤1,37×**, chưa đạt đích FB mới ≤1,01×; không là đo total GPU FLOPs.
`t_train`8,5147h vs9,3213h (−8,7%); p10 .767284 vs .777778 (Δ−.010494), report-only.
Lớp13 F1 .335406 (Δ−.013952); lớp11 .841556 (Δ−.005632); benign .850054 (Δ+.007321).
Một seed; audit bounded, chưa prediction replay/full private-bank rebuild.

| Metric r50 DAGSNet | B+U1 |
|---|---:|
| accuracy | 0.914633247 |
| precision_macro | 0.913456340 |
| precision_micro | 0.914633247 |
| precision_weighted | 0.907816690 |
| recall_macro | 0.901728987 |
| recall_micro | 0.914633247 |
| recall_weighted | 0.914633247 |
| f1_macro | 0.905876977 |
| f1_micro | 0.914633247 |
| f1_weighted | 0.909892142 |

Full BERT r50 **hoãn theo owner17/11**, không tự nối/push lại. Ưu tiên [FB](screening-budget.md).
Job/kernel: [continue](../operations/continue.md).
