# B21 — liều synthetic theo cỡ xe × pha lịch trên CBB (khóa 13/11/2026, trước push)

Owner 13/11: compute tại xe **nhỏ nhất khả thi mà không ảnh hưởng performance**; chạy sim b21; thêm một arm floor 48
(mở lại hướng b đã đóng 09/11 dưới dạng tái phân bổ compute). Nối tiếp [B20](screening-compute-b20.md).

## Bằng chứng → cơ chế → lợi kỳ vọng → dấu hiệu bác bỏ

- **Bằng chứng** (CM full-scale đã audit, `scripts/analyse_b20_cohort.py` → [cohort.txt](../../runs/b20-cohort/cohort.txt)):
  600 xe ≥ 16.384 dòng giữ 63% bước synthetic, 3.820 xe < 4.096 dòng chỉ 14,6%. Mất macro của mọi knob giảm synthetic dốc theo cỡ xe:
  macro riêng cohort < 256 dòng vs ≥ 16k — SD r50 −.005/−.009 vs +.0004/+.0005; SM42 r50 −.012 vs −.0007; SR48 r20 −.014 vs −.0011;
  SR32 r20 −.026 vs −.0023. SD (cắt nửa sau lịch) PASS r50 hai seed với Δ −.0002/−.0001.
- **Cơ chế giả thuyết:** xe nhiều bước nhận lượng dòng synthetic mỗi round tỷ lệ số bước (1:1 với dòng thật) từ cùng một mixture 256 thành phần,
  lợi biên giảm theo số bước; xe ít bước/thiếu lớp cần đủ liều. Cắt liều ở xe lớn và ở nửa sau lịch lấy phần lớn FLOPs với mất mát nhỏ.
- **Khác họ đã đóng:** G1 cắt đều mọi xe từ r1 (FAIL low256/absent); b1 bỏ hẳn synthetic ở xe đủ dòng (hại đơn điệu); B20 SD cắt đều theo thời gian.
  B21 giữ nguyên 64 dòng cho xe < 4.096 dòng và không bỏ hẳn synthetic ở xe nào.
- **Bác bỏ:** arm trượt luật non-inferiority dưới ở một seed; hoặc xe lớn sụt như SR32 r20 mà không co lại.

## Siêu tham số giữ nguyên (đồng nhất `fl_dagsnet_veremi_v2`, [protocol](protocol.md))

Không arm nào đổi LR (cosine 1e-3 → 1e-5 theo round), số round, 1 epoch dòng thật, batch 64, AdamW/weight decay, clip, AMP, dropout, W1, θ⁰,
tham gia mọi xe, cách chấm. hp = CBB + đúng knob dưới. Mốc đổi liều định nghĩa theo lịch: round > R/2 (sim cos20: > 10; full-scale cos50: > 25).
Ngưỡng cỡ xe tính trên số dòng train local của xe trong run đó (sim: fold train), không thêm byte. Byte mỗi round và bootstrap = CBB.

## Arm (method `mobihet_compute`, knob `synth_tier_min=[4096,16384]`, `synth_tier_rows`, `synth_late_from`; knob tắt = CBB bit-exact)

Dòng synthetic/bước (đầu lịch, sau mốc) theo tầng S < 4.096 · M 4.096–16.383 · L ≥ 16.384 dòng train:

| arm | S | M | L | floor | FLOPs/dòng thật full-scale | ghi chú |
|---|---|---|---|---:|---:|---|
| C | 64, 64 | 64, 64 | 64, 64 | 16 | 6,09×F | CBB |
| T1 | 64, 64 | 64, 32 | 64, 32 | 16 | 5,42×F | SD chỉ ở xe ≥ 4.096 dòng |
| T3 | 64, 64 | 64, 32 | 48, 16 | 16 | 4,94×F | |
| T4 | 64, 64 | 48, 32 | 32, 16 | 16 | 4,61×F | |
| T5 | 64, 64 | 32, 16 | 16, 16 | 16 | 4,19×F | 16 dòng ở xe lớn chưa có dữ liệu |
| T4F | như T4 | | | 48 | 5,10×F | tái phân bổ: floor 48 bước synthetic-only cho xe nhỏ |

FLOPs theo quy ước B20: dòng thật 2,99×F; synthetic = 2,99 × Σ_k max(s_k, floor) · (dòng TB của xe k)/64 / Σ_k s_k (s_k = bước thật; tái tạo CBB 6,09 và SD 5,32).
Mô phỏng sinh số dòng lớn nhất của round rồi che dòng thừa của từng xe khỏi BN batch và khỏi trung bình loss, nên `t_train` sim không phản ánh FLOPs.
Test: `tests/test_mobihet_compute.py` (loss/gradient từng xe = CBB chạy với đúng số dòng của xe đó; mọi tầng 64 = CBB; floor hữu hạn).
Panel sim 200 xe: S/M/L = 131/44/25 xe, tỷ trọng bước synthetic 12,4/25,2/62,3% (full-scale 14,6/22,7/62,7%).

## Luật khóa (mỗi seed, so C cùng seed; macro-first, guard phụ report-only kèm trend)

**Sim fold val 200 xe r20 cosine20, seed 42 và 43** (12 kernel): PASS nếu Δf1_macro TB r16–r20 ≥ −.0015 **và** Δf1_macro r20 ≥ −.003
(luật B20/G1/codec). Ứng viên = PASS cả hai seed. Điều khoản “lên theo quỹ đạo” của phụ lục B20 §3 **không áp** cho B21
(mục tiêu owner: không ảnh hưởng performance). T4F báo thêm Δ so T4 cùng seed (tác dụng riêng của floor 48, report-only).

**Thang** ([CONTEXT](../../CONTEXT.md)): ứng viên có cắt đầu lịch (T3/T4/T5/T4F) → pilot full-scale r20 cosine50 seed42 so CBB42 cùng round, cùng luật
trên r16–r20 → PASS → nối r50 + seed43 r50. T1 (không đổi trước r26) → thẳng r50 hai seed như SD. Quota thiếu → arm rẻ FLOPs nhất đi trước.
**Luật r50:** Δmacro TB r46–r50 ≥ −.0015 và Δ r50 ≥ −.003 so CBB cùng seed, **và** macro r50 > .9048 mỗi seed.
Nhiều arm PASS r50 → chọn arm FLOPs thấp nhất. FAIL ở bất kỳ bậc → dừng nhánh, không tune ngưỡng/số dòng, không kéo dài.
Báo đủ 10 metric, p10/low256/TD/benign/absent, Δ theo cohort cỡ xe, byte, FLOPs, `t_train`.

## Kết quả sim (13/11, audit PASS 12/12, [verdict](../../runs/sim-kaggle-b21/verdict-r20.json), [10 metric](../../runs/sim-kaggle-b21/report-r20.md))

Control macro r20 .9136 (s42) / .9131 (s43). Δ so C cùng seed; tầng = macro riêng cohort S/M/L tại r20.

| arm | FLOPs | s42 ΔTB16–20 / Δr20 | s43 ΔTB16–20 / Δr20 | Δp10 s42/s43 | Δ tầng S/M/L s42 | Δ tầng S/M/L s43 | verdict |
|---|---:|---|---|---|---|---|---|
| T1 | 5,42×F | −.0005 / −.0005 | +.0002 / +.0008 | −.0017 / +.0100 | +.0010/−.0007/−.0010 | +.0002/+.0006/+.0011 | **PASS** |
| T3 | 4,94×F | +.0005 / +.0008 | +.0001 / +.0001 | +.0010 / +.0010 | +.0003/−.0008/+.0016 | +.0010/−.0009/+.0005 | **PASS** |
| T4 | 4,61×F | +.0001 / +.0001 | −.0008 / −.0008 | −.0031 / −.0072 | +.0003/−.0021/+.0010 | +.0017/−.0016/−.0005 | **PASS** |
| T5 | 4,19×F | −.0030 / −.0026 | −.0013 / −.0008 | −.0121 / +.0005 | −.0007/−.0031/−.0028 | +.0022/−.0019/−.0007 | **FAIL** (s42 TB) |
| T4F | 5,10×F | +.0006 / +.0010 | −.0001 / +.0001 | −.0069 / +.0075 | +.0054/−.0000/+.0003 | +.0094/−.0004/−.0016 | **PASS** |

- Ứng viên: T1, T3, T4, T4F. T5 dừng (16 dòng ở xe ≥ 4.096 dòng: tầng M −.003/−.002, L −.003; ΔTB r6–10 −.006/−.005).
- Xe nhỏ (tầng S, giữ 64 dòng) không sụt ở arm nào; mất mát nằm ở tầng bị cắt và lớn dần theo mức cắt (M: 64→48 −.002, →32 −.003).
- Cắt đầu lịch làm chậm nửa đầu (T4 ΔTB r6–10 −.003/−.002) rồi co lại tới r20; T1 (không cắt trước mốc) không có pha chậm.
- Floor 48 (T4F − T4, report-only): macro +.0006/+.0007 (TB), tầng S +.005/+.008, absent +.06/+.07, low256 −.033/−.004, giá +0,49×F.
- Byte/round sim 305–317 (C 306), `t_train` 50–53 phút mọi arm (mô phỏng che dòng, không rút shape).
- Audit lần đầu FAIL giả ở check `config`: auditor so chuỗi hp dạng list có dấu cách; bản FAIL giữ ở `audit-config-false-fail.{json,log}`,
  sửa chuẩn hóa chuỗi trong `audit_sim_batch.py`, 13 check còn lại không đổi.

## Full-scale (thang, luật ở trên không đổi)

`scripts/run_b21_full.py` → `runs/b21-full/`: pilot r20 cosine50 seed 42, base/hp/overlay = CBB42 + knob tầng, `synth_late_from=25`
(pilot chỉ đo pha đầu lịch). T4P, T3P cap 5,5 h; T4FP cap 6,0 h (floor 48 thêm bước). So CBB42 cùng round; PASS → nối r50 + seed 43.
T1 r50 hai seed (`runs/b21-t1/`): 13/11 không account nào còn ≥ 13,2 h → phiên 1 dừng theo budget (cap 5,5 h / 4,4 h), nối phiên cùng owner tới r50.


## Audit full-scale 13/11 19:15 VN

- T4P: artifact audit PASS, utility FAIL; macro 0.89865614, Δ CBB -0.00132817; Δ TB cuối5 -0.00158439; train 3.628 h.
- T3P: artifact audit PASS, utility PASS; macro 0.89951889, Δ CBB -0.00046542; Δ TB cuối5 -0.00053078; train 3.769 h.
- T4FP: artifact audit PASS, utility PASS; macro 0.89899785, Δ CBB -0.00098646; Δ TB cuối5 -0.00120349; train 4.169 h.

10 metric và checks: `runs/b21-full/audit.json`. Audit CM/config/source metadata; chưa replay prediction từ full weights.

T1 partial audit: `runs/b21-t1/partial-audit/audit.json`; T142 r22/T143 r18 checks PASS, complete=False, verdict PENDING_R50. Không áp gate r50 lên partial.

## Full-scale r50 — audit chuỗi 15/11 00:30 VN

`python scripts/run_b21_chain.py audit` → [audit.json](../../runs/b21-r50/audit.json): 6/6 head đủ r1–r50, 10 check artifact PASS
(round, n_params, finite, skip ≤ 1%, clients_applied, LR = CBB, hp = launch, overlay SHA, CM support = CBB, 10 metric từ CM) và
`rounds.jsonl` mọi phiên trước là prefix đúng của head (T142 r22→25→34→50, T143 r18→47→50, T3 s42 r20→27→36→50, T4F s42 r20→24→49→50,
T343 r7→50, T4F43 r4→50). Guard/cohort: `runs/b21-r50/analyse_r50.py` → [report.txt](../../runs/b21-r50/report.txt). Δ so CBB cùng seed, r50.

| arm | FLOPs | macro s42 / s43 | Δmacro r50 | Δ TB r46–50 | Δp10 | Δ cohort < 256 dòng | Δ tầng S / M / L (s42; s43) | Δ benign / TD F1 | `t_train` vs CBB | verdict |
|---|---:|---|---|---|---|---|---|---|---|---|
| T1 | 5,42×F | .9076 / .9085 | +.0002 / +.0003 | +.0002 / +.0003 | −.0007 / +.0023 | +.0006 / −.0006 | −.0000/−.0004/+.0006; +.0000/−.0002/+.0006 | +.002 / +.000 | +3% / +1% | **PASS** |
| T3 | **4,94×F** | .9077 / .9083 | +.0003 / +.0002 | +.0003 / +.0002 | +.0011 / +.0028 | +.0016 / +.0009 | +.0007/−.0004/+.0005; +.0008/−.0003/+.0003 | +.003 / −.001 | +1% / −4% | **PASS** |
| T4F | 5,10×F | .9080 / .9087 | +.0006 / +.0005 | +.0006 / +.0005 | **+.0118 / +.0134** | **+.0120 / +.0094** | +.0054/−.0008/+.0001; +.0048/−.0006/+.0001 | +.003 / −.002 | +16% / +11% | **PASS** |
| SD (B20) | 5,32×F | .9072 / .9081 | −.0002 / −.0001 | −.0003 / −.0001 | −.0068 / −.0041 | −.0048 / −.0086 | −.0027/−.0004/+.0004; −.0026/−.0002/+.0005 | +.002 / −.001 | −4% / −10% | PASS (B20) |

- **T1, T3, T4F PASS luật r50 cả hai seed** (Δ TB r46–50 ≥ −.0015, Δ r50 ≥ −.003, macro > .9048). Theo luật khóa (nhiều arm PASS → FLOPs thấp nhất): **T3, 4,94×F (−19% so CBB 6,09)**.
  T4F đắt hơn T3 0,16×F nhưng là arm duy nhất tăng xe nhỏ lặp lại ở hai seed (p10, cohort < 256 dòng, tầng S). **Owner 15/11 chọn T3 làm phương pháp chính**; bổ sung DAGSNet seed41, BERT chờ chốt phạm vi.
- Δmacro +.0002…+.0006 nằm trong nhiễu seed của CBB (±.0004, 3 seed): kết luận là **không kém CBB**, không phải tăng macro.
- Giữ 64 dòng cho tầng S xóa phần mất xe nhỏ của SD (tầng S −.0027/−.0026 → ±.0008; p10 −.007/−.004 → ≥ −.0007) — khớp cơ chế giả thuyết.
- Arm cắt đầu lịch chậm ở pha đầu rồi co lại: T4F Δmacro r10 −.0031/−.0024, r20 −.0010/−.0008, r40 +.0003, r50 +.0006/+.0005;
  T3 r10–r30 −.0005…−.0010 → r50 +.0003/+.0002. Pilot r20 đo pha bất lợi nhất của các arm này.
- Mọi arm đẩy cùng trục benign↔TD: benign F1 +.002…+.003 (recall +.004…+.009), TD precision +.005…+.012 / recall −.002…−.010, TD F1 −.002…+.001.
- Byte/round 266–272 (CBB 265–266; T4F +6 B do Q ở xe floor), bootstrap = CBB. `t_train` trên T4 không giảm (engine che dòng thừa trong nhóm,
  launch-bound); T4F +11–16% vì thêm bước floor. FLOPs là chi phí theo số dòng mỗi xe tự sinh (công thức ở mục Arm), không đo từ `t_train`.
- Giới hạn: 2 seed (CBB có 3 seed), official test đã đọc lặp; audit CM/config/source, chưa replay prediction từ weights; T4 dừng ở pilot theo luật, không mở lại.
