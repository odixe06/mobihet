# B20 — giảm compute tại xe trên CBB (khóa 12/11/2026, trước push)

Owner 12/11: so với `fl_dagsnet_veremi_v2` nên **giữ** DAGSNet W1, protocol, 50 round, batch 64, mọi xe 1 epoch;
compute đo bằng FLOPs (quy ước `fl_dagsnet_veremi_v2/scripts/cost.py`: FLOPs client / dòng thật / round, FlopCounterMode
matmul/conv). Mục tiêu: giảm FLOPs mà **giữ performance CBB** (non-inferiority, không cần tăng F1).

## Bằng chứng → cơ chế → dấu hiệu bác bỏ

- **Bằng chứng (đo 12/11):** CBB = 6,10×F/dòng thật (không phải 4,00×F như bảng cũ). Dòng thật 2,99×F (= FedAvg),
  64 synthetic/bước fwd+bwd ≈ 3,1×F (gồm floor16, 607.110 bước so 586.405 bước thật). Q, bootstrap, sinh mẫu < 0,01×F.
  Đã đóng: giảm đều 64→48/32 dòng (G1 FAIL low256/absent), bỏ floor (Y50), bỏ synthetic xe lớn (b1), RS (b2).
- **Cơ chế giả thuyết:** synthetic mang tri thức toàn mạng cho lớp vắng chủ yếu qua *ranh giới quyết định* (head);
  backbone học đủ biểu diễn từ dòng thật. Khi đó chặn gradient synthetic vào backbone giữ utility, bỏ ~2/3 FLOPs synthetic.
  SD: tín hiệu synthetic cần nhất ở nửa đầu lịch cosine; nửa sau LR nhỏ.
- **Bác bỏ:** Δmacro âm vượt ngưỡng dưới (luật dưới), đặc biệt nếu lớp vắng/xe nhỏ sụt như G1.

## Arm (method `mobihet_compute`, knob tắt = CBB bit-exact, test `tests/test_mobihet_compute.py`)

| arm | thay đổi | FLOPs/dòng thật (đo, toàn mạng) | byte |
|---|---|---:|---|
| C | CBB (`synth_grad=full`) | 6,10×F (5,98 mỗi bước) | = CBB |
| SH | synthetic: backbone no_grad (BN batch stats của synthetic, không cập nhật buffer), head có gradient; dòng thật BN riêng | **4,16×F** (−32%) | = CBB |
| SM | 16 synthetic (1/lớp) full như CBB cùng batch BN với dòng thật, 48 còn lại head-only | **4,65×F** (−24%) | = CBB |
| SD | 64 synthetic/bước đến nửa lịch, 32 từ round > R/2 (sim: >10; r50: >25) | 5,32×F TB r50 (−13%) | = CBB |

Số đo: `FlopCounterMode` trên `_round_private` thật của từng arm (W1, batch 64, 4 xe × 1.280 dòng, CPU) → hệ số mỗi bước,
nhân tỷ lệ bước toàn mạng (synthetic/dòng thật = 1,040). Script: scratch, kết quả ghi lại ở đây; FLOPs không suy từ `t_train`.

## Luật khóa (mỗi seed, so C cùng seed, scope macro-first: guard phụ report-only)

**Sim fold val 200 xe r20 cosine20, seed 42 và 43** (B20, 8 kernel): PASS nếu Δf1_macro TB r16–r20 ≥ −.0015 **và**
Δf1_macro r20 ≥ −.003 (luật G1/codec). Ứng viên = PASS cả hai seed.
Thang (CONTEXT): SH/SM PASS → pilot full-scale r20 cosine50 seed42 so CBB42 cùng round, cùng luật trên r16–r20 →
PASS → nối r50 + seed43 r50; luật r50: Δmacro TB r46–r50 ≥ −.0015 và r50 ≥ −.003 mỗi seed vs CBB42/CBB43.
SD: pilot r20 cosine50 không chạm điểm đổi (r26) ⇒ SD PASS sim lên thẳng r50 hai seed, cùng luật r50.
Nhiều arm cùng PASS r50 → mỗi arm báo riêng; owner chọn arm rẻ nhất PASS. FAIL → dừng nhánh, không tune/kéo dài.
Báo đủ 10 metric, p10/low256/TD/benign/absent (report-only), byte, FLOPs và `t_train`.

## Kết quả sim r20 (12/11, audit 8/8 PASS, [10 metric/byte](../../runs/sim-kaggle-b20/report-r20.md), [verdict](../../runs/sim-kaggle-b20/verdict-r20.json), [quỹ đạo](../../runs/sim-kaggle-b20/trajectory.md))

| arm | FLOPs | Δmacro r20 s42/s43 | Δ TB r16–20 | luật gốc | quỹ đạo Δmacro | absent | → |
|---|---:|---:|---:|---|---|---:|---|
| SH | 4,16×F | −.0167/−.0160 | −.0162/−.0156 | FAIL | closing/widening | −.209/−.246 | **dừng** |
| SM | 4,65×F | −.0037/−.0022 | −.0038/−.0024 | FAIL | closing cả 2 | −.016/−.031 | pilot r20 (phụ lục, ≥ −.005) |
| SD | 5,32×F | −.0006/−.0011 | −.0008/−.0012 | **PASS** | flat/closing | +.010/−.047 | r50 hai seed |

- **SH bác bỏ cơ chế:** chặn gradient synthetic vào backbone làm sụp lớp vắng (absent −.21/−.25, low256 −.07/−.08,
  TD −.04/−.03) ⇒ tri thức toàn mạng cho lớp vắng cần đi vào **biểu diễn**, không chỉ ranh giới head.
- SM (16 dòng full) giữ phần lớn: macro −.002/−.004, TD −.013/−.008 (widening), p10/low256 trái dấu giữa seed.
- SD: guard TD/absent `widening` nhẹ, report-only. `t_train` sim gần như không đổi (SD 2972 vs C 2970 s;
  SH −9%, SM −3%): ở 200 xe GPU launch-bound, FLOPs giảm không thành wall time.

## Full-scale (12–13/11, thang + phụ lục luật)

Runner `scripts/run_b20_full.py` (SD42/SD43 r50, SM42P pilot r20 cos50) và `scripts/run_b20_sm50.py` (SM42 resume r20→r50, SM43 r50);
cùng base `be3b7d5`, overlay và hp CBB42/43 + knob. Audit: round đủ, finite, skip ≤ 1%, hp = launch, overlay SHA, CM support = CBB
cùng seed, 10 metric từ CM, LR = CBB. Luật r50 (phụ lục §4): macro r50 mỗi seed > .9048.

| arm | bậc | macro | CBB cùng round | Δ end | Δ TB 5 round cuối | slope Δ nửa sau | `t_train` (CBB) | verdict |
|---|---|---:|---:|---:|---:|---:|---|---|
| SM42P | pilot r20 cos50 | .8962 | .9000 | −.0037 | −.0039 | +.00022 (closing) | 3,57 h (3,61) | **PASS_TRAJECTORY** → r50 |
| SD42 / SD43 | r50 | chạy | | | | | | |
| SM42 / SM43 | r50 | chạy | | | | | | |

[audit.json](../../runs/b20-full/audit.json). Full-scale cũng không giảm wall time (−1% với −24% FLOPs): T4 launch-bound như sim.


## Audit bổ sung 13/11 19:15 VN

- SD42: artifact audit PASS, utility PASS; macro 0.90715670, Δ CBB -0.00024206; Δ TB cuối5 -0.00025365; train 8.842 h.
- SD43: artifact audit PASS, utility PASS; macro 0.90807031, Δ CBB -0.00007558; Δ TB cuối5 -0.00012187; train 8.414 h.
- SM42P: artifact audit PASS, utility PASS_TRAJECTORY; macro 0.89624239, Δ CBB -0.00374192; Δ TB cuối5 -0.00387803; train 3.567 h.
- SM42: artifact audit PASS, utility PASS; macro 0.90570180, Δ CBB -0.00169696; Δ TB cuối5 -0.00171599; train 8.714 h.
- SM43: artifact audit PASS, utility PASS; macro 0.90584628, Δ CBB -0.00229960; Δ TB cuối5 -0.00234216; train 8.490 h.

10 metric và checks: `runs/b20-full/audit.json`, `runs/b20-sm50/audit.json`. Audit CM/config/source metadata; chưa replay prediction từ full weights.
