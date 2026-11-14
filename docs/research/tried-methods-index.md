# Index phương pháp đã thử — kiểm 13/11/2026

Mỗi dòng là một study: thử cơ chế gì, kết quả, mở đâu. Bảng theo **họ cơ chế** và lý do đóng ở [sổ tay](method-building-brief.md) §4;
bài học dùng lại ở [LESSONS](../../LESSONS.md). Trước khi đề xuất mới: tìm cơ chế ở đây, không mở lại họ đã đóng bằng tên mới.
Verdict giữ nguyên như lúc đóng; số ở doc gốc. Trạng thái: **nhận** = trong CBB · **đóng** = FAIL/dừng · **chạy** = đang chạy.

## Hiện hành — CBB và sau CBB (v4, DAGSNet W1, 5.343 xe; sim = fold val 200 xe r20)

| study | cơ chế / arm | kết quả | doc |
|---|---|---|---|
| B21 | liều synthetic theo cỡ xe (< 4.096 / 4.096–16.383 / ≥ 16.384 dòng) × pha lịch: T1, T3, T4, T5; T4F = T4 + floor 48 | sim hai seed **chạy** (13/11) | [B21](screening-dose-b21.md) |
| B20 | giảm compute synthetic: SH head-only, SM 16 full + 48 head-only, SD 64→32 dòng từ nửa lịch | SH đóng (absent −.21); SD r50 **PASS hai seed** (−.0002/−.0001, 5,32×F); SM r50 **chạy** | [B20](screening-compute-b20.md) |
| N1 / N2 | proxy server DAGSNet; GBDT liên kết data-space | N1 trùng P20 (không code); N2 trần .8065 < .924 — đóng | [sổ tay §5](method-building-brief.md) |
| A1 / B19 (A2') | nhãn mềm posterior generator theo dòng; cụm biên theo nhầm lẫn generator | FAIL gate local — đóng | [B19](screening-boundary-b19.md), [sổ tay §5](method-building-brief.md) |
| B18 | nguồn teacher ngoài fit: S shadow → C/E/H matched source | H FAIL macro hai seed — đóng | [B18](screening-source-b18.md) |
| B17 | macro-first: P prior-coordinate, NR real-NCKD | FAIL macro hai seed — đóng | [B17](screening-macro-b17.md) |
| B16 | KL-of-mean student: MK, HK half-dose | FAIL hai seed — đóng | [B16](screening-mean-b16.md) |
| B15 | SAM ρ .05 | FAIL hai seed (hại xe nhỏ) — đóng | [B15](screening-sam-b15.md) |
| B14 | uplink: MX interpolation, GP/LG pooling Q | FAIL hai seed — đóng | [B14](screening-uplink-b14.md) |
| B13 | SQ sqrt-count Q, RK/SK/JS divergence, LS smoothing | FAIL hai seed — đóng | [B13](screening-module-b13.md) |
| CBB r50 | Core + KM16 w0 .5 + lite6 + QQ8 | **nhận** (.9078 ± .0004, 3 seed) | [proposal](proposal.md), [codec](screening-codec.md) |
| Codec study / combo | codec gói Q/bootstrap; combo r20/r50 CBB vs CQ8 | lite6 + QQ8 **nhận** | [screening-codec](screening-codec.md) |
| G1 / G2 | giảm synthetic 48/32 dòng/bước; G2 bỏ trước chạy | G1 FAIL low256/absent — đóng | [catalog đề xuất](improvement-proposals.md) |
| b12 | liều floor 24–64, synthetic 80–128 dòng, gộp quyết định theo phiên | không PASS hai seed; gộp phiên owner đóng | [catalog đề xuất](improvement-proposals.md) |
| b11 | EM5/EM8 EMA trọng số, QK Q theo cụm, QT2 temperature, PH prior suy luận | FAIL cả 5 — đóng | [catalog đề xuất](improvement-proposals.md) |
| b9 / b10 / lite6 | biến thể KM: K32, pop, part, EM, whitening, share, temp, gap-fill, Q synthetic, K8/K24, cov | FAIL; lite6 **nhận** | [km-lite](screening-km-lite.md), [generators](generators.md) |
| KM50 | KM16 w0 .5 int8 full-scale r50 | +.0085/+.0086 so Core — **nhận** | [km50](screening-km50.md) |
| b1–b8 | sim: trọng số synthetic, oracle, KM-copula (b3), trộn kiến trúc R1 (b4), w0/K/cov/codec (b5–b8) | KM16 ra đời; R1 W1/W0.25 không sụp | [sim fold val](screening-sim-foldval.md) |
| Y50 | oracle synthetic, prior, no-Q, no-floor trên Core r50 | oracle +.010 trần; prior gain Core không sang CBB | [results-y50](results-y50.md) |
| F50 / L50 | bank input r50 vs Core; lossless CDF | bank lợi r20 mất r50 → chọn Core; L50 đo byte | [F50](results-f50.md), [L50](results-l50.md) |
| Generator mới | pair IPF, direct forest+Gauss, marginal transport, marginal copula, standard latent, forest+quantile | local/pilot20 FAIL — đóng | [generators](generators.md), [archive 07/11](../../archive/docs-20261107/README.md), [archive 09/11](../../archive/docs-20261109/README.md) |

## Trước Core (v4 / v2 recipe, đã archive)

| study | cơ chế | doc |
|---|---|---|
| W20 / X20 / C10 | decoupled KD, step floor, nén bootstrap; combo floor + nén, input-bank, NCKD; nén Core 10 round | [archive 07/11](../../archive/docs-20261107/README.md) |
| R20 / V20 | lặp control, replay tiếp nhận teacher cố định, validation tách thời gian | [R/V](../../archive/docs/research/results-r20-v20.md), [V20](../../archive/docs/research/screening-v20.md) |
| Q20 | streaming, input target, server weight, client reception | [Q20](../../archive/docs/research/results-q20.md) |
| P20 / P21 | nội dung teacher, carrier, proxy bank/stream/hybrid/weights | [P20](../../archive/docs/research/results-p20.md), [P21](../../archive/docs/research/results-p21-recovery.md) |
| O20 | teacher replay | [O20](../../archive/docs/research/results-o20.md) |
| N20 | factorial real-KD × synthetic fallback | [N20](../../archive/docs/research/results-n20.md) |
| M20 | teacher và cá nhân hóa distillation | [M20](../../archive/docs/research/results-m20.md), [cơ chế](../../archive/docs/research/analysis-m20-mechanisms.md) |
| K10 / L10 | kiểm cơ chế federated mới; target, hệ số KD, real gate | [K](../../archive/docs/research/results-k10.md), [L](../../archive/docs/research/results-l10.md) |
| H10 / J10 / R10 (S0–S4) | sàng lọc 10 round: synthetic, teacher theo mẫu, trọng số theo lớp/FLOPs | [H](../../archive/docs/research/screening-h10.md), [J](../../archive/docs/research/screening-j10.md), [S](../../archive/docs/research/screening-r10.md) |
| v1 (66 feature) | NTD, hidden-KD, FedDyn, SCAFFOLD, prox, EMA; synthetic P23–P37 (copula, GMM, MADE, CCVR…) | [kế hoạch v2](../../archive/docs/research/improvement-plan.md), `archive/legacy-v1/` |
