# Index phương pháp đã thử — cập nhật Lite2 17/11/2026

Mỗi dòng là một study: thử cơ chế gì, kết quả, mở đâu. Bảng theo **họ cơ chế** và lý do đóng ở [sổ tay](method-building-brief.md) §4;
bài học dùng lại ở [LESSONS](../../LESSONS.md). Trước khi đề xuất mới: tìm cơ chế ở đây, không mở lại họ đã đóng bằng tên mới.
Verdict giữ nguyên như lúc đóng; số ở doc gốc. Trạng thái: **nhận** = trong T3 (nền CBB) · **đóng** = FAIL/dừng · **chạy** = đang chạy.

## Hiện hành — T3 trên nền CBB (v4, DAGSNet W1, 5.343 xe; sim = fold val 200 xe r20)

| study | cơ chế / arm | kết quả | doc |
|---|---|---|---|
| FB-L (18–19/11) | chọn dòng thật theo CE cũ trong ngân sách FB (top/prop/Q tách), engine `compact` (executed 0,997×/1,001×), sim DAGSNet + BERT seed42 | sim 17/17 audit PASS; **6/6 ứng viên PASS cả hai model**; chọn **P4** (ΔTB vs S0 DAGSNet −.0047 / BERT −.0024; vs nền +.0017 / +.0081), L4 sát (−.0047/−.0026); lợi dồn lớp 13 (BERT +.10). **Pilot full-scale r20 (19/11): 12/12 audit PASS, 6/6 arm PASS cả hai model** (ΔTB vs T3 −.0036…−.0048), chọn **L2** (mean ΔTB −.0038; L4 −.0039, chênh trong nhiễu); r50 chờ owner | [FB-L](screening-fbl.md) |
| FB (Fixed-row-budget) | thay dòng thật xe lớn bằng synthetic, compute ≤1,01×; F0/R48/N1–N4/Z0/ZK, DAGSNet sim seed42 | sim 8/8 audit PASS; N2/N4 PASS, N1/N3 FAIL; chọn N4 (ΔTB −.0064, cost 1,000×); Z0/ZK sụp đổ (−.033/−.028); **đóng:** 8 synthetic M/L đổi lấy dòng thật (N1/N3), bỏ synthetic kể cả + distillation Q trên dòng thật (Z0/ZK); pilot N4 + N2 DAGSNet/BERT r20 đã push 17/11 | [FB](screening-budget.md) |
| Lite2 | bootstrap v2 km2 (K6/K8, 64 bin, 25 % xe cho Lloyd, ô < τ không gửi moment bậc hai, Cholesky 5/6 bit) × liều 8 dòng (D8/U1); gate mới theo đích 16/11 | sim seed 42 r20 (17/11): B/A32/A24, T3+D8, T3+U1, **B+U1**, A32+D8 PASS; B+D8 FAIL (TB −.0051); luật chọn **B+U1**; pilot DAGSNet r20 **PASS** (Δ −.0028, bootstrap 340 663 B) ; full DAGSNet r50 **PASS gate cũ** (.905877; .354116 MB; estimator1,364×), chưa đạt1,01×; pilot BERT r20 **PASS** (Δr20−.004746, ΔTB−.005173); chưa r50 BERT đã audit (17/11) | [Lite2](screening-lite2.md) |
| B21 | liều synthetic theo cỡ xe (< 4.096 / 4.096–16.383 / ≥ 16.384 dòng) × pha lịch: T1, T3, T4, T5; T4F = T4 + floor 48 | sim: T5 FAIL; pilot r20: T4 FAIL; **T3 nhận vào phương pháp chính (owner 15/11)**; r50 hai seed PASS: T1 5,42×F, T3 4,94×F, T4F 5,10×F** (Δmacro +.0002…+.0006; T4F p10 +.012/+.013) | [B21](screening-dose-b21.md) |
| B20 | giảm compute synthetic: SH head-only, SM 16 full + 48 head-only, SD 64→32 dòng từ nửa lịch | SH đóng (absent −.21); SD r50 **PASS hai seed** (−.0002/−.0001, 5,32×F); SM r50 PASS (.9057/.9058, 4,65×F, Δ −.0017/−.0023) | [B20](screening-compute-b20.md) |
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
