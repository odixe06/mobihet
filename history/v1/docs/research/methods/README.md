# Phương pháp theo tên

Mỗi file = một phương pháp: cơ chế, trao đổi (R1/R2), bằng chứng đã audit, trạng thái, nguồn. Owner 17/10.
Ghi chú nghiên cứu theo ngày (nguồn của các file này) ở [archive/research](../../archive/research).
Kết quả/thiết kế từng build ở [docs/methods](../../methods/README.md); protocol ở [protocol](../protocol.md).

Trạng thái hiện hành22/10: [bàn giao sau đủ report/P29–31](../../archive/operations/continuation-2026-10-22.md).
Các mục theo vòng dưới giữ quyết định lịch sử; không dùng chữ "chờ/đang chạy" của vòng cũ làm lệnh launch.

## Vòng 13 — A/B/C + đề xuất owner 17/10 (lịch sử)

Owner 17/10 ~16Z: chạy A, B, C và các đề xuất trong file ưu tiên 17/10 (đã tách thành các file dưới, file gốc đã xóa). Thứ tự
owner: P1 → P2 → P3 → P4 (teacher-only trước) → P5/P6 (chỉ sau teacher feasibility); C1/C2 là điều kiện R1, không tính cải tiến
accuracy. Cổng chất lượng so gpool (+0,010, benign ≥ −0,005, worst ≥ −0,010), codec noninferior; [design §Vòng 13](../../methods/full-drift/design.md).
Ngân sách byte chung: [communication budget](../communication-budget.md).

- [Teacher snapshot gộp cố định (`frz`) + Gaussian KL (`gkd`)](frozen-snapshot-teacher.md) — **PASS ep15 s42**: gkd 0,8030 (+0,069 vs gpool), frz +0,037; luật head chờ owner
- [A — dồn mẫu Gaussian vào lớp hiếm local](gaussian-rare-fill.md) — **FAIL** vòng 13 (−0,001)
- [B1 — bỏ mẫu Gaussian proxy gộp bác bỏ](gaussian-teacher-rejection.md) — s42 PASS +0,0108; s43 +0,0082 FAIL cổng (cùng chiều); **horizon ep15 PASS +0,0113 vs gpool** (test 0,760)
- [C — luật gp theo prior huấn luyện thực (`gpe`)](effective-prior-gp.md) — **FAIL** hai seed (gpe −0,094/−0,095)
- [P1 — teacher gộp dạy private trên mẫu Gaussian](gaussian-teacher-kd.md) — **FAIL** vòng 13 (+0,0001; gate phủ 4,3 % mẫu)
- [P2 — nhiều mode trong lớp (= B2)](gaussian-mixture-modes.md) — không mở (hỗn hợp không tăng nhận mẫu)
- [P3 — tách BN synthetic](synthetic-bn-separation.md) — backlog (BN real-only −0,03/−0,04)
- [P4 — ridge/RFF teacher](ridge-rff-teacher.md) — **FAIL** teacher gate (M256 0,378 ≪ 0,717), như Build13
- [P5 — teacher histogram residual](residual-histogram-teacher.md) — chờ cổng teacher
- [P6 — consensus trên synthetic](synthetic-consensus.md) — chờ cổng teacher
- [C1 — moment53](gaussian-moment53.md) — không train (cùng phân phối với m4b), unit test + byte
- [C2 — q2 (mục q2 trong QProxy)](qproxy-quantization.md) — **noninferior PASS ep6 + ep15** (private −0,0008; proxy −0,118; 219 kB/round)

Nguyên tắc kiểm (đề xuất owner §8): dùng gpool s42/s43 ep6 đã audit làm baseline; số W&B đang chạy hoặc official test không dùng
để chọn hướng. Teacher gate/calibration mới cần tách T/C/E theo receiver/session (nếu baseline cũ đã train vào C mới thì cần
matched baseline); group/old đã dùng nhiều lần là dữ liệu phát triển, ghi số lần chọn. Winner lặp seed 43; ghép tối đa hai winner
bằng 2×2; recipe khóa rồi mới full 50×1 epoch. Báo đủ 10 metric + per-class + benign + worst/p10; chẩn đoán gắn cơ chế (mẫu
real/synthetic từng lớp, KL real/synthetic, gate acceptance, BN moments, byte theo loại tri thức). R2: kịch bản đầu mọi private
DAGSNet; sau sàng lọc mới [mixed architectures](mixed-architectures-r2.md); chưa kiểm thì chỉ ghi "R2 tương thích theo thiết kế".
Không chọn (đề xuất owner §10): FedGen nguyên bản (cần latent tương thích; teacher yếu cũ chưa giúp backbone), parameter-sharing
personalization (phá R2), temporal feature mới, đổi dataset, slim model, tăng lần gộp trong epoch.

## Synthetic/CCVR — đã nối nghiên cứu, chốt lại để compact22/10

- [Bàn giao hiện hành](../../operations/continue.md) — final6×50 đủ report; P29–36 xong; vòng 26 online (cvp + cop) FAIL (Δ +0,0003).
- [P37: synthetic đổi trọng số private/proxy trên full model + tài liệu tham khảo](synthetic-weight-dynamics-2026-10-23.md) — không ứng viên;
  oracle dòng thật +0,010 = cổng online; BN/head không phải nút thắt.
- [P36: teacher × support so phương pháp hiện hành, hai seed; GMM lớp tấn công](teacher-support-two-seed-2026-10-22.md) — Project48 + copula PASS;
  GMM FAIL.
- [P35: hoán đổi lớp, độ chịu lệch, trần năng lực GMM/MADE](support-capacity-oracle-2026-10-22.md) — lợi cần mẫu thật của lớp tấn công hiếm;
  bộ sinh học được fit trên pool +0,0044 tối đa, không đủ năng lực.
- [P34: bộ sinh giữ atom/trạng thái first/định danh extractor, port V4-S/G61](structured-support-generator-2026-10-22.md) — **FAIL** +0,0023;
  trần là phụ thuộc Gaussian của copula.
- [P33: client học gì từ support (cặp lỗi, hình học đặc trưng private, own_bal)](client-learning-diagnosis-2026-10-22.md) — lợi support thật =
  recall posMirror + suddenStop từ biến thể client khác.
- [P32: copula × Project48, ablation nhóm feature](synthetic-support-realism-2026-10-22.md) — copula FAIL; 4 nhóm cần cùng thật.
- [P29–31: gradient/support, joint carrier, teacher-nullspace](synthetic-gradient-diagnosis-2026-10-22.md) — P30 utility FAIL;
  P31 chưa chứng minh mất chiều là nguyên nhân chính. Không tune/launch lại adaptation FAIL.
- [Nhận diện update, primary sources, conditional input density §7](synthetic-transfer-identifiability-2026-10-22.md) —
  hướng mới chỉ proposal, chưa code/train; atom/dequantization và contact budget chưa chốt.
- [Synthetic recovery, Probe23–28](synthetic-learning-recovery-2026-10-22.md) — Project48 teacher PASS hai seed Kaggle,
  private transfer/inversion FAIL; class-weighted/real-support oracle/gradient checks đã lưu, chưa online FL.
- [Rà build cũ](synthetic-build-history-audit-2026-10-22.md) — controls, duplicate/veto, scope lịch sử chưa đủ artifact.
- [Nguồn primary, CCVR nén và client/r50](ccvr-budget-primary-sources-2026-10-22.md) — loss/estimator/codec, giới hạn suy luận.

## Vòng 24–25 — sinh mẫu có chủ đích (owner 21/10)

- [Hiệu chỉnh teacher bằng đặc trưng ảo theo lớp (`cv`)](teacher-virtual-feature-calibration.md) — **cvl FAIL** ep15: private−0,0136,
  benign−0,0185; full covariance bị ngân sách loại. Dừng adaptation covariance chung, run cuối giữ phương pháp chốt.
- [Lai ghép dòng own theo cặp lớp bất đồng (`xo`)](own-row-crossover-priority.md) — probe 20: Gaussian bất đồng sai chỗ (cos 0,09), lai ghép ưu tiên cos 0,55; **FAIL** s42 ep15 Δ +0,0076 vs xa (cần +0,010)

## Đang chạy / nền

- [DML — private ↔ proxy chung học tương hỗ](dml-proxy-mutual-learning.md) — Nền; headline DML + m4b; lợi FL +0,04 hai seed
- [m4b — đặt 0 khối 13 feature receiver/vị trí](receiver-mask-m4b.md) — PASS seed 42 + 43, horizon ep15 PASS; headline
- [Gaussian từ moment lớp gộp](gaussian-class-moments.md) — Có lợi; m4b PASS hai seed; pooled > own +0,05

## PASS, dùng ghép

- [Hybrid private-only — Gaussian dạy riêng private trên nền DML-m4b](hybrid-private-only-synthetic.md) — **PASS 3 cổng × 2 seed** (+0,016/+0,024 so m4b); **headline client** (owner 17/10)
- [swappos — tráo khối receiver + đặt 0 vị trí](receiver-swap-swappos.md) — PASS s42 + s43 (DML); Gaussian full +0,052
- [QProxy — lượng tử update proxy](qproxy-quantization.md) — **q3_zl chốt 20/10**, noninferior hai seed trên la_kdpx; byte gói thật, contact chưa kiểm

## Giai đoạn B / R2

- [Mixed architectures — kiểm R2 thật](mixed-architectures-r2.md) — Kế hoạch sau sàng lọc DAGSNet
- [FedBN trên proxy — probe chẩn đoán](fedbn-proxy.md) — Chưa chạy; điều kiện

## Chưa chạy

- [Nhánh B personalized — APFL/FedAPEN, FedRoD, kNN-Per, FedMRL](personalized-branch-b.md) — Chưa chạy; cần holdout theo client
- [Prior-consistent DML — logit adjustment trong loss (FedLC)](prior-consistent-dml.md) — Tạm hoãn
- [NuisanceCov — giảm covariance margin–receiver](nuisance-cov.md) — Đề xuất 15/10; chưa chạy
- [ConflictGraph — chia sẻ lớp hay xung đột khi cập nhật](conflict-graph.md) — Đề xuất 15/10; chưa chạy
- [InputSensitivity — consensus gradient input-margin](input-sensitivity.md) — Dự báo thấp; backlog
- [ErrorCosts — chi phí lỗi theo cặp lớp](error-costs.md) — Dự báo thấp riêng lẻ; adjunct
- [SAM — recipe dự phòng](sam-recipe.md) — Không thêm tri thức FL; dự phòng
- [Họ prototype/header chung — FedProto, FedTGP, FedORGP, FedGH](prototype-headers.md) — Ưu tiên thấp

## FAIL / loại

- [KD cap + frozen teacher — cách private hấp thụ teacher](kd-cap-frozen-teacher.md) — FAIL hai seed (cap −0,03 private, +0,03 proxy; frz gp sập, head cao)
- [FedDyn trên proxy — sửa drift bằng lịch sử](feddyn-proxy.md) — FAIL hai seed (−0,03, client tệ nhất −0,15)
- [SCAFFOLD trên proxy — control variate sửa hướng cập nhật](scaffold-proxy.md) — FAIL (LR* 0,1: −0,010 so sgd, −0,007 so m4b)
- [Response consistency — hai view receiver + JS](response-consistency.md) — FAIL (JS +0,003)
- [Private EMA + BN tính lại](private-ema.md) — Có lợi cho Gaussian không mask; hại DML và Gaussian-m4b
- [FedProx / FedAvgM trên proxy](fedprox-fedavgm.md) — FAIL (không nối)
- [Đổi kênh KD — FedNTD, hidden-state KD (FedKD)](fedntd-fedkd-hidden.md) — FAIL (không quét tiếp)
- [ProxyFuse — trộn private gp với proxy khi triển khai](proxy-fusion.md) — FAIL (−0,032/−0,034 so proxy)
- [ClassTail — CVaR theo lớp trên loss hard-label local](class-tail-risk.md) — FAIL (không nối)
- [CrossClientRank — score moments cho pairwise ranking](cross-client-ranking.md) — FAIL seed 42 (không nối)
- [BoundaryCells — đếm nhãn theo ô input](boundary-cell-counts.md) — FAIL bốn adaptation (dừng)
- [CountSketch EF sửa overshoot](countsketch-ef-repair.md) — FAIL (không nối)
- [Họ chia sẻ tham số — Ditto, FedALA, FedBABU, FedRep](parameter-sharing-personalization.md) — Loại (phá R2)
- [Slim proxy — proxy DAGSNet ×0,3 kênh](slim-proxy.md) — Owner 17/10: không chạy (bỏ hướng giảm độ phức tạp model)
