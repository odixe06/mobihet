# FB — ngân sách dòng cố định: synthetic thay dòng thật ở xe lớn, compute ≤ 1,01× FedAvg (khóa 17/11/2026, trước push)

Nền = [Lite2](screening-lite2.md) **B + U1** (bootstrap km2 B 340 663 B/xe full-scale, compute 1,365×). Yêu cầu: [task brief](../../task_brief.md) §1/§14.
Owner chốt 17/11: (1) cho phép nới "mỗi xe 1 epoch dòng thật/round" ở xe lớn, hoặc bỏ hẳn synthetic và dựng cơ chế distillation, thử thêm nếu cần;
(2) compute ≤ 1,01× FedAvg là đạt (nới từ 1,000×); (3) sim DAGSNet → pilot DAGSNet + BERT; (4) push sim ngay sau preflight PASS, PASS tự lên bậc kế.
Giả định của người viết (chưa bị owner bác): compute tính **trung bình toàn mạng**, không ép từng xe ≤ 1×.

## Bằng chứng → cơ chế → lợi kỳ vọng → dấu hiệu bác bỏ

- **Bằng chứng:** 1 epoch dòng thật đã = 1,000× FedAvg ⇒ mọi dòng synthetic thêm vào đều > 1×; cắt liều không về được 1×
  (Lite2 1,365×; riêng tầng S + floor đã +0,113). Bỏ synthetic theo lớp ở xe đủ dòng hại đơn điệu (b1 A256 −.030 … A4096 −.0015),
  head-only −.016 (SH), mixup thật/synthetic −.0158 (MX) ⇒ giữ dòng synthetic đi qua cả model.
  Tầng M/L (≥ 4.096 dòng) giữ 88,7 % dòng thật; mất macro khi cắt liều nằm ở xe nhỏ ([B21](screening-dose-b21.md)); precision train ≈ .99.
- **Cơ chế:** xe tầng M/L giữ số bước `ceil(n_k/64)` và slot 64 dòng; mỗi bước chỉ train `r` dòng thật đầu slot + `s` synthetic (`r + s ≤ 64`).
  Hoán vị mỗi round của `core.local` làm `r` dòng đó thành tập con ngẫu nhiên mới mỗi round. Dòng thật ngoài ngân sách: không forward, không BN,
  không CE, không góp Q round đó. Packet Q của xe = trung bình trên dòng đã train (trọng số vẫn `n_kc`). Tầng S giữ đủ epoch thật + liều + floor 16.
  Loss, Q, LA, lịch LR, bootstrap B, message và byte không đổi; knob chỉ đếm dòng ⇒ không phụ thuộc kiến trúc.
- **Khác study đã đóng:** G1/T5/D8/U1 chỉ cắt synthetic và giữ đủ dòng thật; A256–A4096 bỏ synthetic theo lớp (lệch cân bằng lớp); MX nội suy dòng.
  FB đổi **dòng thật** lấy synthetic, giữ synthetic cân bằng 16 lớp; gradient kỳ vọng mỗi bước bằng B + U1, chỉ phương sai minibatch tăng (≤ √(64/r)).
  b1 từng nêu "đổi tỷ lệ thật/synthetic ở xe lớn" là hướng chưa thử.
- **Lợi kỳ vọng:** compute 1,365× → ≈ 1,00×; byte = Lite2 (≈ 0,357 MB/50r); macro giảm trong ngân sách gate. Không kỳ vọng tăng macro.
- **Bác bỏ:** không arm ứng viên nào PASS luật sim; hoặc R48 làm tầng L sụt > .003 so B + U1 (thay dòng thật tự nó hại);
  hoặc pilot/full trượt luật dưới. FAIL → dừng arm, báo owner, không tune/kéo dài.

## Cách đếm compute

`scripts/run_sim_budget.py cost`: (dòng thật đã train + dòng synthetic đã train, gồm bước floor) / dòng thật, cỡ xe full-scale
(`cache/train_sizes.npy`), nửa lịch ở liều sau mốc; FedAvg = 1 = 13,297 MFLOP (DAGSNet) / 40,022 (BERT) mỗi dòng thật/round.
Cùng quy ước useful-work của T3/Lite2 nhưng chia cho dòng thật chính xác (số cũ chia cho bước × 64: T3 1,652 ↔ 1,655; B + U1 1,364 ↔ 1,365).
Phần dòng thật được đo trong run (`real_frac` ở `rounds.jsonl`). Engine vẫn đệm/che dòng ngoài ngân sách ⇒ `t_train` không theo FLOPs;
FlopCounter đo thật cần rút shape batch, chỉ làm sau khi utility PASS.
Phân bố theo xe (N1/N2): tầng S trung vị 2,05×, M/L 0,875×; 28,5 % xe ≤ 1,01×.

## Arm — sim fold val 200 xe, r20 cosine20, DAGSNet seed 42 (`scripts/run_sim_budget.py`, 8 kernel, CAP 2,75 h)

hp nền = Lite2 S7 (B + U1, `synth_late_from=10`, floor 16); method `mobihet_budget` (knob tắt = `mobihet_lite` bit-exact);
mỗi arm chỉ đổi `synth_tier_rows` (S/M/L, đầu lịch/sau mốc) và `real_tier_rows` (dòng thật/bước S/M/L).

| arm | synthetic S / M / L | dòng thật/bước | compute | DAGSNet / BERT MFLOP | vai trò |
|---|---|---|---:|---:|---|
| S0 | T3 64,64 / 64,32 / 48,16 | 64 / 64 / 64 | 1,655× | 22,00 / 66,23 | control gate (Lite2 sim, đã audit; bootstrap T3) |
| S7 | U1 64,32 / 32,16 / 16,16 | 64 / 64 / 64 | 1,365× | 18,15 / 54,64 | nền B + U1 (Lite2 sim, đã audit) |
| F0 | U1 | 64 / 64 / 64 | 1,365× | 18,15 / 54,64 | chạy lại S7 qua code mới: nhiễu giữa hai lần chạy |
| R48 | U1 | 64 / 48 / 48 | 1,144× | 15,21 / 45,77 | cô lập: chỉ bớt dòng thật M/L (không lên bậc) |
| N1 | 64,32 / 8,8 / 8,8 | 64 / 48 / 48 | 1,003× | 13,34 / 40,15 | ứng viên |
| N2 | 64,32 / 16,16 / 16,16 | 64 / 40 / 40 | 1,003× | 13,34 / 40,16 | ứng viên |
| N3 | 64,64 / 8,8 / 8,8 | 64 / 45 / 45 | 0,999× | 13,29 / 40,00 | ứng viên (tầng S giữ liều T3) |
| N4 | 64,64 / 16,16 / 16,16 | 64 / 37 / 37 | 1,000× | 13,29 / 40,01 | ứng viên (tầng S giữ liều T3) |
| Z0 | không train synthetic, floor 0 | 64 / 64 / 64 | 1,000× | 13,30 / 40,02 | tham chiếu "bỏ synthetic": CE + LA trên dòng thật |
| ZK | như Z0 + `KL(Q_prev[y] ‖ p)` trên dòng thật | 64 / 64 / 64 | 1,000× | 13,30 / 40,02 | tham chiếu "bỏ synthetic + distillation lớp-Q" |

Z0/ZK vẫn chạy bootstrap B (byte của arm không đại diện); nếu bỏ hẳn synthetic thì message chỉ còn Q (≈ 15 KB/50r).
ZK thuộc họ realKD đã đóng **khi có synthetic**; ở đây chỉ đo mức sàn khi không có synthetic theo gợi ý của owner, không lên bậc tự động.

## Luật khóa (macro-first; guard p10/low256/TD/benign/absent/tầng cỡ xe report-only)

- **Sim** (artifact audit PASS trước; so S0 cùng round): ứng viên PASS nếu Δf1_macro TB r16–20 ≥ −.007 **và** Δf1_macro r20 ≥ −.008
  **và** compute ≤ 1,01×. Nhiều arm PASS → chọn macro TB r16–20 cao nhất. F0/R48/Z0/ZK chỉ báo cáo.
  Z0/ZK đạt ngưỡng macro → báo owner như phương án riêng (cần dựng bản không bootstrap), không tự lên bậc. Một seed: screening, không là significance.
- **Pilot** full-scale 5.343 xe, r1–20 cosine50, DAGSNet + BERT, so T3 seed 42 cùng round: Δmacro DAGSNet ≥ −.006, BERT ≥ −.010, áp cho cả TB r16–20
  và r20; byte bootstrap thật ≤ 382 000 B/xe; `real_frac` đo = estimator. Port BERT = copy byte-identical ba file method.
- **Full r50** (nối checkpoint, cosine50): DAGSNet > .8998, BERT ≥ .8951; tổng ≤ 0,40 MB/xe/50r; compute ≤ 1,01× FedAvg; báo hạng performance/communication/compute.
- PASS tự lên bậc kế; FAIL dừng arm. Mixed DAGSNet + BERT chỉ sau khi đơn model PASS. Báo đủ 10 metric, per-class, cohort cỡ xe, byte, FLOPs, `t_train`.

## Kiểm trước push

- `tests/test_mobihet_budget.py` 6/6: knob tắt và `real_tier_rows` toàn 64 = Lite2 bit-exact (state + info 2 round); loss/gradient khớp tham chiếu
  tường minh, dòng ngoài ngân sách không ảnh hưởng gradient; packet Q = trung bình trên dòng đã train; `real_frac` khớp công thức.
- Smoke local 200 xe 2 round compiled (`runs/sim-kaggle-fb/smoke.sh`, không là kết quả): N2 compile gate PASS, skip 0,
  `real_frac` .663686 = estimator panel, mọi xe có applied step.

## Kết quả

Đã xác nhận đủ8kernel version1 ngày17/11; chưa kết quả artifact/verdict sim. [Receipt launch](../operations/fb-launch-20261117.json). Job/kernel: [continue](../operations/continue.md).
