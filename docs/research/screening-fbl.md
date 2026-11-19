# FB-L — chọn dòng thật theo loss cũ trong ngân sách FB (khóa 18/11/2026, trước push)

Nền = [FB](screening-budget.md) N4 / N2 (compute 1,000× / 1,003× FedAvg). Owner 18/11: thử hướng mới trước, **tạm không nối r50** N2/N4;
phương pháp đề xuất phải đạt trên **mọi loại model** (DAGSNet và FeatureBERT); thử mọi arm khả thi miễn giữ yêu cầu performance /
communication / compute; sim cần **hơn N4 ghép cặp ≥ +.001 TB**; dựng harness đo compute chung song song ([compute-harness](compute-harness.md)).

## Bằng chứng → cơ chế → lợi kỳ vọng → dấu hiệu bác bỏ

- **Bằng chứng** (pilot FB r20, [LESSONS](../../LESSONS.md) mục FB pilot): gap vs T3 là trễ hội tụ — arm ở round r ≈ T3 ở round c·r,
  c ≈ `real_frac` (.61–.65). Mất ở recall (recall macro −.010, precision DAGSNet +.002), dồn vào lớp học chậm: 13 timeDelayAttack
  (BERT −.075/−.089, DAGSNet −.021/−.025), 11 suddenConstantSpeed (−.017…−.020), 1 accelerationMultiplication (−.014/−.018).
  BERT N2/N4 trượt gate pilot (ΔTB −.0104/−.0116 < −.010); bỏ lớp 13 thì ΔTB BERT −.0058/−.0063.
- **Cơ chế:** FB train r dòng thật đầu slot 64 dòng (tập con ngẫu nhiên mỗi round). FB-L giữ nguyên slot, số bước, số dòng r và synthetic,
  chỉ đổi **dòng nào** trong slot được train: mỗi dòng thật lưu CE của lần train gần nhất (chưa train: +∞, ưu tiên tuyệt đối).
  `top`: r dòng CE cũ cao nhất; `prop`: lấy không hoàn lại xác suất ∝ CE cũ (Gumbel-top-k, generator riêng ⇒ hoán vị `core.local`
  trùng FB). CE cũ luôn cao hơn CE hiện tại (model đã tốt lên) nên dòng bị bỏ qua tự lên hạng: không đói dòng.
  `real_select_q = q > 0`: q dòng đầu slot giữ tập ngẫu nhiên của FB và là dòng duy nhất góp packet Q (Q không lệch về dòng khó);
  r − q dòng còn lại chọn theo CE cũ, chỉ góp CE. Tầng S (64 dòng/bước) train đủ slot ⇒ không đổi.
- **Chi phí:** cùng số dòng qua model với nền ⇒ compute estimator = N4/N2; CE theo dòng có sẵn trong loss (0 pass thêm);
  thêm top-k trên 64 phần tử/bước và 4 B/dòng thật bộ nhớ trên xe (lưu trong state để resume). Byte, message, bootstrap không đổi.
- **Khác study đã đóng:** G1/T5/D8/U1 cắt synthetic; A256–A4096 bỏ synthetic theo lớp; B13/B17 đổi loss/teacher; SAM (B15) đổi optimizer.
  Chưa study nào chọn dòng thật theo loss ([tried index](tried-methods-index.md)). Nguồn: [papers](../../papers/README.md)
  (online batch selection, importance sampling, Selective Backprop, Gumbel-top-k; RHO-LOSS là cảnh báo nhiễu nhãn).
- **Lợi kỳ vọng:** thu lại +.002…+.004 macro so nền ở cùng compute, chủ yếu recall lớp 13/11/1. Không kỳ vọng vượt T3.
- **Bác bỏ:** không arm nào PASS cả hai model theo luật dưới; hoặc arm thắng nền nhưng lớp 13/11/1 không tăng (thắng vì lý do khác,
  báo owner, không lên bậc); hoặc precision macro/benign tụt làm macro giảm. FAIL ⇒ đóng họ "chọn dòng thật theo loss", không tune thêm.
- **Rủi ro:** bám dòng nhiễu nhãn; prior CE lệch so với logit-adjust (`la_shift` theo prior thật); Q mềm hơn khi q = 0.

## Arm — sim fold val 200 xe, r20 cosine20, seed 42, **DAGSNet và FeatureBERT** (cùng hp, cùng 200 xe)

hp = hp sim FB của nền (Lite2 B, `synth_late_from=10`, floor 16) + knob FB-L + **`compact=true`** (owner 19/11: engine thực thi
≈ 1,00× FedAvg, cùng hàm với engine mask; [compute harness](compute-harness.md)); method `mobihet_budget` (knob tắt = FB bit-exact,
`tests/test_mobihet_budget.py`). Ba file method byte-identical giữa hai repo (`a4147a1b…`).

| arm | nền | `real_select` | `real_select_q` | useful | executed | vai trò |
|---|---|---|---:|---:|---:|---|
| N4c | N4 (64/37/37, synthetic 64,64/16,16/16,16) | random | 0 | 0,9996× | 0,997× | nền ghép cặp, compact |
| N2c | N2 (64/40/40, synthetic 64,32/16,16/16,16) | random | 0 | 1,003× | 1,001× | nền ghép cặp, compact |
| L4 | N4 | top | 0 | 0,9996× | 0,997× | ứng viên |
| P4 | N4 | prop | 0 | 0,9996× | 0,997× | ứng viên |
| Q4 | N4 | top | 18 | 0,9996× | 0,997× | ứng viên (Q không lệch) |
| L2 | N2 | top | 0 | 1,003× | 1,001× | ứng viên |
| P2 | N2 | prop | 0 | 1,003× | 1,001× | ứng viên |
| Q2 | N2 | top | 20 | 1,003× | 1,001× | ứng viên (Q không lệch) |

Control (owner 19/11): S0 DAGSNet = arm Lite2 sim đã audit; nền ghép cặp N4c/N2c chạy lại **cùng engine compact** (không dùng N4/N2 mask cũ:
khác luồng dropout so ứng viên). FeatureBERT: S0B (T3, hp S0 Lite2 sim, mask — T3 không có knob compact) + N4cB, N2cB + 6 ứng viên,
tất cả cùng hp DAGSNet. 8 kernel DAGSNet + 9 BERT. Nhiễu chạy lại DAGSNet F0−S7 ≈ .0003. N4/N2 mask cũ giữ verdict, không hồi tố.

## Luật khóa (macro-first; per-class/p10/benign/tầng cỡ xe report-only)

- Artifact audit PASS trước. Mỗi model so control **cùng model** cùng round.
- Ứng viên PASS một model nếu: Δmacro TB r16–20 vs S0 ≥ −.007 **và** Δ r20 vs S0 ≥ −.008 **và** compute ≤ 1,01× trên **cả hai thước**
  useful (estimator) và executed (`compute_executed.py` full-scale compact, đã đối chiếu số đếm engine; thêm 19/11 trước push)
  **và** Δmacro TB r16–20 vs nền ghép cặp (N4c hoặc N2c cùng model) ≥ +.001.
- Arm PASS **cả DAGSNet và FeatureBERT** mới đủ điều kiện. Nhiều arm ⇒ chọn trung bình hai model của Δ TB vs S0 cao nhất.
- Không arm nào ⇒ FAIL, đóng FB-L, báo owner. Một seed: screening, không là significance.
- Bậc kế (sau khi báo owner): pilot full-scale arm chọn, r1–20 cosine50, cả hai model, luật pilot FB (Δ vs T3 ≥ −.006 DAGSNet,
  ≥ −.010 BERT, cả TB r16–20 và r20). **Không tự nối r50** (owner 18/11).

## Kiểm trước push

- `tests/test_mobihet_budget.py` 14/14: knob tắt = Lite2/FB bit-exact; top giữ đúng r dòng CE cao, loss = FB trên slot xếp lại;
  dòng chưa train đi trước, đồng hạng giữ thứ tự slot ngẫu nhiên; prop không động tới generator hoán vị; Q chỉ từ q dòng ngẫu nhiên;
  state mang `row_score`; một round cùng `real_frac` và cùng trạng thái generator với FB.
- Smoke local 200 xe 2 round compiled (`runs/sim-kaggle-fbl/smoke.sh`, không là kết quả).
- 19/11: 19/19 với `compact` (bước-so-bước fp64 `random`/`top`/`prop`/`top`+q; `random` không floor trùng state cả round);
  đo engine thật compact = 0,9155× FedAvg executed trên panel, trùng harness ⇒ full-scale 0,997× (N4) / 1,001× (N2)
  ([compute harness](compute-harness.md)). Method byte-identical hai repo `a4147a1b…`.

## Trạng thái 19/11

`compact` xong và kiểm (không có bug: lệch 18/11 là lật kink ReLU ở fp32). Owner 19/11 chốt: mọi arm compact, nền N4c/N2c cùng engine,
gate executed ≤ 1,01×, push batch. Job/kernel: [continue](../operations/continue.md).

## Trạng thái 18/11 (dừng phiên)

Chưa push kernel nào. Code FB-L + test 14/14 + smoke DAGSNet/BERT PASS. Owner hỏi compute thật ⇒ phát hiện engine thực thi FB ≈ 2,07× FedAvg
([compute harness](compute-harness.md)); đề xuất (chưa duyệt) chạy mọi arm với `compact=true`, rerun control N4c/N2c cùng engine và
thêm gate executed ≤ 1,01×. `compact` còn lệch gradient ở bước floor ⇒ sửa trước. Bước tiếp: [continue](../operations/continue.md).

## Kết quả — verdict r20 (19/11 ≈ 07:xx; `runs/sim-kaggle-fbl/verdict-r20.json`, `FBL_DONE`)

Audit artifact 17/17 PASS (DAGSNet 8, BERT 9). Nền compact tái lập nền mask DAGSNet (N4c TB .9070 = N4, N2c .9071 = N2).
Control: DAGSNet S0 r20 .9138 / TB .9134; BERT S0B .9053 / .9044, N4cB .8944 / .8939, N2cB .8936 / .8933 (nền BERT −.011 vs S0, như pilot).

| arm | DAGSNet ΔTB vs S0 / vs nền | BERT ΔTB vs S0 / vs nền | Δr20 vs S0 D / B | ΔTB vs S0 TB 2 model | lớp 13 r20 vs nền D / B | benign D / B | verdict |
|---|---|---|---|---:|---|---|---|
| **P4** | −.0047 / +.0017 | −.0024 / +.0081 | −.0050 / −.0021 | **−.00353** | +.008 / +.102 | −.002 / +.004 | **PASS, chọn** |
| L4 | −.0047 / +.0017 | −.0026 / +.0079 | −.0046 / −.0025 | −.00365 | +.017 / +.102 | −.001 / +.003 | PASS |
| L2 | −.0050 / +.0014 | −.0030 / +.0082 | −.0048 / −.0034 | −.00398 | +.017 / +.098 | −.000 / +.003 | PASS |
| Q2 | −.0053 / +.0011 | −.0028 / +.0084 | −.0055 / −.0027 | −.00404 | +.010 / +.110 | −.001 / +.004 | PASS |
| P2 | −.0049 / +.0015 | −.0041 / +.0071 | −.0053 / −.0042 | −.00446 | +.013 / +.095 | −.002 / +.003 | PASS |
| Q4 | −.0053 / +.0011 | −.0045 / +.0060 | −.0055 / −.0044 | −.00490 | +.006 / +.068 | −.001 / +.004 | PASS |

Compute: useful 0,9996× / 1,003×, executed 0,997× / 1,001× (họ N4 / N2). `t_train` r1–20 DAGSNet 0,84–0,87 h (S0 0,88 h), BERT 1,23–1,49 h (S0B 1,37 h).

- Cơ chế đúng dấu hiệu khóa trước: lợi vs nền dồn vào lớp 13 timeDelayAttack (BERT +.07…+.11, DAGSNet +.006…+.017), 11, 1;
  precision/benign không tụt làm macro giảm (benign BERT tăng). Không chạm điều kiện bác bỏ.
- P4 vs L4 chênh .00012 trung bình hai model — trong nhiễu một seed (≈ .0003); luật chọn theo số ⇒ P4. Q (q > 0) không giúp.
- Một seed, screening; BERT thu lại phần lớn gap pilot (nền −.011 → −.002…−.005 vs S0).
- Bậc kế theo luật: pilot full-scale arm chọn, r1–20 cosine50, hai model, luật pilot FB; **chờ owner** (không tự nối r50).

## Pilot full-scale — mọi arm PASS sim (owner 19/11: "chạy toàn bộ các phương án pass để tìm ra phương pháp tối ưu nhất")

Luật khóa trước push (= luật pilot FB, không đổi): mỗi arm × model, 5.343 xe, seed 42, r1–20 của lịch cosine50, `compact=true`, hp = Lite2 pilot +
knob FB-L của arm sim; control T3 seed 42 cùng model. PASS một model: Δmacro TB r16–20 và Δ r20 vs T3 ≥ −.006 (DAGSNet) / ≥ −.010 (BERT),
bootstrap ≤ 382 000 B/xe, `real_frac` mỗi round = estimator full-scale, audit artifact PASS. **Arm tối ưu:** PASS cả hai model, rồi
trung bình hai model của ΔTB vs T3 cao nhất; báo kèm executed compute (0,997×/1,001×), byte, p10, lớp 13/11/1. Không tự nối r50.
N4/N2 pilot mask (18/11) = tham chiếu report-only. Launcher `scripts/run_fbl_full.py`, `../fl_veremi_bert/scripts/run_fbl_pilot.py`.
Push 19/11 ≈ 07:00–07:30: DAGSNet 6/6 (CAP 5,0 h), BERT P4B (`outlookcheck`, owner cho phép 19/11) + L4B (`novahihi`), CAP 7,5 h.
L2B/Q2B/P2B/Q4B chờ quota (cần ≥ 8,63 h/account rảnh; reset 21/11 00:00 +07). Kernel/owner: [continue](../operations/continue.md).

### Ghi nhận một phần (06:20, trước khi đủ BERT)

Audit artifact (`scripts/audit_sim_batch.py fbl|fbl-bert 20`): DAGSNet 8/8 PASS; BERT L2B, P2B PASS (7 arm còn RUNNING).
Nền compact tái lập nền mask: N4c TB .9070 (N4 mask .9070), N2c .9071 (N2 .9071) ⇒ đổi engine không đổi kết quả.

| DAGSNet | r20 | TB r16–20 | ΔTB vs S0 | Δr20 vs S0 | ΔTB vs nền | executed | Δ lớp 13 / 11 / 1 / benign r20 vs nền | phía DAGSNet |
|---|---:|---:|---:|---:|---:|---:|---|---|
| L4 | .9093 | .9087 | −.0047 | −.0046 | **+.0017** | 0,997 | +.017 / +.013 / +.002 / −.001 | PASS |
| P4 | .9089 | .9087 | −.0047 | −.0050 | +.0017 | 0,997 | +.008 / +.001 / +.010 / −.002 | PASS |
| Q4 | .9084 | .9081 | −.0053 | −.0055 | +.0011 | 0,997 | +.006 / +.006 / +.001 / −.001 | PASS |
| L2 | .9090 | .9084 | −.0050 | −.0048 | +.0014 | 1,001 | +.017 / +.007 / +.005 / −.000 | PASS |
| P2 | .9085 | .9085 | −.0049 | −.0053 | +.0015 | 1,001 | +.013 / +.001 / +.008 / −.002 | PASS |
| Q2 | .9083 | .9081 | −.0053 | −.0055 | +.0011 | 1,001 | +.010 / +.002 / +.007 / −.001 | PASS |

S0 r20 .9138 / TB .9134. Mọi ứng viên DAGSNet qua phía DAGSNet; lợi vs nền +.0011…+.0017 (nhiễu chạy lại ≈ .0003, một seed),
dồn vào lớp 13/11/1 đúng dấu hiệu cơ chế, benign −.000…−.002 (report-only). BERT r20: L2B .9020 (TB .9015), P2B .9011 (TB .9004);
chưa có S0B/N2cB nên chưa tính Δ. 10 metric r20 tính từ CM pooled (audit), lưu ở log phiên; verdict cuối: `verdict-r20.json` khi watcher xong.

### Kết quả pilot full-scale r20 (19/11 ≈ 18:40; `runs/fbl-full/audit.json`, `../fl_veremi_bert/runs/fbl-pilot/<ARM>/audit.json`)

12/12 kernel COMPLETE, pull đủ (song song, ≈ 2 phút). Đủ r1–20: 6 DAGSNet + BERT L4/P4/L2/P2/Q2. **Q4B dừng r19** (CAP 3,68 h,
`budget stop before round 20`) ⇒ nối session 2 `outlookcheck/flb-fbl-q4-top-q18-s42-r20-s2b` (resume r19 → r20; `-p2` 409, `-s2` lỗi glob resume, xem
[bài học vận hành](../operations/lessons.md)). Audit artifact (source/overlay SHA, HP, round 1..20, skip ≤ 1 %, client applied, 10 metric từ CM,
real_frac = estimator, bootstrap 340 663 B ≤ 382 000 B) **12/12 PASS** (Q4B qua `-s2b`: r1–19 trùng byte session 1, config trùng, `resumed rank0 at round 20`). Control T3 seed 42: DAGSNet r20 .8995, BERT r20 .9029.

| arm | DAGSNet r20 | ΔTB / Δr20 / Δp10 | BERT r20 | ΔTB / Δr20 / Δp10 | mean ΔTB | verdict |
|---|---:|---|---:|---|---:|---|
| L2 | .8958 | −.0037 / −.0037 / −.0080 | .8992 | −.0039 / −.0037 / −.0013 | **−.00380** | PASS cả hai |
| L4 | .8957 | −.0038 / −.0038 / −.0028 | .8993 | −.0041 / −.0036 / −.0038 | −.00394 | PASS cả hai |
| P2 | .8961 | −.0036 / −.0034 / −.0040 | .8990 | −.0044 / −.0039 / −.0025 | −.00399 | PASS cả hai |
| P4 | .8958 | −.0037 / −.0037 / −.0060 | .8987 | −.0044 / −.0042 / −.0017 | −.00406 | PASS cả hai |
| Q2 | .8956 | −.0041 / −.0039 / −.0086 | .8986 | −.0044 / −.0043 / −.0008 | −.00427 | PASS cả hai |
| Q4 | .8955 | −.0041 / −.0040 / −.0061 | .8989 | −.0048 / −.0041 / −.0006 | −.00444 | PASS cả hai |

Byte/xe/round DAGSNet 6 425–6 742 (T3 6 417), BERT 7 643–7 907 (T3 7 608); bootstrap 340 663 B (up 98 943 + down 241 720).
`t_train` r1–20 DAGSNet 2,66–2,79 h (T3 3,77 h), BERT 3,21–3,50 h (T3 5,95 h) — wall-clock khác phiên/máy, report-only.
- Mọi arm qua gate cả hai model với biên rộng (≥ −.0044 vs −.006/−.010). Chênh giữa arm ≤ .0005 mean ΔTB, ngang nhiễu một seed (≈ .0003):
  thứ hạng là theo luật số, không phải khác biệt có ý nghĩa. Q (q > 0) không giúp, như sim.
- Lớp 13 DAGSNet r20 vs T3 −.002…+.006, lớp 11 −.014…−.019, lớp 1 −.013…−.015 (gap còn lại dồn ở 11/1, không ở 13).

**Verdict (`scripts/gate_fbl_full.py` → `runs/fbl-full/verdict-pilot.json`, đủ 6 arm × 2 model): 6/6 PASS cả hai; chọn L2**
(L2 −.00380 > L4 −.00394 > P2 −.00399 > P4 −.00406 > Q2 −.00427 > Q4 −.00444). L2 = N2 tier rows 64/40/40 + `real_select=top`, `compact`.
Chênh L2–L4 .00014 trong nhiễu một seed; theo luật số, không phải bằng chứng L2 tốt hơn. Compute họ N2 cao hơn họ N4 (FLOP tổng
0,963× vs 0,920× FedAvg DAGSNet; 0,960× vs 0,917× BERT; [compute harness](compute-harness.md)), byte/round tương đương.
Bậc kế (r50) **chờ owner** — không tự nối.

## Full r50 — L4, ba seed (owner 19/11 ≈ 20:20: "chọn L4, chạy r50 cả hai backbone, thêm seed 43"; ≈ 20:40: "+ seed 41")

Owner chọn **L4** (thay L2 theo luật số; chênh .00014 trong nhiễu, L4 rẻ hơn ≈ 4 % FLOP). Luật khóa trước push = gate FB full r50
([FB](screening-budget.md) dòng "Full r50"), mỗi seed × model, tuyệt đối (không Δ vs T3):
Macro-F1 r50 **DAGSNet > .8998, BERT ≥ .8951**; bootstrap + mọi round ≤ 400 000 B/xe; executed compute ≤ 1,01× FedAvg (L4 đo 0,997×
DAGSNet / 1,001× họ N4 trên engine, tổng FLOP 0,920× / 0,917×, [compute harness](compute-harness.md)); audit artifact PASS (round 1..50,
hp/source/seed, skip ≤ 1 %, client applied, CM support == T3 cùng seed, 10 metric từ CM; s42: r1–20 trùng byte pilot; hop: prefix trùng session trước).
Spec = hàng L4P / L4B pilot đã audit (cùng base SHA, overlay, hp, cosine50, `compact`); seed 41/43 chạy mới r1–50, hp chỉ đổi `seed`
(T3 control seed 41/43 cũng chỉ khác seed: DAGSNet `t3-s41/mh-k-b21-t341-r50`, `b21-chain/mh-k-b21-t343-r50-s7`; BERT `flb-t3-s41-p3`, `flb-t3-s43-p4`).
Báo kèm (report-only): Δ vs T3 cùng seed, p10, lớp 13/11/1, mean ± SD 3 seed, hạng performance/communication/compute so roster.
Thiếu quota một phiên ⇒ budget stop lưu state, nối cross-account (dataset hop); FAIL không kéo dài/tune.
Launcher: `scripts/run_fbl_r50.py` (DAGSNet), `../fl_veremi_bert/scripts/run_fbl_r50.py` (BERT). Kernel/owner: [continue](../operations/continue.md).
