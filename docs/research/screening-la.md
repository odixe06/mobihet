# LA — logit adjustment trong L4 (khóa 24/11/2026, trước push)

Nền = [FB-L](screening-fbl.md) cấu hình L4. Owner 24/11: ablation A3 (bỏ LA) tốt hơn L4. Owner yêu cầu tìm cách cải tiến LA cho hợp
với L4; nếu không có cách nào tốt hơn thì bỏ LA. Phạm vi: chỉ DAGSNet trong repo này, sim 200 xe seed 42/43, PASS thì pilot.
Gate: **chỉ macro, ≥ A3**.

## Bằng chứng → cơ chế → lợi kỳ vọng → dấu hiệu bác bỏ

- **Bằng chứng** (ablation repo `~/project/experiment`, `experiments/ablation-l4/*/results.csv`, R50 official test, so cùng seed
  với hàng `mobihet_l4`): A3 dương ở 24/24 cặp seed, 9 kịch bản. ΔMacro TB: DAGSNet +.0002, BERT +.0002, case1 +.0002,
  CNN-BiGRU +.0014, case3 +.0017, MLP +.0031, case2 +.0042, TabM +.0046, SAE +.0087. Recall macro tăng, precision macro giảm nhẹ.
  So với A4 (≈ L4, Δ ≤ .0003), F1 từng lớp TB 9 kịch bản: lớp 11 +.028, 13 +.005, 1 +.005, 6 +.003, 15 +.003. Các lớp khác ≈ 0.
- **Cơ chế** (CM từng xe R50 seed 42, TabM/DAGSNet/SAE/MLP/BERT, A3 so với A4; [script](../../scripts/analyse_la_cohort.py)):
  L4 train `CE(z + log π_local − log π_global, y)`, suy luận bằng `argmax z`. Kết quả là logit thô mang prior **toàn cục**,
  trong khi test là dữ liệu tương lai của **chính xe đó**, tức prior local.
  - Lớp xe **có** trong train: lớp 1/6/11/15 có shift dương (median +.22…+.36), nên LA làm model đoán thiếu chúng.
    Bỏ LA thì recall lớp 11 trên các xe có lớp 11 tăng: TabM .298→.356, MLP .557→.606, SAE .126→.170, DAGSNet .778→.782.
    Toàn bộ gain macro đến từ đây.
  - Lớp xe **không có** trong train (absent): shift âm, nên LA che lớp vắng khỏi bị CE dòng thật đè xuống. Bỏ LA thì recall
    trên 26.999 dòng absent giảm: TabM .801→.643, DAGSNet .849→.813, BERT .827→.737, SAE .686→.544, MLP .803→.685.
    Số dòng này quá ít để đổi macro.
  - Lịch sử: P20 C15 `real_no_la` chỉ là train local không synthetic (−.0009/−.0000 so với C0). Chưa có study nào bỏ LA khi có synthetic + Q.
- **Arm** (HP `la_mode` ở `methods/mobihet_budget.py`; count, bootstrap, Q, synthetic, byte, compute như L4):
  - `full` = L4.
  - `off` = A3.
  - `absent` (A3m): shift 0 cho lớp xe có, giữ shift L4 cho lớp `n_kc = 0`.
    Lưu ý: Laplace +1 làm shift của lớp vắng thành **dương** ở xe nhỏ đối với lớp hiếm toàn cục.
  - `mask` (A3x): shift 0 cho lớp xe có, −1e4 cho lớp vắng. Đây là restricted softmax kiểu FedRS cứng: dòng thật không bao giờ
    đè lớp vắng, và không phụ thuộc prior toàn cục.
  - Không HP mới, không dò τ.
  - CE lưu để chọn dòng (C5) là chính CE đó, nên cũng đổi theo.
- **Lợi kỳ vọng:** A3m/A3x ≈ A3 về macro (giữ recall lớp có mặt), recall absent về gần L4.
  Có thể hơn A3 một chút nhờ giảm FN lớp vắng; không kỳ vọng hơn A3 ≥ +.001.
- **Bác bỏ:** candidate có macro TB < A3, tức che lớp vắng làm hại lớp có mặt (FP lớp vắng tăng). Khi đó chọn A3 (bỏ LA).
  Sim là dấu hiệu không dùng test cho A3 so với L4 (report-only). DAGSNet ở full-scale chỉ +.0002, nên sim 2 seed có thể không
  phân biệt được. Đọc cùng nhiễu seed ±.0006.

## Luật khóa

- **Sim** (`scripts/run_sim_la.py`): fold train → fold val, 200 xe (`pick` seed 7), r20 cosine20, compact, DAGSNet W1.
  4 arm L4/A3/A3m/A3x × seed 42/43 = 8 kernel. Base `a6e77a1` (origin/main) + overlay `mobihet_{compute,lite,budget}.py`.
  Audit `scripts/audit_sim_batch.py la` (8/8 PASS mới xét).
  - Candidate PASS ⇔ macro pooled fold val **TB r16–20, trung bình 2 seed ≥ A3** cùng batch.
  - Chọn candidate PASS có macro cao nhất. Không có candidate nào PASS thì chọn A3.
  - Mọi số khác (r20, p10, absent, lớp) chỉ report.
- **Pilot** (chỉ khi có candidate PASS): full-scale 5.343 xe, r20 trên lịch cosine50, seed 42/43, candidate được chọn + A3 cùng repo.
  PASS ⇔ macro own-test TB r16–20, trung bình 2 seed ≥ A3. Sau pilot dừng, báo owner (r50 do owner quyết).

## Kết quả

### Audit 24/11/2026, 06:06 +07

Cả 8 kernel version 1 COMPLETE và đã pull đủ file theo bộ lọc audit. Năm output đã có từ lượt collect trước, ba output kéo trong phiên này.
Đối chiếu push receipt và hash batch, hash notebook/metadata đã khóa, hash toàn bộ 103 file provenance mỗi kernel với base + overlay,
config từng arm, log kiểm cache NPY, compile gate hai rank và lịch cosine20. Cả 8 có đúng r1–r20, AMP skip = 0,
mọi xe có bước applied và chuỗi hash Q tiêu thụ khớp hash Q sinh ở round trước. Tính lại đủ 10 metric, p10 và F1 từng lớp từ
cả 160 CM, khớp log đến sai số tuyệt đối 1e-12. Panel mỗi round gồm 200 xe, 1.476.064 dòng train và 211.118 dòng val.
Audit dựa trên CM, provenance và log. Batch sim không lưu prediction từng dòng, trọng số hay payload Q để replay.

| Arm–seed | Kernel chính xác | Audit |
|---|---|---|
| L4–42 | `keocaosu/mh-sim-la-l4-s42-r20/1` | PASS r20 |
| A3–42 | `chuchimnon/mh-sim-la-la-off-s42-r20/1` | PASS r20 |
| A3m–42 | `minhtriethihi/mh-sim-la-la-absent-s42-r20/1` | PASS r20 |
| A3x–42 | `heykewtiie/mh-sim-la-la-mask-s42-r20/1` | PASS r20 |
| L4–43 | `nguyetquedo/mh-sim-la-l4-s43-r20/1` | PASS r20 |
| A3–43 | `lelovo/mh-sim-la-la-off-s43-r20/1` | PASS r20 |
| A3m–43 | `noopthinh/mh-sim-la-la-absent-s43-r20/1` | PASS r20 |
| A3x–43 | `sampanh/mh-sim-la-la-mask-s43-r20/1` | PASS r20 |

### Cổng đã khóa

| Arm | Macro TB s42 | Macro TB s43 | Macro TB hai seed | ΔTB vs A3 | Macro r20 | p10 r20 | Recall absent TB | Cổng |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| L4 | .908718 | .909585 | .909151 | −.000720 | .909395 | .786312 | .701047 | Đối chứng |
| A3 | .909669 | .910074 | .909871 | .000000 | .910379 | .779453 | .618325 | **Chọn** |
| A3m | .908231 | .909796 | .909013 | −.000858 | .909531 | .786589 | .683770 | FAIL |
| A3x | .908270 | .908963 | .908616 | −.001255 | .908902 | .784643 | .691099 | FAIL |

**Chọn A3 (`la_mode=off`) theo luật sim.** Cả hai ứng viên hụt macro ở cả hai seed, nên nhánh pilot theo luật đã khóa kết thúc tại sim.
A3m/A3x tăng recall absent TB +.065/+.073 so với A3 trên 191 dòng absent/round và tăng p10 r20 +.0071/+.0052.
Ở r20, F1 lớp 13 giảm −.0071/−.0076 và lớp 11 giảm −.0032/−.0061. Các số này chỉ để báo cáo.
Kết luận áp dụng cho screening DAGSNet fold val 200 xe. Khi đổi LA, CE dùng để chọn dòng thật cũng đổi.

Mỗi kernel ghi 362.351 B bootstrap/xe và 6.304–6.456 B trao đổi trong 20 round/xe. Train cộng dồn 0,84–0,88 h/kernel.

### Mười metric pooled tại r20

| Arm–seed | Accuracy | P macro | P micro | P weighted | R macro | R micro | R weighted | F1 macro | F1 micro | F1 weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L4–42 | .922063 | .925275 | .922063 | .915164 | .900223 | .922063 | .922063 | .908471 | .922063 | .914871 |
| A3–42 | .922072 | .925474 | .922072 | .915684 | .902282 | .922072 | .922072 | .910136 | .922072 | .915623 |
| A3m–42 | .921688 | .923670 | .921688 | .914985 | .901091 | .921688 | .921688 | .908610 | .921688 | .915063 |
| A3x–42 | .921063 | .923091 | .921063 | .914569 | .901259 | .921063 | .921063 | .908531 | .921063 | .914639 |
| L4–43 | .921712 | .925196 | .921712 | .915036 | .902612 | .921712 | .921712 | .910320 | .921712 | .915213 |
| A3–43 | .922285 | .924426 | .922285 | .915981 | .903580 | .922285 | .922285 | .910622 | .922285 | .916044 |
| A3m–43 | .921456 | .924845 | .921456 | .914943 | .903053 | .921456 | .921456 | .910452 | .921456 | .915105 |
| A3x–43 | .921191 | .923360 | .921191 | .914463 | .902210 | .921191 | .921191 | .909274 | .921191 | .914674 |

Bằng chứng local trong `runs/sim-kaggle-la/`: `batch.json`, `status.json`, `collection/*.collect.log`, `audit.json`,
`audit-detail.json`, `verdict-r20.json`. Output gốc ở `artifacts/sim-kaggle-la/`.

## Quyết định owner 24/11/2026

Nhận **A3 (`la_mode=off`) làm MobiHet chính**, vẫn tên `mobihet`. Đóng trục LA: không còn biến thể LA nào có lợi kỳ vọng vượt
nhiễu (τ-scale là dò HP; prior lúc suy luận, LA trên synthetic/Q thuộc họ prior/ngưỡng đã đóng). Số R50 của MobiHet là 24 run
ablation A3 đã audit ở repo `~/project/experiment`, kiểm lại từ CM/log, cộng 3 seed chạy mới (BERT s43, CNN-BiGRU s43,
mixed-case1 s41). Ablation A1/A2/A4 chạy lại trên nền mới. L4 có LA và ablation L4 chuyển vào
[archive/l4-la-20261124](../../archive/l4-la-20261124/README.md). Kết quả và chi phí hiện hành:
[MobiHet](../../../experiment/docs/research/mobihet.md).
