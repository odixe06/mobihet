# FullDrift — kiểm drift và kỹ thuật FL trên full data (holdout nội bộ)

Owner duyệt 15/10 tối: kiểm kỹ thuật mới **thẳng trên full** vì pilot không tái hiện động học full;
**một seed chung (42)** để so sánh phương pháp, thử seed khác để sau. Cổng dưới đây khóa **trước** khi push.

## Vấn đề đã đo

- Full mỗi round = 1 epoch ≈4.200 bước/client. DML proxy bão hòa ~0,78–0,79 từ r3–r5 trong khi LR còn ~1e-3;
  Gaussian/local đạt đỉnh r1 rồi giảm ([chẩn đoán](../../research/plateau-diagnosis-2026-10-15.md)).
- Trần centralized DAGSNet 0,8532; proxy DML r23 0,787 (test, mô tả).
- Pilot: FedAvg > DML proxy +0,027, nhưng full chưa từng đo FedAvg thuần.
- AntiMem pilot (W&B, chưa audit): chống học thuộc giúp Gaussian private (EMA, swappos) nhưng không giúp proxy.

## Dữ liệu và đánh giá

`prepare(holdout=True, train_cap=None, eval_cap=65536)` trên full 43.045.415 dòng: loại 2% cuối theo thời gian
(mỗi lớp×scenario) và receiver-group hash mod20==0 (~5%) khỏi train; `group` (receiver chưa thấy) và `old`
(temporal) mỗi tập 65.536 dòng. Official test20% (10.761.343) chỉ chạy **một lần sau epoch 6**, chỉ báo cáo.
Epoch 1–6 của lịch cosine T50 gốc (LR giống headline). Mean20client, đủ 10 metrics + per-class.

## Arm và cơ chế (seed 42)

| arm | thay đổi so với recipe full | vì sao có thể giúp | dấu hiệu sai |
|---|---|---|---|
| `fd_dml` | recipe full DML (bit-exact epoch `full_run`) | đối chứng | — |
| `fd_dml_sub4` | tổng hợp proxy 4 lần/epoch (~1.050 bước), optimizer giữ trong epoch, cùng LR/thứ tự batch | kiểm nhân quả drift: ít bước giữa hai lần tổng hợp | proxy group ep6 < đối chứng +0,010 ⇒ drift không phải nút thắt |
| `fd_dml_prox` | FedProx μ=0,01 trên proxy, neo model chung đầu round | kéo proxy về model chung, giảm drift, không thêm byte | drift không giảm, hoặc F1 < +0,010 |
| `fd_dml_avgm` | FedAvgM: server velocity β=0,9 trên delta trung bình (BN trung bình thường) | giữ hướng chung qua round, tắt nhiễu client; không thêm byte | F1 < +0,010 hoặc mất ổn định |
| `fd_fedavg` | một model chung, chỉ CE, không private/KD | kiểm xem KD với private học thuộc có kéo proxy xuống | (so mô tả với `fd_dml`) |
| `fd_fedavg_sub4` | FedAvg + 4 lần tổng hợp/epoch | drift trong bối cảnh FedAvg kinh điển | < `fd_fedavg` +0,010 |
| `fd_gaussian` | recipe full Gaussian | đối chứng Gaussian | — |
| `fd_gaussian_swappos` | swap 50% khối 9 feature receiver (dòng khác receiver cùng client) + đặt0 4 vị trí | cắt shortcut receiver; pilot +0,016–0,018 | private gp group < +0,010 |
| role `private_ema` | EMA trọng số private β=0,5^0,5/epoch, BN tính lại trên 65.536 dòng local cố định | trung bình quỹ đạo, giảm học thuộc; pilot Gaussian +0,036–0,039 | < +0,010 so private |

Mọi arm ghi `drift_mean` (‖θ_k−θ_chung‖/‖θ_chung‖ proxy sau train local), `prox_mean`, `server_velocity_norm`,
`swap_fraction`. Byte: sub4 gấp 4 lần/epoch (báo kèm), prox/avgm/FedAvg không đổi byte.

## Cổng khóa (group holdout, epoch 6, seed 42 — sàng lọc)

Primary: DML/FedAvg = proxy head; Gaussian = private gp; EMA = private_ema gp so private gp cùng arm.
- Kỹ thuật **có triển vọng** nếu technique − đối chứng ≥ +0,010 và benign ≥ −0,005 ở epoch 6.
- `fd_dml_sub4` dương ⇒ drift là nút thắt (ưu tiên kỹ thuật giảm drift/tăng tần suất tổng hợp có giới hạn byte);
  âm/≈0 ⇒ drift không phải nút thắt, hướng sang trần thông tin/LR.
- `fd_fedavg` vs `fd_dml` proxy: so sánh mô tả, quyết định headline FL cho bước sau (owner chốt).
- Một seed là sàng lọc, không là kết luận cuối; kỹ thuật có triển vọng mới chạy seed khác/r50.
- Không chọn epoch/HP theo holdout hay test; không quét μ/β/PARTS sau khi thấy kết quả. Old báo kèm.

## Giới hạn

Epoch 6/50 không chứng minh r50. sub4 tăng byte 4×, cần đánh giá lại dưới B2/contact.
FedProx/FedAvgM dẫn xuất cho SGD; ở đây optimizer AdamW của recipe (chỉ thêm số hạng/velocity).
Không resume qua session: probe một session/arm.

## Vòng 2 — owner duyệt 15/10 ~22:00Z (cổng khóa trước push; PUSHED 16/10 02:19Z)

Kết quả vòng 1: [results](results.md). Owner chọn **cả bốn hướng proxy**; Gaussian swappos/EMA chờ (ưu tiên proxy).
Seed 42; đối chứng 6 epoch lấy từ vòng 1 (cùng fixture/seed/code path, chạy tất định).

| arm | thay đổi | cơ chế/kỳ vọng | cổng (group, epoch 6 trừ khi ghi) |
|---|---|---|---|
| `fd_dml_sub4_q4`, `fd_fedavg_sub4_q4` | sub4 + QProxy q4 (port antimem: up int4 stochastic theo hàng, server exact/EF, down int4 nearest) | giữ lợi sub4 với byte ≈0,52× dense | Δ so sub4 vòng 1 ≥ −0,010 primary và benign ≥ −0,010 |
| `fd_dml_sub16_q4`, `fd_fedavg_sub16_q4` | 16 lần tổng hợp/epoch + q4 | drift sub4 còn 0,18; tổng hợp thường hơn giảm tiếp | Δ so `*_sub4_q4` ≥ +0,010, benign ≥ −0,005 |
| `fd_dml_sub4_prox`, `fd_fedavg_sub4_prox` | sub4 + FedProx μ0,01 | prox giữ benign (+0,098 vòng 1) trong khi sub4 tăng macro | benign Δ so sub4 ≥ +0,010 và primary Δ ≥ −0,005 |
| `fd_dml`, `fd_dml_sub4`, `fd_fedavg`, `fd_fedavg_sub4` @ep15 | cùng arm, 15 epoch (thư mục `_e15`) | lợi sub4 còn khi đi xa hơn về r50? | ep15: sub4 − đối chứng ≥ +0,010 (DML, FedAvg riêng); ep1–6 tái lập vòng 1 |

Không quét μ/β/PARTS/bit sau khi thấy kết quả; old/test báo kèm, test chỉ epoch cuối.

### Cổng phụ private gp — owner 16/10 ~03:00Z (khóa ~04:00Z, trước mọi kết quả ep6 vòng 2)

Owner đổi mục tiêu sang **dự đoán của client**: primary = private gp (A, giữ R2), báo kèm góc nhìn theo prior nhãn
từng client và client tệ nhất ([hướng client](../../research/client-model-directions-2026-10-16.md)).
Lúc khóa đã thấy W&B r2 (chưa audit), chưa thấy epoch 6. Arm DML dùng **cùng rule** với cổng proxy, đo trên private gp group:
`fd_dml_sub4_q4` vs `fd_dml_sub4` noninferior; `fd_dml_sub16_q4` vs `fd_dml_sub4_q4` gain; `fd_dml_sub4_prox` vs
`fd_dml_sub4` benign; `fd_dml_sub4@e15` vs `fd_dml@e15` ep15 horizon. FedAvg không có private (R2) ⇒ chỉ mô tả.
Vòng 1 private gp (đã thấy trước khi khóa, chỉ mô tả): sub4 +0,035, prox −0,013, avgm +0,002.
Mã: `PRIVATE_GATES` trong [report_fd.py](../../../methods/full-drift/builds/fd-probe/scripts/report_fd.py).

## Vòng 3 — mục tiêu client (owner 16/10; cổng khóa ~04:05Z trước push)

Owner 16/10: primary = **private gp group ep6** (A), giữ **R1** (trao đổi ngắn, contact/byte hạn chế) và **R2**
(private có thể khác kiến trúc; hiện DAGSNet để so). Mọi arm vòng 3 **không thêm byte, không thêm contact**
(1 tổng hợp/epoch như `fd_dml`) và chỉ đổi phía client, nên dùng được với private bất kỳ kiến trúc.
Đối chứng `fd_dml` vòng 1 (tất định; vòng 2 kiểm lại ep1–6 qua `fd_dml@e15`). Seed 42.

| arm | thay đổi | bằng chứng → cơ chế | cổng private gp group ep6 |
|---|---|---|---|
| `fd_dml_m4b` | đặt 0 khối 9 feature receiver + 4 vị trí (train và eval, cả hai model; như AntiMem) | pilot DML private gp +0,021/+0,028 hai seed → bỏ shortcut receiver mà private học thuộc | Δ ≥ +0,010, benign ≥ −0,005 |
| `fd_dml_swappos` | swap 50% khối receiver giữa receiver khác cùng client + đặt 0 4 vị trí | pilot +0,019/+0,025; Gaussian full +0,052 → phi tương quan receiver–nhãn | như trên |
| `fd_dml_ntd` | KL proxy→private chỉ trên lớp không đúng (FedNTD, Lee et al. NeurIPS 2022), trọng số giữ nguyên | head→gp +0,12 cho thấy private lệch prior local mạnh → bỏ xung đột giữa CE và lớp đúng của thầy, giữ tri thức lớp khác | như trên |
| `fd_dml_q4` | QProxy q4 (up int4 stochastic, server exact/EF, down int4) ở 1 tổng hợp/epoch | pilot q4 ±0,001 byte 7,65× nhỏ → R1: payload/contact nhỏ hơn mà client không mất | private gp và proxy head Δ ≥ −0,010, benign ≥ −0,010 |

Proxy head của m4b/swappos/ntd báo mô tả. Kèm cột local (prior nhãn từng client) và client tệ nhất, mô tả.
Không quét tỉ lệ swap/feature/nhiệt độ NTD sau khi thấy kết quả. Session `nguyenminhtran060113` g: m4b ‖ swappos,
h: ntd ‖ q4, 4,5h mỗi session (arm ≈3,62h + setup 0,1h).

## Vòng 4 — client, quota ngắn (cổng khóa 16/10 ~06:30Z trước push)

Owner 16/10: dùng account còn quota ngắn cho pilot; giữ R1/R2. `odixe0502` (4,0h, reset 17/10 00:00Z), session 3,95h:
GPU0 `fd_dml_hkd`, GPU1 `fd_gaussian_m4b` (arm ≈3,62h/2,5h + setup 0,1h). Đối chứng vòng 1 cùng fixture/seed.

| arm | thay đổi | bằng chứng → cơ chế | cổng private gp group ep6 |
|---|---|---|---|
| `fd_dml_hkd` | private khớp tầng ẩn 256-d của proxy qua projector tuyến tính **local** (khởi tạo đơn vị, không upload); MSE chuẩn hóa theo ‖h_proxy‖², cùng trọng số thích nghi với KL; proxy không đổi (FedKD hidden loss, Wu et al. Nat. Commun. 2022) | ProxyFuse: private đúng mà proxy sai chỉ ≈2% ⇒ private là bản chép kém của proxy; logit KL chỉ chở 16 số/dòng → thêm kênh đặc trưng | vs `fd_dml`: Δ ≥ +0,010, benign ≥ −0,005 |
| `fd_gaussian_m4b` | Gaussian + đặt 0 13 feature receiver/vị trí (train, draw, eval) | AntiMem pilot Gaussian m4b +0,025 hai seed (lớn nhất); full swappos +0,052 → bỏ shortcut receiver | vs `fd_gaussian`: Δ ≥ +0,010, benign ≥ −0,005 |

0 byte thêm, 1 contact/epoch (R1); projector/feature mask chỉ phía client (R2). Không quét trọng số/scale sau kết quả.

## Vòng 5 — m4b: horizon và ghép (owner 16/10 ~08:20Z; cổng khóa ~08:25Z trước push)

Vòng 3: m4b private gp +0,036, proxy +0,052 (ep6). Owner: kiểm horizon + ghép trước; headline full chỉ đổi khi
horizon + seed 43 xác nhận. Session `nguyenminhtran060113` j (8,95h): GPU0 `fd_dml_m4b:15`; GPU1 `fd_dml_m4b_q4` →
`fd_dml_m4b_sub4_q4`.

| arm | so với | câu hỏi | cổng (group) |
|---|---|---|---|
| `fd_dml_m4b@e15` | `fd_dml@e15` (vòng 2) | lợi m4b còn ở ep15? | private gp ep15 Δ ≥ +0,010 (proxy báo kèm, cùng rule); ep1–6 phải tái lập vòng 3 |
| `fd_dml_m4b_q4` | `fd_dml_m4b` | R1: nén 7,65× có giữ lợi? | private gp và proxy Δ ≥ −0,010, benign ≥ −0,010 |
| `fd_dml_m4b_sub4_q4` | `fd_dml_m4b` | m4b cộng với tổng hợp thường hơn? (4× contact, byte ≈0,52× dense) | private gp Δ ≥ +0,010, benign ≥ −0,005 |

## Vòng 6 — seed 43 xác nhận m4b (owner 16/10 ~16:45Z; cổng khóa ~16:50Z trước push)

Owner: headline full chỉ đổi sang m4b khi horizon (vòng 5) + seed 43 xác nhận; chọn 2 session trên `anhtruong1909`
(9,36h live, reset 17/10 00:00Z), mỗi session 4,2h (DML ep6 đo 205–217 phút + setup). Cùng source/fixture vòng 5
(fd.py SHA không đổi), chỉ `--seed 43`; smoke local vốn chạy seed 43. Đối chứng cùng seed trong vòng (không tái dùng s42).
Session k: GPU0 `fd_dml`, GPU1 `fd_dml_m4b`; session l: GPU0 `fd_dml_sub4_q4`, GPU1 `fd_dml_m4b_sub4_q4`.
Lúc khóa đã thấy W&B s42 (chưa audit) `fd_dml_m4b_sub4_q4` ep6 private gp 0,732/proxy 0,849, `m4b_q4` 0,701/0,816, `m4b@e15` ep14 0,728/0,849.

| so sánh (seed 43, group ep6) | câu hỏi | cổng private gp (primary) | proxy head (cùng rule, báo kèm) |
|---|---|---|---|
| `fd_dml_m4b` vs `fd_dml` | lợi m4b lặp lại ở seed khác? | Δ ≥ +0,010, benign ≥ −0,005 | như trên |
| `fd_dml_m4b_sub4_q4` vs `fd_dml_m4b` | tổng hợp 4×/epoch cộng thêm trên m4b? | như trên | như trên |
| `fd_dml_m4b_sub4_q4` vs `fd_dml_sub4_q4` | m4b cộng thêm trên sub4_q4? | như trên | như trên |
| `fd_dml_sub4_q4` vs `fd_dml` | lợi sub4_q4 lặp lại? | như trên | như trên |

Quy tắc headline (owner): đổi khi `fd_dml_m4b@e15` PASS (vòng 5) **và** `fd_dml_m4b` vs `fd_dml` seed 43 PASS cả private
gp lẫn proxy. Một seed thêm vẫn là sàng lọc, không paired CI. R1: sub4 = 4 contact/epoch; owner 16/10 yêu cầu
mô phỏng contact thực tế trước khi chọn tần suất tổng hợp (sub16 chờ).

## Vòng 7 — seed 43 swappos + Gaussian m4b (owner 16/10 ~17:25Z; cổng khóa ~17:29Z trước push)

Owner: dùng nốt `annhannh331` (5,29h, reset 00:00Z) lặp kỹ thuật đã PASS ở seed 43. Session m (4,2h): GPU0 `fd_dml_swappos`,
GPU1 `fd_gaussian_m4b`, `--seed 43`, cùng source vòng 5/6. `fd_gaussian` s43 không vừa quota (cần 4,85h trên một GPU).

| so sánh (seed 43, group ep6) | cổng |
|---|---|
| `fd_dml_swappos` vs `fd_dml` (s43, vòng 6) | private gp Δ ≥ +0,010, benign ≥ −0,005; proxy head cùng rule báo kèm |
| `fd_gaussian_m4b` s43 | **mô tả**: so `fd_gaussian` s42 (0,5882) và `fd_gaussian_m4b` s42 (0,6553); không có đối chứng cùng seed |
