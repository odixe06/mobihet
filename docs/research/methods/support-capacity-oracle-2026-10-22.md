# P35 — nguyên nhân theo lớp, độ chịu lệch manifold và trần năng lực của bộ sinh học được (frozen-head, Project48)

22/10/2026 ~13:15Z; owner: "tiến hành thực hiện đo để tìm thêm các nguyên nhân và thông tin" sau [P34 FAIL](structured-support-generator-2026-10-22.md).
**Protocol và cách đọc dưới đây khóa trước khi chạy**; K, số epoch, σ cố định, không chọn lại sau khi thấy kết quả.

## Câu hỏi còn mở sau P32–34

1. Lợi posMirror/suddenStop của support thật (+0,076 / +0,054 F1) cần **mẫu thật của chính hai lớp đó**, hay của benign kề chúng? (P33: hình học
   mức lớp không giải thích ΔF1 lớp ⇒ nghi hiệu ứng theo cặp.)
2. Head private chịu được mẫu lệch khỏi dòng thật bao nhiêu trước khi mất lợi? (định mức precision bộ sinh cần đạt)
3. Một mô hình **phụ thuộc phi tuyến học được** có tái tạo được lợi của support thật không, hay trần nằm ở chính việc sinh mẫu? Mọi bộ sinh
   tham số đến nay (Gaussian, copula, V4-S, G61, P34) ≤ +0,0025.

## Harness và arm

Harness P32 nguyên vẹn (xa s42 ep15, fixture 2,5 %, 20 client, Linear cuối 1.000 bước, teacher Project48, bank 4.096 dòng lịch lớp đều seed 24).
Bank real và copula sinh lại đúng seed P32 (kiểm: teacher class_agreement khớp JSON P32); gaussian client 0 phải tái lập P32. Mốc dùng lại kết quả
đã audit: gaussian/copula/real (P32), own_bal (P33b), pcop (P34). PS = {positionMirroring, suddenStop}.

| nhóm | arm | cách tạo | câu hỏi |
|---|---|---|---|
| A nguyên nhân | `real_cPS` | bank real, dòng PS thay bằng dòng copula cùng vị trí | lợi PS mất khi chỉ PS giả? |
| | `cop_rPS` | bank copula, dòng PS thay bằng dòng real | chỉ cần PS thật? |
| | `real_cB` | bank real, dòng benign thay bằng copula | benign giả làm mất gì? |
| | `cop_rB` | bank copula, dòng benign thay bằng real | chỉ cần benign thật? |
| B độ chịu lệch | `real_j05`, `real_j20` | bank real + N(0, σ²) trên 53 feature active, σ = 0,05 / 0,20 (z) | lợi giảm theo σ thế nào |
| C năng lực (fit trên pool, R3-ineligible, chỉ chẩn đoán) | `gmm8`, `gmm32` | mỗi lớp một GMM full covariance K = 8 / 32 trên asinh(z) 53 chiều; EM trên GPU (k-means++ + 10 vòng Lloyd, ≤ 100 vòng EM, tol 1e-4, reg 1e-4), seed 53/54 | Gaussian nhiều mode (EM cộng được ⇒ federated được) đủ chưa? |
| | `made` | MADE-MoG có điều kiện lớp: 2 lớp ẩn 512, 10 thành phần Gaussian/chiều, log σ ≥ −6, Adam 1e-3, batch 2.048, 30 epoch, seed 55, trên asinh(z) | mô hình autoregressive phi tuyến đủ chưa? |

Mẫu GMM/MADE chặn trong khoảng min–max của pool theo từng feature (asinh) để giữ hữu hạn fp16.

**Kiểm chép (memorization):** trung vị khoảng cách 1-NN (asinh(z), 53 chiều) từ mẫu tới pool so với của 4.096 dòng thật **không** nằm trong pool
(dòng fixture chưa sample, 256/lớp, seed 57). Tỷ lệ < 0,8 ⇒ arm bị đánh dấu "chép", không tính là năng lực.

Đo kèm mọi arm: hình học đặc trưng private như P33 (precision/recall/landing, tham chiếu train client khác), teacher đồng ý, ΔF1 theo lớp,
predictions → confusion → 10 metric.

## Cách đọc (khóa)

G_PS = ΔF1 posMirror + ΔF1 suddenStop so gaussian (real 0,130; copula 0,041).

- **A:** `real_cPS` mất ≥ 50 % của (real − copula) theo G_PS ⇒ cần mẫu thật **của chính lớp PS**; `cop_rPS` lấy lại ≥ 50 % ⇒ PS thật là đủ.
  Ngược lại mà các arm benign mới làm G_PS đổi ≥ 50 % ⇒ hiệu ứng nằm ở benign kề. Mean/benign/worst báo kèm.
- **B:** tỷ lệ giữ lợi mean (arm − gaussian)/(real − gaussian) tại σ = 0,05 và 0,20 — định mức độ chính xác cho bộ sinh.
- **C:** arm **đủ năng lực** khi mean − gaussian ≥ +0,008, G_PS ≥ +0,065 (½ real), benign ≥ −0,005, worst ≥ −0,010 và không "chép".
  GMM đủ ⇒ bước sau thiết kế GMM federated (EM với thống kê cộng, kế toán R1); chỉ MADE đủ ⇒ mật độ neural federated (nặng hơn, R1/R2 riêng);
  không arm nào đủ ⇒ đường "bộ sinh tốt hơn" đóng ở năng lực này cho harness frozen-head; chuyển trọng tâm sang nguồn tri thức.
- Chẩn đoán, không cổng phương pháp; không official test; old/group là development set.

## Kết quả — 22/10 ~13:35Z, local RTX, 833 s

[JSON](../../methods/full-drift/results/probe35-capacity-oracle-s42.json) (+ `.npz` predictions, `.pt` heads),
[source](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe35.py). Bank real/copula khớp P32 (teacher agreement bằng nhau); gaussian
client 0 tái lập P32 bit-exact; **360 lượt predictions → confusion → 10 metric khớp**.

| arm | mean | Δ vs gaussian | benign | worst | G_PS | Δ posMirror | Δ suddenStop | precision / recall / landing | precision posMirror / suddenStop |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| gaussian (P32) | 0,8285 | — | 0,7815 | 0,7830 | — | — | — | 0,49 / 0,14 / 0,42 | 0,39 / 0,60 |
| copula (P32) | 0,8308 | +0,0024 | 0,7875 | 0,8048 | +0,041 | +0,019 | +0,022 | 0,45 / 0,80 / 0,66 | 0,35 / 0,35 |
| real (P32) | 0,8398 | +0,0114 | 0,7917 | 0,8166 | +0,129 | +0,076 | +0,054 | 0,86 / 0,86 / 0,82 | 0,86 / 0,88 |
| `real_cPS` | 0,8317 | +0,0032 | 0,7902 | 0,8108 | +0,015 | −0,014 | +0,029 | 0,80 / 0,85 / 0,79 | 0,35 / 0,35 |
| `cop_rPS` | 0,8351 | +0,0066 | 0,7859 | 0,7996 | +0,107 | **+0,079** | +0,028 | 0,52 / 0,81 / 0,70 | 0,86 / 0,88 |
| `real_cB` | 0,8390 | +0,0105 | 0,7871 | 0,8131 | +0,121 | +0,077 | +0,044 | 0,83 / 0,86 / 0,81 | 0,86 / 0,88 |
| `cop_rB` | 0,8318 | +0,0033 | 0,7918 | 0,8076 | +0,040 | +0,011 | +0,029 | 0,49 / 0,80 / 0,67 | 0,35 / 0,35 |
| `real_j05` | 0,8349 | +0,0064 | 0,7859 | 0,8012 | +0,075 | +0,053 | +0,022 | 0,57 / 0,84 / 0,68 | 0,47 / 0,35 |
| `real_j20` | 0,8294 | +0,0010 | 0,7820 | 0,7861 | +0,016 | +0,015 | +0,001 | 0,35 / 0,43 / 0,44 | 0,23 / 0,29 |
| `gmm8` | 0,8329 | +0,0044 | 0,7915 | 0,8080 | +0,054 | +0,022 | +0,033 | 0,56 / 0,69 / 0,69 | 0,59 / 0,44 |
| `gmm32` | 0,8329 | +0,0044 | 0,7910 | 0,8074 | +0,056 | +0,016 | +0,040 | 0,59 / 0,75 / 0,70 | 0,51 / 0,51 |
| `made` | 0,8311 | +0,0026 | 0,7908 | 0,8060 | +0,040 | +0,013 | +0,027 | 0,47 / 0,82 / 0,64 | 0,38 / 0,30 |

Kiểm chép: trung vị 1-NN tới pool / của dòng thật ngoài pool (1,84) = gmm8 1,67, gmm32 1,52, made 1,99 ⇒ không chép (mẫu xa dòng thật hơn cả
dòng thật mới). GMM hội tụ ≤ 100 vòng EM; MADE NLL held-out −48,5 → −61,3 → −75,3 ở epoch 10/20/30 (còn giảm; 30 epoch đã khóa).

- **A — lợi PS cần mẫu thật của chính lớp PS:** `real_cPS` mất 129 % của (real − copula) theo G_PS; `cop_rPS` lấy lại 74 % (posMirror +0,079 = real).
  Benign giả gần như không mất gì: `real_cB` mất 10 % G_PS, giữ 92 % lợi mean; `cop_rB` lấy lại 0 %. suddenStop chỉ lấy lại một nửa khi riêng PS thật
  (+0,028 vs +0,054) ⇒ phần còn lại phụ thuộc độ thật của lớp khác (benign giữ +0,044). Phần mean còn thiếu ở `cop_rPS` (0,0048) nằm ở timeDelay
  (−0,015 vs +0,014), constPosOff, dataReplay — cũng là lớp tấn công kề benign.
- **B — head chịu lệch rất ít:** nhiễu đẳng hướng σ = 0,05 z giữ 57 % lợi mean (suddenStop còn +0,022: atom chung bị nhòe), σ = 0,20 còn 8 %.
- **C — không bộ sinh học được nào đủ năng lực:** GMM 8/32 +0,0044 (39 % khoảng, G_PS 0,054–0,056 < 0,065; benign +0,010, worst +0,025 — tốt nhất
  trong mọi bộ sinh từ thống kê gộp tới nay, hơn copula +0,002); MADE +0,0026. Theo khóa: đường "bộ sinh tốt hơn" **đóng ở năng lực này** cho
  harness frozen-head; GMM 8 ≈ GMM 32 ⇒ thêm thành phần không thêm lợi.
- Giới hạn: một seed, frozen-head, fixture 2,5 %; MADE có thể chưa hội tụ (NLL held-out còn giảm, phần giảm có thể là co thành phần quanh atom);
  pool-fit là trần lạc quan so với bản federated.

## Kết luận và hướng tiếp (đề xuất, chưa code)

1. Thông tin mà support thật chở tới head private là **vị trí chính xác của vài lớp tấn công hiếm local cạnh benign** (posMirror, suddenStop, rồi
   timeDelay, constPosOff, dataReplay) ở receiver của client khác — độ chính xác cỡ 0,05 z. Mẫu benign giả không làm hại.
2. Mọi bộ sinh từ thống kê gộp (Gaussian, copula, V4-S, G61, P34, GMM, MADE) đặt mẫu các lớp này quá xa manifold đặc trưng private
   (precision posMirror ≤ 0,59 vs 0,86) ⇒ trần quan sát ~+0,0044 mean (GMM, pool-fit), dưới cổng +0,005, chưa tính byte R1 (GMM full-cov ~190 k số/lớp
   × vòng EM — vượt R1 nhiều lần).
3. Đề xuất cho owner chọn: (a) đóng hướng bộ sinh, chuyển sang nguồn tri thức (teacher Project48/CCVR đã PASS teacher) và động học online;
   (b) một phép thử cuối, khóa riêng: GMM chỉ cho lớp tấn công hiếm + benign Gaussian (dồn byte vào lớp mang lợi) kèm kế toán byte R1 — chỉ đáng
   nếu (a) không có ứng viên tốt hơn.
