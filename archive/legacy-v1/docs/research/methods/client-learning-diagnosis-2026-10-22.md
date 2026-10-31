# P33 — client học gì từ một support: phân tích cách học của head private trước khi thiết kế bộ sinh

22/10/2026 ~11:00Z. Owner: "phải phân tích kỹ cách học của client, để đưa ra phương án phù hợp dựa trên các kiến thức đã biết".
Đọc trước [P32](synthetic-support-realism-2026-10-22.md), [P29–31](synthetic-gradient-diagnosis-2026-10-22.md),
[probe 14–22, vòng 24–25](../../methods/full-drift/results.md). **Protocol và cách đọc dưới đây ghi trước khi chạy.**

## 1. Mô hình học của client — từ bằng chứng đã audit

Mỗi bước, head private nhận ba tín hiệu: CE own (logit-adjusted), KL teacher cố định trên dòng own (`kdpx`), KL teacher trên 256 mẫu
tổng hợp (`gkd`). Teacher = snapshot proxy gộp đầu round.

| # | dữ kiện | nguồn | hệ quả cho bộ sinh |
|---|---|---|---|
| 1 | Backbone private đủ: refit Linear trên dữ liệu thật gộp ≈ 0,84 bất kể có FL; lợi frz/gkd nằm hết ở tầng quyết định | P14, P16A, P18 | mẫu tổng hợp chỉ cần dạy **tầng quyết định**, nhìn qua backbone của chính private |
| 2 | Head chỉ được neo nơi có input: khởi tạo = proxy, chỉ KD trên dòng own vẫn rơi 0,85 → 0,83 trong 1 epoch; vùng ngoài giá đỡ local bị ngoại suy ⇒ false positive trên benign của receiver lạ | P14 hướng lỗi, P15 `pinit_kdonly` | vùng mẫu tổng hợp rơi vào (trong không gian đặc trưng private) quyết định ranh giới nào được giữ |
| 3 | Thiếu input chứ không thiếu nhãn: dòng thật client khác **không nhãn** + teacher cố định 0,842 ≈ proxy; Gaussian 0,792 | P15 `cov_real`, P16B | độ thật của support là kênh chính; nhãn teacher đủ tốt |
| 4 | Tri thức teacher chỉ tới private ở nơi có mẫu: KL của Gaussian rơi vào cặp tốc độ (cos 0,09 với lỗi thật); private chững khi teacher còn lên | P20, results §"Vì sao" mục 6 | thêm lượng/trọng số/ưu tiên trong Gaussian vô ích (g512, gk2, xo) |
| 5 | Lỗi dạy được ở cặp teacher đáng tin: posMirror→benign 91 %, constPosOff→benign 96 %, sybil→benign 86 %; benign↔timeDelay là đồng xu (dòng trùng hệt khác nhãn) | results §"Vì sao" mục 2, P14 T5 | lợi tối đa đến từ cặp tin cậy; support "thật" ở cặp đồng xu không thêm gì |
| 6 | Support thật + Project48 +0,0114 so Gaussian, copula +0,0024; Gaussian hóa một nhóm (`snd_kin`, `sess`, `geo`, `prof`) mất 42–75 % lợi | P32 | phần thiếu là cấu trúc liên feature |
| 7 | Bộ sinh "thật hơn" ở input đã FAIL: V4-S (atom/profile/first), V6 G61 (primitive + tái dựng vận tốc), copula vòng 18 | [V4-S](../../lessons/gaussian-support-constraints.md), [V6](../../lessons/geometry-primitive-moments.md), LESSONS | đo độ thật ở input chưa dự báo được utility |

**Kiểm định dữ liệu 22/10 (pool 1.000.369 dòng train fixture, raw sau inverse scaler):** mọi định danh extractor
(vx = spd·sin(hed), vy = spd·cos(hed), dist = hypot(dx, dy), bearing, hed_diff_cos, rel_speed, closing, spd_diff, noise_mag,
implied = dpos/dt, spd_residual = implied − spd, acl_residual = dspd/dt − acl) đúng tới sai số z ≤ 0,016 ở mọi dòng; first = 1 ⇒ 13 feature
sess = 0 và sess_idx = 0 ở 100 % dòng; snd profile luôn one-hot. `dt_skew` và `sender_share` **không** tái dựng được từ giá trị fp16.
Atom tại raw 0 lớn: snd_spd 30 %, vx/vy 27–28 %, dspd 52 %, dhed 51 %, residual 30 %. "Đứng yên ⇒ residual = 0" **chỉ** đúng ở suddenStop
(99,8 %) và dosAttack (66 %); benign đứng yên có residual ≠ 0 (nhiễu GPS) ⇒ cấu trúc atom theo lớp, không phải một chế độ chung.

## 2. Khoảng tri thức còn mở

Dữ kiện 1–3 nói head học từ **đặc trưng private của mẫu**, nhưng mọi đo độ thật trước đây ở input (KS, 1-NN input, teacher agreement).
Chưa ai đo support qua backbone của chính private: mẫu nằm ngoài manifold đặc trưng của dữ liệu thật (precision), bỏ sót vùng client khác
(recall), hay rơi vào vùng của lớp khác (landing — mẫu nhãn c nằm cạnh dòng thật lớp c′). Cũng chưa tách lợi +0,0114 của support thật
theo cặp lỗi và theo độ tin cậy teacher. Thiếu hai mảnh này thì thiết kế bộ sinh là đoán.

## 3. Phép đo (khóa trước chạy; không train, không official test)

Dùng lại nguyên P32: weights xa s42 ep15, fixture 2,5 %, cùng seed sinh support (gaussian 24, real 27, copula 32, hybrid 320+i) ⇒ support
giống hệt; **kiểm**: teacher class_agreement của mọi arm khớp JSON P32, và head P32 đã lưu tái hiện đúng predictions `.npz` trên group.
Teacher = Project48 (calibrated), như các arm P32 đọc.

- **A — kế toán cặp** (head P32 đã lưu, group): với mỗi support S ∈ {copula, real, hyb_G}, so với gaussian: dòng được sửa
  (gaussian sai, S đúng) / bị hỏng (ngược lại) theo cặp (lớp thật, dự đoán gaussian) và theo teacher đúng/sai trên dòng đó; ΔF1 theo lớp ×
  bin tỷ lệ lớp trong dữ liệu local.
- **B — hình học trong không gian đặc trưng private** h (256-d, đầu vào Linear cuối), từng client k × lớp c: tham chiếu = 256 dòng thật
  lớp c của **client khác** (pool trừ dòng own và trừ dòng đã dùng làm support real; seed 33); support = 256 dòng lớp c của bank P32.
  Precision/recall k-NN (k = 3, Kynkäänniemi 2019): precision = phần mẫu nằm trong hợp các cầu bán kính k-NN của tham chiếu; recall = phần
  tham chiếu nằm trong hợp các cầu của mẫu. Landing = 1-NN của mẫu trong tham chiếu đủ 16 lớp (16 × 256) có cùng lớp với nhãn mẫu.
  Thêm `own` (dòng own của client, ≤ 256/lớp) làm mốc: phần vùng client khác mà dữ liệu local đã phủ.
- **C — giải thích**: trên các ô (support ∈ 8 support ≠ gaussian, lớp c), Spearman giữa ΔF1 lớp (group, TB 20 client, so gaussian) và
  Δprecision / Δrecall / Δlanding (cùng trung bình). Báo thêm cùng phép trên tham chiếu group (256 dòng/lớp, seed 34) làm kiểm độ bền.

## 4. Cách đọc (khóa trước chạy)

- **A:** nếu ≥ 50 % số dòng sửa ròng (sửa − hỏng) của real nằm ở dòng teacher đúng trong cặp tin cậy (posMirror, constPosOff, sybil,
  dataReplay với benign) ⇒ bộ sinh phải ưu tiên cấu trúc của các lớp đó (geo/sess); nếu lợi rải đều ⇒ không ưu tiên lớp.
  Lợi tập trung ở lớp local hiếm ⇒ vấn đề phủ lớp; ở lớp local nhiều ⇒ vấn đề lệch receiver.
- **C:** chỉ số có |ρ| lớn nhất và ≥ 0,5 là **mục tiêu tối ưu offline** cho bộ sinh (rẻ hơn harness); chọn hướng sửa:
  precision ⇒ mẫu lệch manifold (cần cấu trúc/định danh/atom); recall ⇒ thiếu mode (cần mixture/điều kiện); landing ⇒ nhãn rơi nhầm vùng
  (cần sửa vị trí lớp, vd. tách mode theo lớp). Không chỉ số nào ≥ 0,5 ⇒ hình học mức lớp không giải thích; ghi giới hạn, không thiết kế
  bộ sinh theo nó.
- Chẩn đoán, không cổng utility; group là development set đã dùng nhiều lần, tham chiếu chính là train client khác để không chọn bằng group.

## 5. P33b — đa dạng client khác hay cân bằng lớp trên dòng thật (khóa ~11:12Z, sau P33 A/B/C, trước chạy)

P33 A cho thấy lợi của support thật nằm ở lớp hiếm local (0,1–10 %), và dòng own cũng nằm đúng manifold (precision 0,79). Chưa đo support
0 byte từ chính dòng own trong harness có teacher (vòng 9 `replay` là nhãn cứng, không teacher). Cùng harness P32, teacher Project48,
thay bank bằng dòng own của chính client k (16.384 dòng đã sample, seed 36+k): `own_bal` 256 dòng/lớp (có lặp nếu thiếu), `own_nat` 4.096 dòng
theo phân phối local. Kiểm tái lập: arm gaussian của client 0 phải cho đúng predictions P32.

- `own_bal` − gaussian ≥ +0,005 và ≥ 50 % khoảng real − gaussian (+0,0114) ⇒ phần lớn lợi là **cân bằng lớp trên dòng thật**; phương án 0 byte
  là ứng viên chính, bộ sinh chỉ đáng làm nếu thắng `own_bal`.
- `own_bal` ≤ gaussian + 0,002 ⇒ lợi cần **dòng của client khác** (đa dạng receiver); bộ sinh phải tái tạo vùng client khác.
- Giữa hai mức ⇒ cả hai góp; mọi bộ sinh sau phải có `own_bal` làm đối chứng và báo phần vượt nó.
- `own_nat` chỉ mô tả (KL thêm trên dòng own đã có trong `kdpx`).

## 6. Kết quả — 22/10 ~11:20Z, local RTX

P33 [JSON](../../methods/full-drift/results/probe33-client-learning-s42.json), [source](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe33.py),
123 s: support/teacher khớp P32 (class_agreement mọi arm bằng nhau), head P32 tái hiện đúng predictions 20 client × 9 support.
P33b [JSON](../../methods/full-drift/results/probe33b-own-support-s42.json) (+ `.npz`), [source](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe33b.py),
126 s: gaussian client 0 tái hiện P32 bit-exact; 80 lượt predictions → confusion → 10 metric khớp.

### A — lợi của support thật nằm ở đâu (Project48, group, TB 20 client)

| lớp | F1 gaussian | Δ real | Δ copula | Δ own_bal |
|---|---:|---:|---:|---:|
| positionMirroring | 0,463 | **+0,076** | +0,019 | −0,006 |
| suddenStop | 0,833 | **+0,054** | +0,022 | +0,003 |
| constantPositionOffset | 0,846 | +0,023 | +0,009 | +0,002 |
| timeDelayAttack | 0,264 | +0,014 | −0,013 | −0,010 |
| dataReplay | 0,712 | +0,013 | +0,004 | +0,003 |
| benign | 0,782 | +0,010 | +0,006 | +0,002 |
| 10 lớp còn lại | 0,64–0,996 | −0,007 … +0,003 | | |

posMirror + suddenStop = 0,130/16 = **71 %** khoảng real − gaussian (+0,0114). Theo tỷ lệ lớp local: bin 0,1–1 % góp 0,0057, 1–10 % 0,0038,
≥ 10 % 0,0009 (fixture không có ô lớp vắng).

Sửa ròng theo cặp (lớp thật → dự đoán sai của head gaussian; tổng 20 client × 65.536 dòng group):

| cặp | copula | real | hyb_snd_kin | hyb_geo | hyb_sess | hyb_prof |
|---|---:|---:|---:|---:|---:|---:|
| posMirror → benign | 109 | **1.894** | **−510** | 706 | 799 | 1.078 |
| suddenStop → benign | 9 | **1.634** | 182 | 153 | **−113** | 775 |
| posMirror → constPosOff | 112 | 1.105 | 39 | 481 | 591 | 197 |
| benign → constPosOff | 1.061 | 1.123 | 1.068 | 1.408 | 426 | 50 |
| benign → timeDelay | 1.507 | 472 | 2.753 | 2.205 | −24 | −1.575 |
| timeDelay → benign | −886 | −13 | −1.317 | −1.117 | 44 | 911 |

- Real: sửa ròng 9.242 = teacher đúng +11.942, teacher sai −2.700; cặp benign với lớp tin cậy có teacher đúng 5.657 = **61 %** (≥ 50 %
  ⇒ theo khóa: ưu tiên cấu trúc các lớp đó). Cặp suddenStop → benign (ngoài danh sách tin cậy đã khóa) đứng thứ hai.
- Hai loại lợi khác nhau: real dạy **recall tấn công** (posMirror/suddenStop bị gọi benign); copula chủ yếu giảm false positive benign → tấn công
  và đổi đồng xu benign ↔ timeDelay (sửa 1.507, hỏng 886).
- Nhóm quyết định theo lớp: Gaussian hóa `snd_kin` đảo dấu posMirror → benign (−510); Gaussian hóa `sess` xóa suddenStop → benign (−113).

### B/C — support trong không gian đặc trưng private (tham chiếu = dòng train client khác; TB 20 client × 16 lớp)

| support | precision | recall | landing |
|---|---:|---:|---:|
| gaussian | 0,487 | 0,135 | 0,423 |
| copula | 0,452 | 0,800 | 0,657 |
| real | 0,863 | 0,861 | 0,824 |
| hyb_snd_kin / rcv_kin / geo | 0,657 / 0,860 / 0,705 | 0,737 / 0,854 / 0,854 | 0,719 / 0,820 / 0,774 |
| hyb_sess / rate / prof | **0,473** / 0,762 / 0,676 | 0,661 / 0,839 / 0,447 | 0,513 / 0,808 / 0,794 |
| own (mốc, dòng của chính client) | 0,793 | 0,801 | 0,819 |

Theo lớp: precision suddenStop gaussian 0,60, **copula 0,35**, real 0,88, own 0,82; posMirror 0,39 / 0,35 / 0,86 / 0,75; landing posMirror
0,11 / 0,23 / 0,52 / 0,57. Tham chiếu group cho cùng thứ tự (bảng trong JSON).

- **C theo khóa: không chỉ số mức lớp nào giải thích ΔF1 lớp** — Spearman precision 0,10, recall 0,32, landing −0,05 (group 0,18 / 0,32 / −0,05;
  ô từng client 0,09 / 0,17 / −0,02). Không thiết kế bộ sinh theo hình học mức lớp.
- Mô tả post-hoc (9 support, không phải cổng): trung bình support thì precision ↔ mean F1 ρ 0,90, landing 0,95, recall 0,68; worst ↔ recall 0,98.
  Mức lớp thất bại vì F1 của benign và các lớp có lợi phụ thuộc mẫu của **lớp kề** (hiệu ứng theo cặp), không chỉ mẫu của chính lớp.

### P33b — dòng own làm support (0 byte)

| support | mean | benign | worst |
|---|---:|---:|---:|
| gaussian (P32) | 0,8285 | 0,7815 | 0,7830 |
| **own_bal** | **0,8269** | 0,7837 | 0,7787 |
| own_nat | 0,8282 | 0,7850 | 0,7807 |
| real (P32) | 0,8398 | 0,7917 | 0,8166 |

Theo khóa: own_bal ≤ gaussian + 0,002 ⇒ lợi cần **dòng của client khác**; cân bằng lớp trên dòng own không thêm gì (posMirror −0,006,
suddenStop +0,003). Dòng own nằm đúng manifold (precision 0,79) nhưng là vùng head đã được neo bằng CE/KD own.

### Kiểm dữ liệu phụ (trước khi thiết kế)

r_dd = ddist + closing·dt (quan hệ vật lý khoảng cách–tốc độ tiếp cận, dòng non-first): median |r_dd| benign 0,22 m, **posMirror 0,28 m**,
constSpeedOffset 3,5, dataReplay 9,4, randomPosOffset 41, reversedHeading 22, suddenStop 3,1. Lời khai bị đối xứng tự nhất quán ⇒ không mã hóa
residual vật lý để bắt posMirror; quan hệ này phân biệt các lớp vốn đã đúng (F1 ≥ 0,94).

## 7. Mô hình học của client sau P33

1. Head private học từ support **chỉ ở nơi support, nhìn qua backbone của chính nó, nằm trên manifold dòng thật của client khác**. Dòng own —
   cả cân bằng lớp — không thêm gì: vùng đó đã được CE/KD own neo.
2. Phần có giá trị là **biến thể ở receiver khác của các lớp tấn công hiếm local giáp benign**: posMirror, suddenStop (71 %), rồi constPosOff,
   dataReplay. Support thật dạy recall tấn công; giảm false positive benign chỉ là phần nhỏ mà copula đã lấy được.
3. Gaussian/copula phủ vùng (copula recall 0,80) nhưng nửa số mẫu lệch manifold (precision 0,45–0,49). Copula tệ hơn Gaussian ở suddenStop
   (0,35 vs 0,60): nhòe atom trong bin phá atom chung "dừng xe" (spd, acl, dspd, residual = 0 cùng lúc, 99,8 % dòng suddenStop non-first).
4. Cấu trúc cần giữ theo lớp: atom chung của `sess`/`snd_kin` cho suddenStop; quan hệ `snd_kin` × `geo` cho posMirror.

## 8. Đề xuất P34 (chưa code; cổng khóa trong protocol P34 trước khi chạy harness)

**Bằng chứng:** §6–7. **Cơ chế:** bộ sinh giữ (a) atom chính xác — khe riêng cho giá trị atom công khai (raw 0 từ scaler), lấy đúng giá trị,
không nhòe; (b) trạng thái first theo lớp — first = 1 ⇒ 14 feature sess = 0 chính xác; copula riêng mỗi trạng thái; (c) định danh extractor —
sinh primitive (spd, heading sin/cos chuẩn hóa, dx, dy, nhiễu vị trí, residual) rồi tính vx, vy, dist, bearing, hed_diff_cos, rel_speed, closing,
spd_diff, noise_mag, implied = spd + residual, dpos = implied·dt, dspd = (acl_residual + acl)·dt; `dt_skew`, `sender_share` giữ làm primitive
(không tái dựng được từ fp16); (d) profile categorical. Mẫu posMirror/suddenStop rơi vào manifold dòng thật client khác ⇒ head nhận ràng buộc
nơi dòng group nằm ⇒ recall hai lớp lên.

- Arm (Project48, harness P32): gaussian, copula, real (mốc); **V4-S** và **V6 G61** port nguyên bản (tiền lệ); own_bal (0 byte);
  mới `cop_atom` (a), `cop_atom_state` (a + b), `pcop` (a + b + c + d) để quy lợi cho từng phần.
- **Dấu hiệu cơ chế (offline, đo bằng hình học P33 trên train client khác, trước harness):** precision private suddenStop ≥ 0,75 và
  posMirror ≥ 0,60 (copula 0,35 / 0,35; real 0,88 / 0,86).
- **Cổng đề xuất:** `pcop` − gaussian ≥ +0,005 mean, benign ≥ −0,005, worst ≥ −0,010; hơn copula, V4-S, G61 và own_bal ≥ +0,003;
  F1 posMirror + suddenStop đạt ≥ ½ lợi của real ở hai lớp đó.
- **Falsifier:** precision đạt dấu hiệu mà utility không qua ⇒ precision không là nguyên nhân đủ; precision không lên ⇒ cấu trúc viết tay không
  đủ. Cả hai ⇒ chuyển capacity-oracle của bộ học mật độ có điều kiện ([identifiability §7](synthetic-transfer-identifiability-2026-10-22.md)),
  không tune P34.
- R3: chỉ histogram/count/tổng normal-score cộng được dưới SecureSum; R2: mẫu ở input; R1: báo byte, nén sau (copula P32 ~1,9 MB/client).
- Giới hạn: frozen-head, một seed, fixture 2,5 %; online ep15/s43/horizon riêng.
