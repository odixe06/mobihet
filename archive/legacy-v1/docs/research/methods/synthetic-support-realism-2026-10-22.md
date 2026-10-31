# P32 — support thật hơn có cần teacher đủ đúng? Copula × Project48 và ablation độ thật theo nhóm feature

22/10/2026 ~10:30Z; owner cho tiếp nghiên cứu synthetic, được thử local và Kaggle ngắn. Đọc trước
[P29–31](synthetic-gradient-diagnosis-2026-10-22.md), [P23–28](synthetic-learning-recovery-2026-10-22.md),
[build cũ/veto](synthetic-build-history-audit-2026-10-22.md). **Protocol và cổng dưới đây ghi trước khi chạy.**

## Bằng chứng dẫn tới phép thử

- **Gaussian sai chỗ vì nhiễu liên tục trên feature thật là atom** (probe 20): bất đồng teacher trên mẫu Gaussian rơi vào cặp tốc độ
  (constantSpeedOffset→randomSpeedOffset), cos với lỗi thật 0,09; thêm nhiễu covariance vào chính dòng own cũng kéo về randomSpeedOffset,
  thay feature bằng giá trị copula (biên thật) thì không. P29 đo cùng hiện tượng ở target mass: Gaussian dồn 22,9 % vào randomSpeedOffset (thật 5,5 %).
- **Copula sửa biên nhưng chỉ từng thử với teacher gốc:** vòng 18 `cop` online +0,0031 (FAIL). Probe 20: copula align ×3 Gaussian.
- **Support thật chỉ có giá khi teacher đúng hơn** (P27, frozen-head, 20 client): support thật − Gaussian = +0,0041 với teacher gốc,
  **+0,0129** với Project48 (worst +0,035). Lợi +0,003 của cop với teacher gốc cùng cỡ trần +0,004 của chính support thật ⇒ thất bại của cop
  chưa bác độ thật của support; nó bị trần teacher chặn.
- **Teacher tốt hơn mà support Gaussian thì hại client tệ nhất** (P24/P26: Project48/Gaussian worst −0,017; client18 benign→suddenStop ×200).

Chưa ai đo copula với teacher Project48, cũng chưa đo phần nào của support thật tạo ra +0,0129.

## Giả thuyết

- **H32a (bổ trợ):** dưới Project48, copula (biên lớp thật kể cả atom, tương quan normal-score; cộng được dưới SecureSum, không record rời
  client) thu phần lớn khoảng support thật − Gaussian và sửa worst-client. Cơ chế: bỏ nhiễu ở feature atom ⇒ target mass không dồn vào lớp
  tốc độ, KL tổng hợp đặt gradient vào vùng dữ liệu thật mà teacher mới đúng.
- **H32b (định vị):** phần mất của Gaussian tập trung ở vài nhóm feature. Thay riêng nhóm G của dòng thật bằng Gaussian có điều kiện
  (theo phần còn lại, cùng moment lớp) làm mất phần lớn lợi ⇒ bộ sinh chỉ cần cấu trúc chính xác ở G.

## Protocol (khóa trước chạy)

Harness P24 nguyên vẹn: weights xa s42 ep15 (audited), fixture 2,5 %, 20 client × 16.384 dòng own (seed 22), backbone/BN private đóng băng,
Linear cuối 1.000 bước AdamW, 512 own + 256 synthetic/bước, cùng seed `2400+k`, own CE logit-adjusted + KL real + KL synthetic,
trọng số `1/(CE+proxy_CE)`. Bank 4.096 dòng, lịch lớp đều (seed 24) như gkd; teacher gốc và Project48 như P24 (cùng basis/bank/fit).
Mọi support ước lượng từ **cùng pool 20 × 16.384 dòng đã sample** (như input moment của Gaussian P24), 13 feature m4b = 0.

| support | cách tạo | dùng được theo R3? |
|---|---|---|
| gaussian | `virtual(im, ic)` như P24 (đối chứng, tái lập P24) | có (moment gộp) |
| copula | port đúng `fd.make_copula/draw_copula` vòng 18: histogram 256 bin asinh(z) theo lớp × feature active + tương quan normal-score theo lớp, nearest-PD | có (hai pha SecureSum, one-shot) |
| real | dòng thật cùng lớp từ pool (seed 27) | **không** — oracle chẩn đoán |
| hyb_G | dòng thật của `real`, nhóm G thay bằng N(μ_G + Σ_GR Σ_RR⁻¹(x_R − μ_R), Σ_G·R) của lớp | **không** — ablation chẩn đoán |

Nhóm active (53 feature): `snd_kin` (pos/spd/acl/hed noise, spd, acl, vx, vy, hed sin/cos, noise_mag của sender), `rcv_kin` (spd, acl, vx, vy,
hed sin/cos của receiver), `geo` (dx, dy, dist, bearing, hed_diff, rel/closing speed, spd_diff, dist_road_edge), `sess` (delay_s, first_in_session,
mọi `f_sess_*`), `rate` (alias_age, rx/sender rate, sender_share), `prof` (profile sender one-hot).

Arm: {gốc, Project48} × {gaussian, copula, real} và Project48 × hyb_G cho 6 nhóm (12 arm), đủ 20 client, chấm old/group.
Predictions/heads lưu; mỗi kết quả kiểm lại predictions → confusion → 10 metric. Không official test, không chọn bước/trọng số theo holdout.

## Cổng và cách đọc (khóa trước chạy)

- **Hợp lệ:** real/P48 − gaussian/P48 ≥ +0,010 mean group (tái lập hướng P27 trên pool này). Không đạt ⇒ không kết luận H32a/H32b.
- **H32a PASS** khi copula/P48 − gaussian/P48 ≥ **+0,005** mean-private group, benign ≥ −0,005, worst ≥ −0,010.
  Báo tỷ lệ thu hồi (copula − gaussian)/(real − gaussian).
- **Bổ trợ** (mô tả): (copula − gaussian) dưới P48 so dưới teacher gốc; ≤ +0,002 chênh ⇒ không ủng hộ "teacher chặn trần".
- **H32b:** mất do nhóm G = real/P48 − hyb_G/P48; nhóm mất ≥ 0,003 gọi là quan trọng; tổng các nhóm không buộc cộng tuyến tính.
- **Falsifier:** H32a sai nếu copula < +0,005 khi khoảng real hợp lệ ⇒ biên thật không đủ, phần thiếu là cấu trúc liên feature (H32b chỉ nhóm).
  Không tune bin/bước/trọng số để cứu copula; không nới cổng.
- **PASS ⇒** xác nhận T4 hai seed trên Kaggle (notebook ngắn kiểu `fd-ccvr-probes`), rồi mới thiết kế arm online (copula one-shot + Project48)
  với kế toán R1 (copula one-shot nhiều byte: histogram 16 × 53 × 256; phải nén/thưa và chia contact). Online vẫn cần cổng ep15 +0,010,
  s43, horizon và R2 riêng. **FAIL ⇒** dùng H32b chọn bộ sinh có cấu trúc cho nhóm quan trọng, protocol mới.

## Kết quả — 22/10 ~10:40Z, local RTX, 620 s

[JSON](../../methods/full-drift/results/probe32-support-realism-s42.json) (+ `.npz` predictions, `.pt` heads),
[source](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe32.py). Đủ 20 client; **480 lượt predictions → confusion → 10 metric
khớp JSON** (kiểm trong script). Mean = macro-F1 group trung bình 20 private; benign = F1 benign trung bình; worst = client thấp nhất.

| teacher / support | mean | benign | worst | Δ mean / benign / worst vs gaussian cùng teacher |
|---|---:|---:|---:|---|
| gốc / gaussian | 0,8231 | 0,7773 | 0,7996 | — |
| gốc / copula | 0,8232 | 0,7798 | 0,8050 | +0,0001 / +0,0025 / +0,0054 |
| gốc / real (oracle) | 0,8272 | 0,7853 | 0,8130 | +0,0041 / +0,0080 / +0,0134 |
| Project48 / gaussian | 0,8285 | 0,7815 | 0,7830 | — |
| **Project48 / copula** | **0,8308** | 0,7875 | 0,8048 | **+0,0024** / +0,0059 / **+0,0218** |
| Project48 / real (oracle) | 0,8398 | 0,7917 | 0,8166 | +0,0114 / +0,0102 / +0,0336 |

Ablation (Project48; dòng thật, chỉ Gaussian hóa nhóm G theo điều kiện phần còn lại):

| nhóm G Gaussian hóa | mean | mất so real | teacher đồng ý trên mẫu | target mass nổi bật |
|---|---:|---:|---:|---|
| `snd_kin` (12) | 0,8313 | **0,0085** | 0,730 | suddenStop 0,149 |
| `sess` (16) | 0,8317 | **0,0081** | 0,534 | randomSpeedOffset 0,223 (≈ Gaussian 0,229) |
| `geo` (10) | 0,8345 | **0,0053** | 0,757 | dataReplay 0,123 |
| `prof` (3, one-hot) | 0,8350 | **0,0048** | 0,835 | — |
| `rate` (6) | 0,8379 | 0,0020 | 0,836 | — |
| `rcv_kin` (6) | 0,8383 | 0,0015 | 0,834 | — |
| (real / gaussian / copula) | | | 0,837 / 0,444 / 0,649 | rSpdOff 0,054 / 0,229 / 0,081 |

- **Hợp lệ:** real − gaussian (Project48) +0,0114 ≥ +0,010 ⇒ tái lập hướng P27 trên pool này.
- **H32a FAIL theo cổng khóa:** copula +0,0024 < +0,005 (thu hồi 21 % khoảng mean). Nhưng copula sửa **65 % khoảng worst-client** (+0,0218 /
  +0,0336) và benign +0,0059: biên thật đã chữa phần false-positive làm Project48/Gaussian hại client tệ nhất (P24). Không tune bin/bước.
- **Bổ trợ yếu:** copula − gaussian = +0,0001 (gốc) vs +0,0024 (Project48), chênh +0,0023 — vừa qua ngưỡng mô tả 0,002; support thật cũng
  +0,0041 vs +0,0114. Teacher tốt hơn làm độ thật có giá hơn, nhưng copula chưa đủ thật.
- **H32b — phần thiếu là cấu trúc liên feature, không phải biên:** 4 nhóm quan trọng (snd_kin, sess, geo, prof); Gaussian hóa **một** nhóm đã
  mất 42–75 % lợi; tổng mất 4 nhóm (0,027) gấp 2,4 lần cả khoảng (0,0114) ⇒ không cộng tuyến tính, lợi cần các nhóm cùng thật đồng thời.
  `sess` một mình tái hiện gần đúng bệnh randomSpeedOffset của Gaussian; `snd_kin` tạo false positive suddenStop; one-hot profile Gaussian hóa
  (0/1 thành số thực) cũng mất 0,0048.
- Giới hạn: frozen-head, một seed, fixture 2,5 %, pool equal-per-client; hybrid dùng Gaussian có điều kiện (linear) — mất do nhóm G gồm cả mất
  quan hệ phi tuyến với phần còn lại. Copula one-shot 963 kB lên + 960 kB xuống/client (codec fp32, chưa nén thưa) — vượt R1 nếu không nén.

**Hướng tiếp (đề xuất, chưa code):** bộ sinh giữ cấu trúc cho 4 nhóm quan trọng — phần rời rạc lấy đúng (profile one-hot theo categorical lớp;
`first_in_session` kèm điền atom cả nhóm `sess`; chế độ xe đứng yên speed/dpos/residual = 0), phần liên tục sinh **theo từng chế độ** để giữ
quan hệ tất định (vd. vx, vy với speed/heading; dist với dx, dy). Cổng cùng harness: ≥ +0,005 mean so gaussian/Project48, benign ≥ −0,005,
worst ≥ −0,010; dấu hiệu cơ chế: teacher đồng ý trên mẫu tiến gần 0,837, mass randomSpeedOffset/suddenStop/dataReplay về mức thật.

**Tiền lệ phải dùng lại, không đặt tên mới — V4-S** ([bài học](../../lessons/gaussian-support-constraints.md),
[report](../../archive/methods/gaussian-sampling/builds/support-mala/results/report.md)): sampler analytic giữ binary/profile categorical và
first-session zero-imputation (`core/feature_support_schema.py`, `.compat/mh/v4_support.py`) — pilot r20 hai seed `s_structural` 0,579/0,582
vs `diag` 0,581/0,590 ⇒ FAIL. Điều kiện khác: private Gaussian kiểu cũ, teacher gốc, luật gp, chưa frz/gkd; P27/P32 cho thấy giá trị độ thật
phụ thuộc teacher, nên V4-S chưa bác hướng này dưới Project48. P33 phải có **arm V4-S nguyên bản** (đọc source, port đúng) và chỉ thêm phần V4-S
chưa có (chế độ xe đứng yên, quan hệ tất định giữa feature, sinh liên tục theo chế độ) như contrast riêng so với arm V4-S.

**Cập nhật 22/10 ~11:20Z:** trước khi code, owner yêu cầu phân tích cách client học ⇒ P33 thành chẩn đoán ([client learning](client-learning-diagnosis-2026-10-22.md));
bộ sinh giữ cấu trúc ở trên đổi tên **P34**, đề xuất cụ thể ở P33 §8 (V4-S vẫn là arm nguyên bản, thêm G61 và own_bal).
