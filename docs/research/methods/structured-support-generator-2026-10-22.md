# P34 — bộ sinh giữ atom, trạng thái first và định danh extractor (frozen-head, Project48)

22/10/2026 ~11:30Z; owner duyệt "thực hiện ngay" đề xuất [P33 §8](client-learning-diagnosis-2026-10-22.md). **Protocol, arm và cổng dưới đây
khóa trước khi chạy**; không tune sau khi thấy kết quả.

## Bằng chứng → cơ chế

P33: 71 % lợi support thật (+0,0114) là recall posMirror + suddenStop vs benign ở receiver lạ; dòng own không thêm gì; Gaussian/copula nửa số mẫu
lệch manifold đặc trưng private; copula nhòe atom chung "dừng xe" nên suddenStop lệch hơn Gaussian (precision 0,35 vs 0,60); Gaussian hóa `snd_kin`
đảo dấu sửa posMirror, `sess` xóa sửa suddenStop. Kiểm định dữ liệu: mọi định danh extractor đúng ở mọi dòng; first = 1 ⇒ 14 feature sess = 0.
Cơ chế: mẫu giữ (a) atom chính xác, (b) trạng thái first, (c) định danh extractor, (d) profile categorical rơi vào manifold dòng thật client khác ⇒
head nhận ràng buộc nơi dòng group nằm ⇒ recall posMirror/suddenStop lên.

## Harness và support

Harness P32 nguyên vẹn (xa s42 ep15, fixture 2,5 %, 20 client × 16.384 dòng own, Linear cuối 1.000 bước, teacher Project48, bank 4.096 dòng,
lịch lớp đều seed 24, seed adapt 2400+k). Mọi thống kê từ cùng pool 20 × 16.384 dòng (= tổng SecureSum). Mốc dùng lại kết quả đã audit cùng harness:
gaussian/copula/real (P32), own_bal (P33b); kiểm tái lập: gaussian client 0 phải ra đúng predictions P32.

| arm | cách sinh | vai trò |
|---|---|---|
| `v4s` | port nguyên bản `.compat/mh/v4_support.py` arm cấu trúc: atom/first/profile, Gaussian **diag** phần không atom, 13 feature sess điền khi first | tiền lệ V4-S |
| `g61` | port `geometry_moments.py` G61 joint pool: Gaussian full-cov 61 chiều theo (lớp × trạng thái first × profile), atom sess theo trạng thái, tái dựng vx/vy/hed_diff_cos | tiền lệ V6 |
| `cop_atom` | copula P32 (256 bin asinh) + **(a)** khe atom riêng tại raw 0 mỗi feature (lưới bin dời để atom nằm trên mép bin), lấy đúng giá trị | thêm (a) |
| `cop_atom_state` | (a) + **(b)** copula riêng theo lớp × first; first = 1 ⇒ 14 feature sess = raw 0 | thêm (b) |
| `pcop_noid` | (a) + (b) + **(d)** profile sender thành một biến categorical trong copula | thêm (d) |
| **`pcop`** | (a) + (b) + (d) + **(c)** copula chỉ trên 34 primitive; tính vx, vy (hai phía, heading chuẩn hóa), hed_diff_cos, dist, bearing, rel_speed, closing, spd_diff, snd_noise_mag; non-first: implied = max(spd + residual, 0), dpos = implied·dt, dspd = (acl_residual + acl)·dt, dt_send = dt − dt_skew | **arm chính** |

Port tiền lệ chỉ đổi cho m4b: bỏ nhóm profile receiver (đặt 0 bởi m4b), cột m4b giữ 0. Nhãn bank giống P32; seed sinh 41–46 theo thứ tự bảng.

## Cổng (khóa)

- **G1 utility:** `pcop` − gaussian ≥ **+0,005** mean-private group; benign ≥ −0,005; worst ≥ −0,010.
- **G2 vượt tiền lệ / 0 byte:** `pcop` − max(copula, `v4s`, `g61`, own_bal) ≥ **+0,003** mean.
- **PASS = G1 ∧ G2.** Mô tả kèm (không cổng): G3 ΔF1 posMirror + suddenStop của `pcop` ≥ ½ của real (≥ +0,065 tổng); dấu hiệu cơ chế offline:
  precision đặc trưng private suddenStop ≥ 0,75, posMirror ≥ 0,60 (tham chiếu, seed như P33); quy lợi từng phần theo chuỗi arm; byte bảng.
- **Falsifier:** precision đạt mà G1 không qua ⇒ precision không là nguyên nhân đủ; precision không lên ⇒ cấu trúc viết tay không đủ. Cả hai ⇒
  chuyển capacity-oracle bộ học mật độ có điều kiện ([identifiability §7](synthetic-transfer-identifiability-2026-10-22.md)); không tune bin/bước/seed.
- **PASS ⇒** xác nhận Kaggle T4 hai seed trước khi thiết kế online + kế toán R1. Không official test; old/group là development set.

## Kết quả — 22/10 ~11:45Z, local RTX, 465 s — **FAIL**

[JSON](../../methods/full-drift/results/probe34-structured-support-s42.json) (+ `.npz` predictions, `.pt` heads),
[source](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe34.py). Gaussian client 0 tái lập P32 bit-exact; **240 lượt predictions →
confusion → 10 metric khớp**; định danh `pcop` lũy đẳng (sai số 5e-7 z); kiểm riêng: `derive` trên dòng thật p99 lệch 0,0013 z (không lỗi code).

| arm | mean | Δ vs gaussian | benign | worst | teacher đồng ý | precision / recall / landing (đặc trưng private) | byte bảng lên / xuống |
|---|---:|---:|---:|---:|---:|---|---:|
| gaussian (P32) | 0,8285 | — | 0,7815 | 0,7830 | 0,444 | 0,49 / 0,14 / 0,42 | — |
| copula (P32) | 0,8308 | +0,0024 | 0,7875 | 0,8048 | 0,649 | 0,45 / 0,80 / 0,66 | 963 / 960 kB |
| own_bal (P33b) | 0,8269 | −0,0015 | 0,7837 | 0,7787 | — | — | 0 |
| `v4s` | 0,8284 | −0,0001 | 0,7815 | 0,7858 | 0,407 | 0,35 / 0,45 / 0,39 | 13 / 11 kB |
| `g61` | 0,8284 | −0,0001 | 0,7836 | 0,7890 | 0,524 | 0,46 / 0,57 / 0,50 | 395 / 211 kB |
| `cop_atom` | 0,8305 | +0,0020 | 0,7876 | 0,8035 | 0,621 | 0,44 / 0,80 / 0,62 | 967 / 963 kB |
| `cop_atom_state` | 0,8306 | +0,0021 | 0,7885 | 0,8052 | 0,634 | 0,47 / 0,81 / 0,64 | 1,89 / 1,89 MB |
| `pcop_noid` | 0,8310 | +0,0025 | 0,7883 | 0,8065 | 0,629 | 0,47 / 0,83 / 0,63 | 1,81 / 1,81 MB |
| **`pcop`** | **0,8308** | **+0,0023** | 0,7865 | 0,8000 | 0,601 | 0,41 / 0,81 / 0,60 | 1,20 / 1,19 MB |
| real (P32, oracle) | 0,8398 | +0,0114 | 0,7917 | 0,8166 | 0,837 | 0,86 / 0,86 / 0,82 | — |

ΔF1 theo lớp so gaussian (TB 20 client) và precision đặc trưng private:

| lớp | v4s | g61 | cop_atom | cop_atom_state | pcop_noid | pcop | real |
|---|---:|---:|---:|---:|---:|---:|---:|
| positionMirroring ΔF1 | −0,004 | −0,007 | +0,018 | +0,006 | +0,020 | −0,001 | **+0,076** |
| suddenStop ΔF1 | +0,005 | +0,012 | +0,018 | +0,025 | +0,024 | +0,026 | **+0,054** |
| posMirror precision | 0,22 | 0,35 | 0,34 | 0,34 | 0,36 | **0,22** | 0,86 |
| suddenStop precision | 0,39 | 0,49 | 0,57 | 0,54 | 0,59 | 0,64 | 0,88 |

- **Cổng:** G1 FAIL (+0,0023 < +0,005; benign +0,0049, worst +0,0170 đạt); G2 FAIL (−0,00001 so max(copula, v4s, g61, own_bal) = copula).
  G3 mô tả: posMirror + suddenStop +0,025 < ½ real 0,065. Dấu hiệu cơ chế FAIL: precision suddenStop 0,64 < 0,75, posMirror 0,22 < 0,60.
- **Falsifier đã khóa trúng nhánh "precision không lên ⇒ cấu trúc viết tay không đủ".** Không tune bin/seed/primitive.
- **Quy lợi từng phần (mô tả):** atom chính xác −0,0004 so copula (precision suddenStop 0,35 → 0,57 nhưng F1 không thêm); trạng thái first +0,0001;
  profile categorical +0,0004; định danh extractor −0,0002 và **posMirror −0,021**. Mọi biến thể copula nằm trong +0,0020…+0,0025 ⇒ trần là phụ thuộc
  Gaussian (tương quan normal-score) của copula, không phải atom/trạng thái/profile.
- **Vì sao định danh làm hại posMirror:** primitive do copula sinh có phụ thuộc sai (heading × bearing × vận tốc theo hình học đường); tính lại
  closing/rel_speed/dist bằng công thức lan cái sai đó vào các feature mà copula vốn có biên đúng (pcop_noid). Định danh đúng trên primitive sai
  không làm mẫu thật hơn trong mắt private.
- V4-S và G61 dưới Project48 vẫn ≈ Gaussian (−0,0001): tiền lệ FAIL lặp lại cả khi teacher tốt hơn.
- Ghi chú đo: `atom_rate` của `f_snd_vx`/`f_sess_dspd` ở `pcop` thấp vì giá trị tính lại khác atom fp16 của fixture ~1e-7 z (làm tròn), không
  ảnh hưởng model.

**Bước tiếp theo protocol:** capacity-oracle của bộ học mật độ có điều kiện, phi tuyến
([identifiability §7](synthetic-transfer-identifiability-2026-10-22.md)) — fit trên pool (R3-ineligible, chỉ chẩn đoán) để biết một mô hình học được
có tái tạo được posMirror/suddenStop (đặc trưng private precision, ΔF1) trước khi thiết kế bản federated. Chưa code; cần owner duyệt.
