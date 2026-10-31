# P36 — nguồn tri thức × support so với phương pháp hiện hành, hai seed; GMM chỉ cho lớp tấn công (frozen-head)

22/10/2026 ~13:50Z; owner: "thử nghiệm cả hai phương án" của [P35 §Kết luận](support-capacity-oracle-2026-10-22.md): (a) nguồn tri thức/online,
(b) GMM chỉ cho lớp tấn công hiếm + kế toán R1. **Protocol, arm và cổng dưới đây khóa trước khi chạy.**

## Bằng chứng → giả thuyết

- Các probe trước khóa cổng so **Project48 + gaussian** để cô lập support. Với quyết định phương pháp, đối chứng đúng là **phương pháp hiện hành**
  (teacher gốc + gaussian, 0,8231 ở harness s42). Số đã audit: Project48 + gaussian +0,0054 nhưng worst −0,017 (P24 FAIL); Project48 + copula
  0,8308 = +0,0077 mean, benign +0,010, worst +0,005 (P32); Project48 + gmm8 0,8329 (P35). Teacher tốt hơn hại client tệ nhất khi support là
  Gaussian; support thật hơn bù lại. **H36a:** Project48 + copula qua cổng so phương pháp hiện hành trên **cả hai seed** ⇒ đáng chạy online.
- P35: lợi nằm ở mẫu thật của lớp tấn công hiếm; benign giả không hại; GMM tốt nhất (+0,0044 so Project48 + gaussian) nhưng full-cov EM nhiều
  vòng vượt R1. **H36b:** GMM chỉ cho 5 lớp tấn công S5 = {positionMirroring, suddenStop, timeDelayAttack, constantPositionOffset, dataReplay}
  giữ phần lợi của GMM; bản diagonal (ít byte) giữ được phần lớn.

## Harness

P32 nguyên vẹn trên hai bộ weights đã audit cùng phương pháp `q3_…_la_kdpx_zl` ep15: **s42 = xa** (`fd-s42xa-20261015-r22-v1`), **s43 = ya-v2**
(`fd-s43ya-20261015-r23-v2`). Pool dòng (seed 22), bank gaussian (seed 24), copula (seed 32), gmm8 (P35 seed 53, cùng thứ tự fit) không phụ thuộc
weights ⇒ giống hệt ở hai seed. Teacher Project48 hiệu chỉnh lại theo snapshot của từng seed như P32. Kiểm: s42 các arm trùng P32/P35 phải ra
đúng mean đã audit (orig_gauss, p48_gauss, p48_cop, p48_gmm8).

| arm | teacher | support |
|---|---|---|
| `orig_gauss` | gốc | gaussian — **đối chứng = phương pháp hiện hành** |
| `orig_cop` | gốc | copula (mô tả) |
| `p48_gauss` | Project48 | gaussian (mô tả) |
| **`p48_cop`** | Project48 | copula — **arm chính (a)** |
| `p48_gmm8` | Project48 | GMM8 full mọi lớp (trần P35, mô tả) |
| `p48_gmm8_att` | Project48 | GMM8 full cho S5, gaussian cho lớp khác (cùng dòng với gmm8/gaussian) |
| `p48_gmm8d_att` | Project48 | GMM8 **diagonal** cho S5 (fit mới, seed 61, EM như P35), gaussian lớp khác |
| **`p48_copgmm8d`** | Project48 | GMM8 diagonal cho S5, **copula** cho lớp khác — **arm chính (b)** |

## Cổng (khóa)

- **(a) PASS** khi trên **cả s42 và s43**: `p48_cop` − `orig_gauss` ≥ +0,005 mean, benign ≥ −0,005, worst ≥ −0,010. PASS ⇒ dựng arm online
  (token Project48 per round + `cop` one-shot, source tag mới) và chạy **Kaggle ep15 s42** so xa với cổng online đã có (+0,010 mean, benign ≥ −0,005,
  worst ≥ −0,010); kế toán byte R1 báo kèm. FAIL ⇒ không launch online.
- **(b) PASS** khi trên cả hai seed: `p48_copgmm8d` − `p48_cop` ≥ +0,003 mean, benign/worst không kém `p48_cop` quá −0,005/−0,010, **và** byte
  GMM diagonal (10 bước k-means + số vòng EM thực, mỗi vòng thống kê lên + tham số xuống, fp32) ≤ **0,5 MB/client**. FAIL ⇒ đóng hướng GMM.
- **Kế toán R1 cả stack (báo, không là cổng của P36):** Project48 14 lần hiệu chỉnh/15 round (lên 78.400 B + xuống 16.448 B mỗi lần) + copula mã
  không mất (đếm uint16 lên, CDF fp32 xuống, tổng normal-score và Cholesky fp32) + GMM; so 1,5 × xa ở 15 round (7,77 MB) và 50 round.
  *Sửa trước khi chạy (~14:00Z):* bản đầu đưa Project48 + copula vào điều kiện byte của (b); hai thành phần đó đã 2,82 MB > 2,59 MB ngân sách thêm ở
  15 round nên điều kiện FAIL sẵn bất kể GMM. R1 của stack là việc riêng (bin, mã thưa, tần suất hiệu chỉnh) trước khi chốt online dài.
- Mô tả: `p48_gmm8_att` − `p48_gmm8` (nhắm lớp có giữ lợi), `p48_gmm8d_att` − `p48_gmm8_att` (diag vs full), byte từng thành phần.
- Không official test; old/group là development set; không đổi K/seed/số vòng sau khi thấy kết quả.

## Kết quả — 22/10 ~14:20Z, local RTX, 901 s

[JSON](../../methods/full-drift/results/probe36-teacher-support-2seed.json) (+ `.npz`, `.pt`), [source](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe36.py).
**640 lượt predictions → confusion → 10 metric khớp**; s42 tái lập đúng mean đã audit của `orig_gauss`, `p48_gauss`, `p48_cop` (P32) và `p48_gmm8` (P35).
Δ so **phương pháp hiện hành** (`orig_gauss`: s42 0,8231, s43 0,8211):

| arm | s42 mean / benign / worst | s43 mean / benign / worst | s42 / s43 Δ posMirror, suddenStop |
|---|---|---|---|
| `orig_cop` | +0,0001 / +0,0025 / +0,0054 | −0,0015 / +0,0046 / +0,0027 | −0,010, −0,000 / −0,004, −0,017 |
| `p48_gauss` | +0,0053 / +0,0042 / **−0,0166** | +0,0011 / −0,0029 / **−0,0251** | +0,055, +0,031 / +0,081, +0,051 |
| **`p48_cop`** | **+0,0077 / +0,0101 / +0,0052** | **+0,0051 / +0,0075 / −0,0019** | +0,074, +0,053 / +0,094, +0,066 |
| `p48_gmm8` | +0,0098 / +0,0141 / +0,0084 | +0,0092 / +0,0147 / +0,0105 | +0,077, +0,064 / +0,111, +0,093 |
| `p48_gmm8_att` | +0,0100 / +0,0130 / +0,0088 | +0,0087 / +0,0142 / +0,0067 | +0,074, +0,063 / +0,110, +0,090 |
| `p48_gmm8d_att` | +0,0086 / +0,0108 / +0,0071 | +0,0052 / +0,0088 / −0,0004 | +0,067, +0,058 / +0,085, +0,083 |
| `p48_copgmm8d` | +0,0081 / +0,0110 / +0,0066 | +0,0059 / +0,0094 / +0,0052 | +0,071, +0,057 / +0,093, +0,085 |

- **(a) PASS** cả hai seed (s43 sát ngưỡng +0,0051). Teacher Project48 một mình hại client tệ nhất ở cả hai seed; copula sửa phần đó, còn teacher
  gốc + copula không thêm gì ⇒ lợi là **tương tác teacher tốt hơn × support thật hơn**. Theo khóa: dựng arm online `cvp` + `cop`, Kaggle ep15 s42 so xa.
- **(b) FAIL:** `p48_copgmm8d` − `p48_cop` +0,0004 / +0,0008 < +0,003; byte EM diag 3,32 MB > 0,5 MB (k-means + tới 92 vòng EM). Đóng hướng GMM.
  Mô tả: GMM full chỉ cho S5 giữ trọn lợi GMM mọi lớp (khớp P35: benign giả không hại) nhưng EM full 42,9 MB — không chở được qua R1.
- **R1 stack `p48_cop`** (Project48 14 / 49 lần × 94.848 B + copula): **sửa ~14:55Z** — khóa `bytes.stack_*.r50` trong JSON ngoại suy byte xa
  tuyến tính (17,3 MB) là sai; theo `result.json` xa đã audit: bootstrap 1,81 MB + q3 223 kB/round ⇒ 5,16 MB / 15 round, 12,96 MB / 50 round.
  Copula mã không mất 1,49 MB: **×1,55 (15 round), ×1,47 (50 round)**; copula fp32 như `fd.py` (1,92 MB): ×1,63 / ×1,51. Round có hiệu chỉnh
  = 223 + 95 kB > 231 kB/contact ⇒ thống kê cần contact riêng; copula bootstrap cần ≥ 7 contact. Online screen chạy đúng cấu hình harness
  (copula 256 bin fp32, hiệu chỉnh mỗi round); nén (bin, mã thưa, 16-bit, tần suất hiệu chỉnh) là bước R1 bắt buộc trước run dài.

## Online — vòng 26 FAIL (22/10 ~19:15Z)

`cvp` + `cop` Kaggle ep15 s42: private head group 0,8249 vs xa 0,8246 (**Δ +0,0003**, benign +0,0053, worst +0,0026), 8,20 MB/client (×1,59).
Lợi frozen-head (a) không chuyển online: teacher hiệu chỉnh chỉ hơn gốc ở ep2–4; posMirror +0,120 giữ nhưng suddenStop −0,025, timeDelay −0,078.
Chi tiết: [results §Vòng 26](../../methods/full-drift/results.md).
