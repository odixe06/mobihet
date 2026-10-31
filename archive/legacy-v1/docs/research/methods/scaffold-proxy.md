# SCAFFOLD trên proxy — control variate sửa hướng cập nhật

**Trạng thái (17/10 ~04:40Z):** FAIL (s42) — dừng.

## Cơ chế
Proxy chuyển SGD thuần (Option II giả định SGD), private giữ AdamW. Gradient proxy `g + c − c_i` (cộng sau clip);
`c_i⁺ = c_i − c + (w_base − y_i)/(Kη)`, K = bước áp thật (bỏ bước AMP skip); server `c ← c + mean Δc_i`.

## Trao đổi
Thêm `c` xuống và `Δc_i` lên: FP32 ≈ 1,58 MB/chiều. Giai đoạn A không dùng byte để loại.

## Thiết kế kiểm
`sgd{1,3,10}` (c=0, lr 0,01/0,03/0,1 × cosine) và `scaf{1,3,10}`. LR chọn theo **old-holdout proxy** của control sgd
(khóa trước). Cổng: scaf vs sgd cùng LR (cô lập correction) và scaf vs `fd_dml_m4b` (thực dụng).

## Bằng chứng
Audited ([kết quả vòng 8–10](../../methods/full-drift/results.md)). LR* = 0,1 (old-holdout proxy sgd 0,7134 / 0,7585 / 0,8011). `scaf10` vs `sgd10`
private gp −0,010 (benign −0,039, client tệ nhất −0,125); vs `fd_dml_m4b` −0,007 (tệ nhất −0,149) ⇒ FAIL. LR khác:
scaf1/3 −0,128/−0,058, sgd1/3/10 −0,065/−0,028/+0,003 so m4b. Byte thêm ≈181 MiB mỗi chiều/6 epoch.
 Kiểm local: scaf ep1 = sgd ep1 (control 0), đại số Option II client/server PASS.

## Nguồn
Karimireddy et al., *SCAFFOLD*, ICML 2020, Eq. 3–5 [PMLR](https://proceedings.mlr.press/v119/karimireddy20a.html). [Plan §5.3](../../archive/research/algorithm-improvement-plan-2026-10-17.md).
