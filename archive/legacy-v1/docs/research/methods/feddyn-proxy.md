# FedDyn trên proxy — sửa drift bằng lịch sử

**Trạng thái (17/10 ~04:40Z):** FAIL cả hai seed — dừng.

## Cơ chế
Chỉ tham số trainable của proxy: local `F_i(w) − <a_i,w> + α/2‖w − w_t‖²` (gradient cộng trước clip),
`a_i ← a_i − α(w_i − w_t)`; server `h ← h − α·mean(w_i − w_t)`, `w ← mean(w_i) − h/α` (chỉ từ tổng cohort).
BN buffers trung bình như thường. Private giữ loss DML-m4b.

## Trao đổi
Như DML (a_i local, h server, không gửi thêm) — ưu điểm R1 so SCAFFOLD.

## Bằng chứng
Audited ([kết quả vòng 8–10](../../methods/full-drift/results.md)): vs `fd_dml_m4b` private gp −0,032 (s42) / −0,034 (s43), client tệ nhất −0,183 / −0,145,
proxy head −0,044 / −0,037. Drift giảm mạnh (W&B r2 0,11 so 0,60) nhưng kéo proxy về model chung làm hại cả hai model.
 Rủi ro đã biết: round 1 server ngoại suy `w0 + 2·Δ̄` (đúng thuật toán; FedAvgM β0,9 từng phân kỳ).
Đại số client/server kiểm local PASS. Variant dự án (AdamW, DML không tĩnh): không thừa hưởng định lý.

## Cổng
vs `fd_dml_m4b`: private gp Δ ≥ +0,010, benign ≥ −0,005; chỉ giảm drift mà private không tăng ⇒ dừng.

## Nguồn
Acar et al., *Federated Learning Based on Dynamic Regularization*, ICLR 2021, Alg. 1 [arXiv:2111.04263](https://arxiv.org/abs/2111.04263).
[Plan §5.1](../../archive/research/algorithm-improvement-plan-2026-10-17.md).
