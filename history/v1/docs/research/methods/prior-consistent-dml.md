# Prior-consistent DML — logit adjustment trong loss (FedLC)

**Trạng thái (17/10):** Tạm hoãn.

## Cơ chế
CE trên `z + log π_k`, KL với proxy trên `z + log π_g`; triển khai `z + log π_g` (A) hoặc `z + log π_k` (B).
Rule `gp` hiện là bản hậu nghiệm (head→gp +0,12 global).

## Lý do hoãn
Khi proxy cũng train CE trên dữ liệu local, đổi tham số `z = u + log π_k` cho cùng hàm loss; chưa tách được cơ chế mới nếu
không đo prior thực của proxy. gp có thể sửa quá tay khi KD kéo private về global.

## Nguồn
Menon et al., ICLR 2021 [arXiv:2007.07314](https://arxiv.org/abs/2007.07314); FedLC ICML 2022 [arXiv:2209.00189](https://arxiv.org/abs/2209.00189).
[Hướng client §4.2, §5](../../archive/research/client-model-directions-2026-10-16.md).
