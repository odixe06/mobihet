# Mixed architectures — kiểm R2 thật

**Trạng thái (17/10):** Kế hoạch sau sàng lọc DAGSNet.

## Thiết kế
Private DAGSNet, MLP, CNN1D cùng 66 feature/16 lớp; proxy DAGSNet chung. Gán kiến trúc theo seed độc lập nhãn/kích thước
client, xoay gán. So local/no-peer, Gaussian và candidate thắng trên **cùng** gán; báo theo kiến trúc + client tệ nhất.

## PASS khi
Giao thức chạy mixed cohort không cần shape alignment của private, lợi FL giữ ở từng nhóm. Một cohort DAGSNet nhiều seed
không phải heterogeneous architecture.

## Nguồn
[Plan §7.4](../../archive/research/algorithm-improvement-plan-2026-10-17.md); HtFLlib KDD 2025 [arXiv:2506.03954](https://arxiv.org/abs/2506.03954).
