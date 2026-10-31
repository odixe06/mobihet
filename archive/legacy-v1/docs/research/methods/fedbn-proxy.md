# FedBN trên proxy — probe chẩn đoán

**Trạng thái (17/10):** Chưa chạy; điều kiện.

## Cơ chế
Giữ BN affine + running state của proxy local, không gộp. Private đã có BN riêng (áp lại là no-op).

## Rủi ro
BN local có thể giữ shortcut receiver; không còn một proxy global để chấm nếu chưa định nghĩa calibration.

## Điều kiện mở
Chỉ khi log BN/feature shift cho thấy vấn đề còn sau m4b. Không đổi mục tiêu A sang local để gọi PASS.

## Nguồn
Li et al., *FedBN*, ICLR 2021 [arXiv:2102.07623](https://arxiv.org/abs/2102.07623). [Plan §5.4](../../archive/research/algorithm-improvement-plan-2026-10-17.md).
