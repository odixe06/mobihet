# Nhánh B personalized — APFL/FedAPEN, FedRoD, kNN-Per, FedMRL

**Trạng thái (17/10):** Chưa chạy; cần holdout theo client.

## Cơ chế
Định nghĩa B: dự đoán trên phân bố riêng của client. APFL/FedAPEN/Fed-POE: hệ số α_k trộn private và proxy học trên split
local. FedRoD: hai head trên một thân (chung/riêng). kNN-Per: bộ nhớ kNN local trên embedding proxy. FedMRL: trộn
representation proxy + private qua projector, head Matryoshka.

## Trao đổi
0 byte thêm (local); hợp R2.

## Bằng chứng nội bộ
Theo góc local (prior-reweighted), private head ≈ proxy macro F1 (0,677 so 0,675), accuracy hơn +0,047 ⇒ hai model bổ sung.

## Việc tiếp
Fixture holdout theo client (2% cuối theo thời gian mỗi client×lớp), rồi α_k → FedRoD → kNN-Per → FedMRL theo chi phí.
Dấu hiệu sai: local F1 < max(private, proxy) + 0,010.

## Nguồn
APFL [arXiv:2003.13461](https://arxiv.org/abs/2003.13461); FedAPEN KDD 2023 [doi](https://doi.org/10.1145/3580305.3599344);
FedRoD ICLR 2022 [arXiv:2107.00778](https://arxiv.org/abs/2107.00778); kNN-Per ICML 2022 [arXiv:2111.09360](https://arxiv.org/abs/2111.09360);
FedMRL NeurIPS 2024 [arXiv:2406.00488](https://arxiv.org/abs/2406.00488). Tổng hợp: [hướng client §4.4](../../archive/research/client-model-directions-2026-10-16.md).
