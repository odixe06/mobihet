# V3 — carrier proxy DAGSNet, adaptation từ nguồn primary

Đợt mới sau V2-A. [Nguồn/cơ chế](../../../../../research/v3-carriers-sources.md),
[manifest khóa](../../../../../research/pilot-v3.md), [ledger](rebuild.md).

R mượn random-coordinate projection/parameter-space EF từ [Stich2018](https://arxiv.org/abs/1809.07599)
và [Karimireddy2019](https://proceedings.mlr.press/v97/karimireddy19a.html).
T mượn lịch phủ đều từ [FedRolex](https://research.google/pubs/fedrolex-model-heterogeneous-federated-learning-with-rolling-sub-model-extraction/),
không triển khai submodel extraction và không nhận kết quả R2 của nguồn.
O là projection ngẫu nhiên trực giao fresh từnground của dự án; không gọi PowerSGD nguyên bản.
P triển khai hai cohort phases theo [PowerSGD](https://arxiv.org/abs/1905.13727), thêm local epoch,private KD,BNlocal.

Mọi learned model DAGSNet; projection/mask/QR là codec trên parameter delta.
Không chuyển theorem SGD/compressor lý tưởng thành bảo đảm AdamW/non-IID/float32/IDS.
Không raw/per-record upload. Trusted aggregation chưa crypto/DP. R2 chỉ thiết kế.
U đo chất lượng trước contact; chỉ U PASS mới được thử R1 hữu hạn.
