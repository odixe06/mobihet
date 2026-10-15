# Phương pháp đang làm và bài học từ các hướng đã kiểm

28 nhóm FL + recipe EMA. ClassTail và CrossClientRank thêm15/10; CountSketch sửa là adaptation của họ cũ, không đếm mới.
9 hồ sơ còn làm việc; adaptation lịch sử giữ archive. Gaussian, DML và Hybrid có pilot tích cực;
chưa có proof đủ r50/R1/R2.

V2–V10 là nhãn đợt thí nghiệm, [không phải các version thay thế nhau](operations/experiment-names.md).

## Trạng thái hồ sơ trong methods/

| phương pháp | trạng thái | tài liệu |
|---|---|---|
| Proxy DAGSNet chưng cất hai chiều | Pilot có lợi; full đang nối tới r50 | [Thiết kế](methods/proxy-mutual-distillation/design.md), [build/kết quả](methods/proxy-mutual-distillation/README.md) |
| Gaussian từ moment lớp gộp | Pilot có lợi; full đang nối tới r50 | [Thiết kế](methods/gaussian-class-moments/design.md), [build/kết quả](methods/gaussian-class-moments/README.md) |
| Kết hợp Gaussian moment và proxy DAGSNet | Pilot hai seed PASS; full đang nối tới r50 | [Thiết kế](methods/gaussian-proxy-hybrid/design.md), [build/kết quả](methods/gaussian-proxy-hybrid/README.md) |
| BoundaryCells — đếm nhãn theo ô input | P2/P4 direct/teacher r20 hai seed FAIL; dừng adaptation | [Thiết kế](methods/boundary-cell-counts/design.md), [build/kết quả](methods/boundary-cell-counts/README.md) |
| ResponseRisk — consistency chống receiver shortcut | Fidelity probe PASS; chưa train augmentation | [Thiết kế](methods/receiver-response-consistency/design.md), [build/kết quả](methods/receiver-response-consistency/README.md) |
| EMA backbone + local BN + head triển khai | Thiết kế đã chốt; chưa triển khai | [Thiết kế](methods/ema-deployment-recipe/design.md), [build/kết quả](methods/ema-deployment-recipe/README.md) |
| ClassTail — histogram CE classwise, tail-risk hardlabel local | Calibrationr2 PASS; paired utilityr20 FAIL; không nối thêm | [Nguồn/cơ chế](research/new-directions-2026-10-15.md), [contract](methods/class-tail-risk/builds/class-tail-pilot/README.md) |
| CountSketch EF sửa overshoot | Co norm PASS; smallr20 hai seed qualityFAIL, mediumr17/r14 kém own; không nối | [Sửa và cổng cũ](methods/proxy-countsketch-repair/builds/contractive-ef-pilot/README.md) |
| CrossClientRank — pooled positive/negative score moments | Probe/diagnostic đã audit; utility42 recoveryr4,43 chờ startup; chưa utilityPASS | [Nguồn](research/new-directions-2026-10-15-afternoon.md), [contract](methods/cross-client-ranking/builds/rank-probe/README.md) |

## Đã kết thúc — giữ bài học, source và báo cáo trong archive

| phương pháp | kết luận có phạm vi | bài học / bằng chứng |
|---|---|---|
| Prototype trung bình lớp | Đã kiểm; đối chứng tham chiếu | [Bài học](lessons/class-prototypes.md), [hồ sơ lưu trữ](archive/methods/class-prototypes/README.md) |
| Truyền snapshot qua contact hữu hạn | Đã kiểm; hạ tầng, còn giới hạn | [Bài học](lessons/contact-fragmentation.md), [hồ sơ lưu trữ](archive/methods/contact-fragmentation/README.md) |
| Trao đổi tri thức tại head | Đã kiểm; không đủ lợi để ưu tiên | [Bài học](lessons/federated-head-transfer.md), [hồ sơ lưu trữ](archive/methods/federated-head-transfer/README.md) |
| Chưng cất logit trên điểm Gaussian | Đã kiểm; không đủ lợi để ưu tiên | [Bài học](lessons/gaussian-logit-distillation.md), [hồ sơ lưu trữ](archive/methods/gaussian-logit-distillation/README.md) |
| Fourier sketch phân phối lớp | Pilot hai seed FAIL | [Bài học](lessons/fourier-distribution-sketch.md), [hồ sơ lưu trữ](archive/methods/fourier-distribution-sketch/README.md) |
| Synthetic book khớp latent mean | Pilot hai seed FAIL | [Bài học](lessons/latent-mean-synthetic-books.md), [hồ sơ lưu trữ](archive/methods/latent-mean-synthetic-books/README.md) |
| Nén update proxy bằng CountSketch | Pilot hai seed FAIL; EF lỗi số | [Bài học](lessons/proxy-countsketch.md), [hồ sơ lưu trữ](archive/methods/proxy-countsketch/README.md) |
| Proxy trong cơ sở sinh từ seed | Pilot FAIL: K512 lỗi số, K1024 teacher yếu | [Bài học](lessons/proxy-seeded-subspace.md), [hồ sơ lưu trữ](archive/methods/proxy-seeded-subspace/README.md) |
| DAGSNet học tỷ số mật độ cho Gaussian | Pilot hai seed FAIL | [Bài học](lessons/gaussian-density-ratio.md), [hồ sơ lưu trữ](archive/methods/gaussian-density-ratio/README.md) |
| Proxy: tọa độ ngẫu nhiên + bù lỗi | Pilot hai seed FAIL | [Bài học](lessons/proxy-random-coordinates.md), [hồ sơ lưu trữ](archive/methods/proxy-random-coordinates/README.md) |
| Proxy: huấn luyện mạng con luân phiên | Pilot hai seed FAIL | [Bài học](lessons/proxy-rolling-subnetwork.md), [hồ sơ lưu trữ](archive/methods/proxy-rolling-subnetwork/README.md) |
| Proxy: projection trực giao đổi seed | Pilot hai seed FAIL | [Bài học](lessons/proxy-orthogonal-projection.md), [hồ sơ lưu trữ](archive/methods/proxy-orthogonal-projection/README.md) |
| Proxy: PowerSGD hai lượt | Pilot hai seed FAIL | [Bài học](lessons/proxy-powersgd.md), [hồ sơ lưu trữ](archive/methods/proxy-powersgd/README.md) |
| Gaussian được vận chuyển bằng MALA | Pilot hai seed FAIL | [Bài học](lessons/gaussian-mala-transport.md), [hồ sơ lưu trữ](archive/methods/gaussian-mala-transport/README.md) |
| Gaussian giữ atom, profile và first-session | Pilot hai seed FAIL | [Bài học](lessons/gaussian-support-constraints.md), [hồ sơ lưu trữ](archive/methods/gaussian-support-constraints/README.md) |
| Teacher từ kernel soft target trên anchor chung | Pilot hai seed FAIL | [Bài học](lessons/kernel-anchor-distillation.md), [hồ sơ lưu trữ](archive/methods/kernel-anchor-distillation/README.md) |
| Synthetic book khớp gradient DAGSNet | Pilot hai seed quality FAIL; cost PASS | [Bài học](lessons/gradient-synthetic-books.md), [hồ sơ lưu trữ](archive/methods/gradient-synthetic-books/README.md) |
| Moment primitive + tái dựng feature hình học | Pilot FAIL: G59 lỗi số, G61 quality/byte FAIL | [Bài học](lessons/geometry-primitive-moments.md), [hồ sơ lưu trữ](archive/methods/geometry-primitive-moments/README.md) |
| Sketch covariance Nyström | Pilot hai seed quality FAIL; compression PASS | [Bài học](lessons/covariance-nystrom.md), [hồ sơ lưu trữ](archive/methods/covariance-nystrom/README.md) |
| Residual hàm từ nhãn local trên anchor | Pilot hai seed quality FAIL; cost PASS | [Bài học](lessons/anchor-function-residual.md), [hồ sơ lưu trữ](archive/methods/anchor-function-residual/README.md) |
| Joint-energy bằng DAGSNet | Cost/sampler FAIL; candidate r20 chưa đo | [Bài học](lessons/dagsnet-joint-energy.md), [hồ sơ lưu trữ](archive/methods/dagsnet-joint-energy/README.md) |

Không rerun adaptation FAIL y nguyên. P2 contact là bằng chứng lịch sử còn giới hạn, yêu cầu R1 vẫn mở. Không chốt utility từ RUNNING/COMPLETE; pull và audit đúng source/controls/cổng trước kết luận.

[Mục lục](README.md) · [Bài học chung](../LESSONS.md) · [Index hiện tại](file-index.md) · [Index lịch sử](archive/file-index.md).
