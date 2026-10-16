# Phương pháp đang làm và bài học từ các hướng đã kiểm

30 nhóm FL + recipe EMA. ClassTail, CrossClientRank thêm15/10 chiều; ProxyFuse, AntiMem/QProxy, FullDrift thêm15/10 tối.
CountSketch sửa là adaptation của họ cũ, không đếm mới. 11 hồ sơ còn làm việc; adaptation lịch sử giữ archive.
Gaussian, DML và Hybrid có pilot tích cực; chưa có proof đủ r50/R1/R2. Owner15/10: proxy head là headline DML/Hybrid.
**Owner16/10: mục tiêu = dự đoán của client** (private gp primary, giữ R1/R2): [tổng hợp hướng client](research/client-model-directions-2026-10-16.md). [Chẩn đoán chững sớm](research/plateau-diagnosis-2026-10-15.md).

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
| CrossClientRank — pooled positive/negative score moments | Utility seed42 audited r20 **FAIL** (pooled−CE +0,0017, −own −0,0003); seed43 recovery v2 đang chạy, chỉ mô tả | [Nguồn](research/new-directions-2026-10-15-afternoon.md), [utility](methods/cross-client-ranking/builds/rank-utility/README.md) |
| ProxyFuse — trộn private-gp với proxy khi triển khai | Probe hậu kiểm hai seed **FAIL** (−0,032/−0,034 so proxy); dẫn tới proxy headline | [Thiết kế](methods/proxy-fusion/design.md), [kết quả](methods/proxy-fusion/results.md) |
| AntiMem — receiver-swap, M4b, swap+vị trí, EMA+BN local; QProxy q8/q4 | Audited r20 hai seed: Gaussian private PASS mọi kỹ thuật (m4b +0,025, EMA +0,035–0,039); DML/Hybrid proxy FAIL, proxy EMA hại; **QProxy q4 PASS** (±0,001, byte 7,65× nhỏ) | [Thiết kế/cổng](methods/anti-memorization/design.md), [kết quả](methods/anti-memorization/results.md) |
| FullDrift — full holdout nội bộ: sub4, FedProx, FedAvgM, FedAvg thuần, Gaussian swappos, private_ema | Audited seed42 ep6: **sub4 có triển vọng** (DML proxy +0,026, FedAvg +0,034, byte 4×); FedProx 0 (benign +0,098); FedAvgM phân kỳ; FedAvg ≈ DML; **Gaussian swappos +0,052**, EMA Gaussian +0,024/+0,036 | [Thiết kế/cổng](methods/full-drift/design.md), [kết quả](methods/full-drift/results.md) |
| FullDrift vòng 3 (client) — DML private: m4b, swappos, FedNTD; q4 ở 1 tổng hợp/epoch | Cổng private gp khóa 16/10 ~04:05Z; Audited seed42 ep6: **m4b private +0,036/proxy +0,052, swappos +0,033/+0,049 PASS**; FedNTD −0,042 FAIL; q4 noninferior. Vòng 4 audited: **hidden-state KD (FedKD) +0,006 FAIL**, **Gaussian m4b +0,067 PASS** | [Thiết kế §Vòng 3–4](methods/full-drift/design.md), [nguồn](research/client-model-directions-2026-10-16.md) |
| FullDrift vòng 2 — sub4/sub16 + q4, sub4+FedProx, horizon ep15 | Audited 16/10: **sub16_q4 vs sub4_q4 PASS** (proxy +0,041, private +0,017; 16 contact/epoch — R1 chờ mô phỏng contact); sub4_q4 proxy FAIL benign (−0,016), private noninferior; sub4_prox FAIL; **sub4 horizon ep15 PASS** (proxy +0,027, private +0,025), `fd_dml` proxy phẳng ep4–15 | [Thiết kế §Vòng 2](methods/full-drift/design.md), [kết quả](methods/full-drift/results.md) |
| FullDrift vòng 5–6 — m4b horizon ep15, m4b+q4, m4b+sub4+q4; seed 43 | Vòng 5 audited: **m4b@e15 PASS** (private +0,036, proxy +0,086, benign không sụp), m4b_q4 noninferior, **m4b_sub4_q4 +0,031 private PASS**. Vòng 6 seed 43 (dml, m4b, sub4_q4, m4b_sub4_q4) PUSHED 16/10 16:52Z; vòng 7 seed 43 (swappos, Gaussian m4b) PUSHED 17:32Z | [Thiết kế §Vòng 5–6](methods/full-drift/design.md), [kết quả](methods/full-drift/results.md) |

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
