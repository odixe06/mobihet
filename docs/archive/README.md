# Tài liệu lưu trữ

[Đề xuất đang làm](../proposals.md) · [Bài học dùng tiếp](../lessons/README.md) · [Index toàn archive](file-index.md).

[Handoff hiện tại](../operations/continue.md) · [Bàn giao17–20/10 đã đóng](operations/continuation-2026-10-17.md) · [Recovery r24](operations/final-recovery-2026-10-20.md).

Thiết kế/report của build đã kết thúc; không dùng handoff cũ thay CONTEXT hiện tại. V2–V10 là ID đợt kiểm nghiệm.

| hồ sơ | kết luận |
|---|---|
| [Prototype trung bình lớp](methods/class-prototypes/README.md) | Đã kiểm; đối chứng tham chiếu |
| [Truyền snapshot qua contact hữu hạn](methods/contact-fragmentation/README.md) | Đã kiểm; hạ tầng, còn giới hạn |
| [Trao đổi tri thức tại head](methods/federated-head-transfer/README.md) | Đã kiểm; không đủ lợi để ưu tiên |
| [Chưng cất logit trên điểm Gaussian](methods/gaussian-logit-distillation/README.md) | Đã kiểm; không đủ lợi để ưu tiên |
| [Fourier sketch phân phối lớp](methods/fourier-distribution-sketch/README.md) | Pilot hai seed FAIL |
| [Synthetic book khớp latent mean](methods/latent-mean-synthetic-books/README.md) | Pilot hai seed FAIL |
| [Nén update proxy bằng CountSketch](methods/proxy-countsketch/README.md) | Pilot hai seed FAIL; EF lỗi số |
| [Proxy trong cơ sở sinh từ seed](methods/proxy-seeded-subspace/README.md) | Pilot FAIL: K512 lỗi số, K1024 teacher yếu |
| [DAGSNet học tỷ số mật độ cho Gaussian](methods/gaussian-density-ratio/README.md) | Pilot hai seed FAIL |
| [Proxy: tọa độ ngẫu nhiên + bù lỗi](methods/proxy-random-coordinates/README.md) | Pilot hai seed FAIL |
| [Proxy: huấn luyện mạng con luân phiên](methods/proxy-rolling-subnetwork/README.md) | Pilot hai seed FAIL |
| [Proxy: projection trực giao đổi seed](methods/proxy-orthogonal-projection/README.md) | Pilot hai seed FAIL |
| [Proxy: PowerSGD hai lượt](methods/proxy-powersgd/README.md) | Pilot hai seed FAIL |
| [Gaussian được vận chuyển bằng MALA](methods/gaussian-mala-transport/README.md) | Pilot hai seed FAIL |
| [Gaussian giữ atom, profile và first-session](methods/gaussian-support-constraints/README.md) | Pilot hai seed FAIL |
| [Teacher từ kernel soft target trên anchor chung](methods/kernel-anchor-distillation/README.md) | Pilot hai seed FAIL |
| [Synthetic book khớp gradient DAGSNet](methods/gradient-synthetic-books/README.md) | Pilot hai seed quality FAIL; cost PASS |
| [Moment primitive + tái dựng feature hình học](methods/geometry-primitive-moments/README.md) | Pilot FAIL: G59 lỗi số, G61 quality/byte FAIL |
| [Sketch covariance Nyström](methods/covariance-nystrom/README.md) | Pilot hai seed quality FAIL; compression PASS |
| [Residual hàm từ nhãn local trên anchor](methods/anchor-function-residual/README.md) | Pilot hai seed quality FAIL; cost PASS |
| [Joint-energy bằng DAGSNet](methods/dagsnet-joint-energy/README.md) | Cost/sampler FAIL; candidate r20 chưa đo |

Pilot cũ của các hướng còn làm tiếp:
- [pilot-baselines](methods/shared-baselines/builds/pilot-baselines/README.md).
- [fragmented-snapshots](methods/contact-fragmentation/builds/fragmented-snapshots/README.md).
- [hybrid-pilot](methods/gaussian-proxy-hybrid/builds/hybrid-pilot/README.md).

Code/artifact tại [archive](../../archive/README.md). Protocol/current source lineage vẫn được giữ cho audit/resume.
