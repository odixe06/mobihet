# Bài học dùng tiếp

[LESSONS kỹ thuật](../../LESSONS.md) · [bài học vận hành](../operations/lessons.md) · [Đề xuất hiện tại](../proposals.md) · [Bằng chứng lưu trữ](../archive/README.md).

Chỉ chứa bài học. Thiết kế, code, log và số đo chi tiết của build đã kết thúc nằm trong archive.

- [Prototype trung bình lớp](class-prototypes.md): Mean prototype không chở đủ ranh giới; không ưu tiên biến thể mean.
- [Truyền snapshot qua contact hữu hạn](contact-fragmentation.md): P2 audit deadline/byte/prior; chưa chứng minh đầy đủ R1.
- [Trao đổi tri thức tại head](federated-head-transfer.md): P1 head-only LG/FedGH không giải quyết backbone học thuộc.
- [Chưng cất logit trên điểm Gaussian](gaussian-logit-distillation.md): Teacher local yếu; nhãn Gaussian mạnh hơn DS-FL/FedGen adaptation.
- [Fourier sketch phân phối lớp](fourier-distribution-sketch.md): S-a KME không hơn Gaussian đủ cổng; không mở S-b.
- [Synthetic book khớp latent mean](latent-mean-synthetic-books.md): DM tĩnh và mean-latent refreshed không vượt Gaussian liên tục.
- [Nén update proxy bằng CountSketch](proxy-countsketch.md): V2-A sketch residual khuếch đại collision; không rerun y nguyên.
- [Proxy trong cơ sở sinh từ seed](proxy-seeded-subspace.md): V2-B teacher gate FAIL; không KD/ZO/contact.
- [DAGSNet học tỷ số mật độ cho Gaussian](gaussian-density-ratio.md): V2-C ESS thấp, không hơn diag/own/shuffle theo cổng.
- [Proxy: tọa độ ngẫu nhiên + bù lỗi](proxy-random-coordinates.md): V3-R small/medium không giữ đủ lợi dense32.
- [Proxy: huấn luyện mạng con luân phiên](proxy-rolling-subnetwork.md): V3-T hơn communication-only nhưng thua owncodec.
- [Proxy: projection trực giao đổi seed](proxy-orthogonal-projection.md): V3-O small/medium không giữ đủ lợi dense32.
- [Proxy: PowerSGD hai lượt](proxy-powersgd.md): V3-P adaptive basis có đóng góp nhưng compression utility FAIL.
- [Gaussian được vận chuyển bằng MALA](gaussian-mala-transport.md): V4-M không qua utility; clip floor làm score correction bằng0 ở nhiều lớp.
- [Gaussian giữ atom, profile và first-session](gaussian-support-constraints.md): V4-S fidelity PASS không dẫn tới utility hơn diag.
- [Teacher từ kernel soft target trên anchor chung](kernel-anchor-distillation.md): V5-H coverage cao đi cùng target gần uniform; teacher gate FAIL.
- [Synthetic book khớp gradient DAGSNet](gradient-synthetic-books.md): V5 IPC1/4 đạo hàm khả thi nhưng codebook không đủ utility.
- [Moment primitive + tái dựng feature hình học](geometry-primitive-moments.md): V6 không faithful ở nhánh missing; joint effect có ở control full66.
- [Sketch covariance Nyström](covariance-nystrom.md): V7 rank4/8 giảm byte nhưng không giữ đủ lợi fullcov.
- [Residual hàm từ nhãn local trên anchor](anchor-function-residual.md): V9 M64/512 không qua teacher quality; không privateKD/full y nguyên.
- [Joint-energy bằng DAGSNet](dagsnet-joint-energy.md): V10 Gaussian-base+MH hợp lệ về base nhưng không qua cổng runtime/sampler.
