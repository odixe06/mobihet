# Đề xuất theo tên phương pháp

Mở thư mục theo cơ chế; mã H/V/S là nhãn lịch sử để truy vết. Mỗi thư mục có giải thích, bằng chứng, bài học và liên kết source/artifact.

Source, notebook, training manifest và DVC giữ đường dẫn gốc. Một report có thể chứa nhiều phương pháp; mở đúng arm được ghi tại từng thư mục. Không sao chép số đo làm nguồn số thứ hai.

[Tổng hợp bài học](evidence-summary.md) · [Quyết định mới](../docs/operations/decisions-2026-10-12.md) · [Index toàn dự án](../docs/file-index.md)

| thư mục / phương pháp | mã cũ | trạng thái |
|---|---|---|
| [proxy-mutual-distillation](proxy-mutual-distillation/README.md) | H1/P1/P2 | P1 có lợi; P2 byte/prior còn giới hạn. |
| [gaussian-class-moments](gaussian-class-moments/README.md) | H2/P1/P2 | P1 có lợi; deadline/bootstrap P2 chưa chứng minh đủ. |
| [federated-head-transfer](federated-head-transfer/README.md) | H3/P1 | LG/FedGH không được ưu tiên tiếp. |
| [class-prototypes](class-prototypes/README.md) | H4/P1 | Đối chứng tham chiếu; không ưu tiên thêm biến thể mean. |
| [gaussian-logit-distillation](gaussian-logit-distillation/README.md) | H5/P1 | DS-FL/FedGen adaptation trượt ưu tiên. |
| [contact-fragmentation](contact-fragmentation/README.md) | H6/P2 | Hạ tầng R1; P2 có giới hạn đã audit. |
| [fourier-distribution-sketch](fourier-distribution-sketch/README.md) | S-a/KME | S-a paired FAIL; không mở S-b. |
| [latent-mean-synthetic-books](latent-mean-synthetic-books/README.md) | S-a/DM; V5 mean controls | Tĩnh FAIL; V5 refreshed/trained vẫn thua Gaussiancontinuous. |
| [proxy-countsketch](proxy-countsketch/README.md) | V2-A | Hai seed FAIL; EF partial numerical. |
| [proxy-seeded-subspace](proxy-seeded-subspace/README.md) | V2-B | K512 numerical r1; K1024 teacher FAIL; không KD/ZO/contact. |
| [gaussian-density-ratio](gaussian-density-ratio/README.md) | V2-C | Hai seed FAIL; không contact. |
| [proxy-random-coordinates](proxy-random-coordinates/README.md) | V3-R | Hai seed đãaudit; small/medium pairedFAIL. |
| [proxy-rolling-subnetwork](proxy-rolling-subnetwork/README.md) | V3-T | Hai seed đãaudit; small/medium pairedFAIL. |
| [proxy-orthogonal-projection](proxy-orthogonal-projection/README.md) | V3-O | Hai seed đãaudit; small/medium pairedFAIL. |
| [proxy-powersgd](proxy-powersgd/README.md) | V3-P | Hai seed đãaudit; medium pairedFAIL. |
| [gaussian-mala-transport](gaussian-mala-transport/README.md) | V4-M | Hai seed đãaudit; pairedFAIL. |
| [gaussian-support-constraints](gaussian-support-constraints/README.md) | V4-S | 10 arm đủ20round đã audit; paired FAIL. |
| [kernel-anchor-distillation](kernel-anchor-distillation/README.md) | V5-H/K2b | Hai seed đã audit trước cleanup; small/medium teacher FAIL. |
| [gradient-synthetic-books](gradient-synthetic-books/README.md) | V5-G/J/K3 | IPC1/4 đủhai seed; qualityFAIL, costPASS. |
| [geometry-primitive-moments](geometry-primitive-moments/README.md) | K5/V6 G59/G61 | V6 đãaudit: G59numericalFAIL; G61quality/compressionpairedFAIL. |

## Backlog đã đề xuất, chưa nằm trong 20 thư mục đầu

| phương pháp | mã cũ | trạng thái |
|---|---|---|
| [covariance-nystrom](covariance-nystrom/README.md) | S1 | Thiết kế chưa pilot; đáng kiểm dựa trên lợi full covariance H2 và nút thắt upload P2. |
| [gaussian-proxy-hybrid](gaussian-proxy-hybrid/README.md) | H1+H2 | Đã được đồng ý hướng kết hợp; chưa pilot kết hợp có đủ đối chứng riêng. |
| [anchor-function-residual](anchor-function-residual/README.md) | K2 | Thiết kế chưa sẵn sàng; đồng bộ probe/teacher vẫn có chi phí lớn. K2b kernel-anchor là hướng khác đã pilot. |
| [dagsnet-joint-energy](dagsnet-joint-energy/README.md) | K4 | Thiết kế chưa ready: bản không base measure bị bác bỏ về normalizability trước train. |

[Tổng số và phân loại triển vọng](status-summary.md). Mỗi cơ chế đếm một lần; không đếm seed/size/arm thành hướng mới.
