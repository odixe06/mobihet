# Đề xuất theo tên phương pháp

Mở thư mục theo cơ chế; mã H/V/S là nhãn lịch sử để truy vết. Mỗi thư mục có giải thích, bằng chứng, bài học và liên kết source/artifact.

Source, notebook, training manifest và DVC giữ đường dẫn gốc. Một report có thể chứa nhiều phương pháp; mở đúng arm được ghi tại từng thư mục. Không sao chép số đo làm nguồn số thứ hai.

[Tổng hợp bài học](evidence-summary.md) · [Quyết định mới](../docs/operations/decisions-2026-10-12.md) · [Index toàn dự án](../docs/file-index.md)

| thư mục / phương pháp | mã cũ | trạng thái |
|---|---|---|
| [proxy-mutual-distillation](proxy-mutual-distillation/README.md) | H1/P1/P2 | P1 có lợi; full100%train/lịch50round đãlaunchcalibration. R1/R2 cònthiếu. |
| [gaussian-class-moments](gaussian-class-moments/README.md) | H2/P1/P2 | P1 có lợi; full100%train/lịch50round đãlaunchcalibration. R1/R2 cònthiếu. |
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

## Bốn cơ chế bổ sung, triển khai V7–V10

| phương pháp | mã cũ | trạng thái |
|---|---|---|
| [covariance-nystrom](covariance-nystrom/README.md) | S1/V7 | Hai seed đãaudit: rank4/8 compressionPASS, qualitypairedFAIL; khôngcontact/full. |
| [gaussian-proxy-hybrid](gaussian-proxy-hybrid/README.md) | H1+H2/V8 | Seed43 đãaudit/utilityPASS; seed42 cònpull/audit, chưapairedconfirmation. |
| [anchor-function-residual](anchor-function-residual/README.md) | K2/V9 | Seed42 audit13arm×20round:M64/M512costPASS/qualityFAIL; seed43chờaudit. K2b V5 làcơ chếkhác. |
| [dagsnet-joint-energy](dagsnet-joint-energy/README.md) | K4/V10 | Gaussian-base+MH hai seed đãaudit: costFAIL/r1samplerFLFAIL, F1r20chưađo. UnbasedR66normalizabilityFAIL vẫnđúng. |

[Tổng số và phân loại triển vọng](status-summary.md). Mỗi cơ chế đếm một lần; không đếm seed/size/arm thành hướng mới.

## Hướng mới 2026-10-13

| phương pháp | trạng thái / dự báo |
|---|---|
| [BoundaryCells — đếm nhãn theo vùng input](boundary-cell-counts/README.md) | P2/P4 bounded probe xong; direct/teacher và gp hiệu dụng đã chốt; chưa DAGSNet utility |
| [ResponseRisk — consistency theo phản ứng receiver](receiver-response-consistency/README.md) | Swap local được phép với fidelity gate; dự báo thấp/vừa, chưa pilot |
| [EMA + BN + head riêng](ema-deployment-recipe/README.md) | Recipe H1/H2, không họ FL mới; fresh2×2+own đã chốt |

24nhóm lịch sử +2hướng mới =26nhóm đề xuất; InputSensitivity/ErrorCosts trong
[ghi chú](../docs/research/new-fl-primary-sources-2026-10-13.md) còn backlog, không cộng vào26.
Phân loại24nhóm đãaudit giữ riêng; probe mới chưa tăng số pilot tích cực.
