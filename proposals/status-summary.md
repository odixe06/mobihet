# Tổng số hướng và triển vọng — 2026-10-12

**24 nhóm phương pháp** được đề xuất cho repo hiện hành; không đếm toàn bộ archive hay tênpaper chỉ được tham khảo.
Gộp seed42/43, small/medium, FO512/1024, IPC1/4, EF/noEF, own/shuffle và phiên bản theo cùng cơ chế.
20 thư mục đầu có triển khai/đối chứng/thiết kế; thêm4backlog đã có ở tài liệu cũ để tránh bỏ sót.

## Phân loại sau audit V3–V5

| phân loại | số | phương pháp |
|---|---:|---|
| Có bằng chứng pilot tích cực, tiếp tục ưu tiên | 2 | Gaussian class moments; proxy mutual distillation DAGSNet |
| Cơ chế mới đáng ưu tiên kiểm, chưa có utilityPASS | 2 | Gaussian+proxy hybrid; covariance Nyström |
| Chờ audit V3–V5 | 0 | Đủ18kernel; report V3/V4/V5 có48/20/46run đãaudit |
| Bản thử hiện tại không ưu tiên tiếp | 16 | headtransfer; Gaussianlogitdistillation; Fouriersketch; latentmeanbooks; CountSketch; seededsubspace; densityratio; randomcoordinates; rolling-subnetwork; orthogonalprojection; PowerSGD; MALA; supportconstraints; kernelanchors; gradientbooks; geometryprimitive K5 |
| Thiết kế chưa sẵn sàng | 2 | anchorfunctionresidual; DAGSNetjointenergy |
| Đối chứng hoặc hạ tầng | 2 | classprototypes; contactfragmentation |

**4 hướng đáng ưu tiên** là đánh giá từ bằng chứng và cơ chế, không phải4hướng đã thắng.
Chỉ2 có bằng chứng pilot tích cực; 2 chưa có kiểm nghiệm chất lượng của phương pháp mới.
Trong24nhóm có2nhóm đối chứng/hạ tầng; nếu chỉ tính các nhóm còn lại thì22.
**0 hướng đã xác minh đủ R1+R2**.

| 4 hướng tiếp tục ưu tiên | vì sao giữ | bằng chứng còn thiếu |
|---|---|---|
| Gaussian class moments | P1 có lợi từ covariance và tri thức quần thể | Byte/bootstrap/deadline thực; đa kiến trúc |
| Proxy mutual distillation DAGSNet | P1 teacher/private có lợi | Prior hiệu dụng; truyền model dưới contact hữu hạn; đa kiến trúc |
| Gaussian+proxy hybrid | Hai thành phần riêng có lợi, cơ chế có thể bổ sung nhau | Pilot kết hợp cùng đối chứng từng thành phần |
| Covariance Nyström | Có thể giảm uplink covariance mà giữ tương quan | Probe toán không thay F1; cần utility và contact pilot |

[Danh mục24hướng](README.md) · [Bài học/bằng chứng](evidence-summary.md).

## Giới hạn trạng thái

Đã đủ audit18kernel V3–V5. Bốn carrier V3 đều pairedFAIL với dense32 làm reference;
V4 MALA/support pairedFAIL; V5 anchor small/medium và gradientIPC1/4 pairedFAIL.
Latentmean tĩnhFAIL; đối chứng refreshed/trained của V5 vẫn không hơn Gaussiancontinuous.
Đây là kết luận cho adaptation/cấu hình đã thử, không bác bỏ toàn họ FL tương ứng.
Không tune/bestround/head đổi để cứu cổng. StatusCOMPLETE không thay audit.

Nguồn trạng thái: [V3 gates](../papers/v3/gates.json), [V4 gates](../papers/v4/gates.json),
[V5 gates](../papers/v5/gates.json), [P1](../docs/research/probes/pilot-p1.md).

Sau V6: K5G59/G61 quality/compressionFAIL; không còn trong nhóm ưu tiên adaptation hiện tại.
Nyström/hybrid chưa chứng minh tốt hơn Gaussian hoặc giải quyết contact.
Jointenergy bảnkhôngbase đãnormalizabilityFAIL; thiết kếbase khác cầncontract mới, không tự coi đã đủ.

Không cộng paperbaseline/cácarchitectureMLP bịloại khỏiDAGS-only hoặc archive recordsharing vi phạmR3
thành hướng khảthi mới. Phân loại sẽ cập nhật sau collect/audit, giữ số24 nếu không phát sinh cơ chế mới.
