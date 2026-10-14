# Tổng số hướng và triển vọng — 2026-10-14

**26 nhóm phương pháp** hiện được đề xuất:24nhóm lịch sử dưới đây +BoundaryCells/ResponseRisk mới.
Hai hướng mới chưa có utilityDAGSNet; bounded probe không tăng số phương pháp pilot đã xác nhận.
EMA là cải tiến recipe, InputSensitivity/ErrorCosts còn backlog; không cộng thêm vào26.
[Dự báo trước kiểm](../docs/research/performance-feasibility-2026-10-13.md),
[hai hướng mới](README.md#hướng-mới-2026-10-13).
Không đếm toàn bộ archive hay tênpaper chỉ được tham khảo.
Gộp seed42/43, small/medium, FO512/1024, IPC1/4, EF/noEF, own/shuffle và phiên bản theo cùng cơ chế.
20 thư mục đầu có triển khai/đối chứng/thiết kế; thêm4cơchế từbacklog cũ đã triển khai V7–V10.

## Phân loại 24 nhóm lịch sử sau audit V3–V7 và V10

| phân loại | số | phương pháp |
|---|---:|---|
| Có bằng chứng pilot tích cực, tiếp tục ưu tiên | 3 | Gaussian class moments; proxy mutual distillation DAGSNet; Gaussian+proxy hybrid paired PASS |
| Chờ audit V3–V5 | 0 | Đủ18kernel; report V3/V4/V5 có48/20/46run đãaudit |
| Bản thử hiện tại không ưu tiên tiếp | 19 | headtransfer; Gaussianlogitdistillation; Fouriersketch; latentmeanbooks; CountSketch; seededsubspace; densityratio; randomcoordinates; rolling-subnetwork; orthogonalprojection; PowerSGD; MALA; supportconstraints; kernelanchors; gradientbooks; geometryprimitive K5; Gaussian-base jointenergy K4; covariance Nyström; anchorfunctionresidual V9 |
| Đối chứng hoặc hạ tầng | 2 | classprototypes; contactfragmentation |

**3 hướng tiếp tục ưu tiên có bằng chứng pilot:** Hybrid đã qua cả hai seed.
Nyström/residual đóng adaptation hiện tại sau paired quality FAIL.
Trong24nhóm có2nhóm đối chứng/hạ tầng; nếu chỉ tính các nhóm còn lại thì22.
**0 hướng đã xác minh đủ R1+R2**.

| 3 hướng tiếp tục ưu tiên | vì sao giữ | bằng chứng còn thiếu |
|---|---|---|
| Gaussian class moments | P1 có lợi từ covariance và tri thức quần thể | Full100%train đã audit r9–r10/50; continuation theo quota/17Oct; R1/R2 còn thiếu |
| Proxy mutual distillation DAGSNet | P1 teacher/private có lợi | Full100%train đã audit r7/50; prior/contact/R2 còn thiếu |
| Gaussian+proxy hybrid | Pilot paired PASS; full-data đầu có lợi trong matched controls | 7arm đã audit r2–r3/50; chưa full dài/R1/R2; cost/budget đo riêng |

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
Nyström rank4/8 compressionPASS nhưngqualitypairedFAIL. Hybrid paired PASS, chưa R1 proof.
Jointenergy bảnkhôngbase đãnormalizabilityFAIL. V10Gaussian-base+MH đãaudit14arm hai seed:
costFAIL vàr1samplerFLFAIL, chưacandidateF1r20. Khôngmởprivate/contact hoặc rerun y nguyên.
K2 residual V9 đủ 26 arm-seed×20round đã audit: M64/M512 cost PASS/quality paired FAIL.
Xem [expansion](../docs/operations/expansion-2026-10-12.md), [V10gates](../papers/v10/gates.json),
[fullledger](../papers/full-validation/rebuild.md).

Không cộng paperbaseline/cácarchitectureMLP bịloại khỏiDAGS-only hoặc archive recordsharing vi phạmR3
thành hướng khảthi mới. Phân loại utility24nhóm cũ cập nhật sau collect/audit;
BoundaryCells/ResponseRisk vẫn chỉ đề xuất/probe, chưa pilotPASS hoặc R1/R2proof.

BoundaryCells calibration T4 cả hai seed đã audit/cost PASS (1,49–1,60×single-DAGS);
utility r20 chưa đo. ResponseRisk fidelity PASS; cấu hình probe/λJS/refresh đã chốt.
EMA recipe cấu hình đã chốt, chưa utility; vẫn không cộng vào 26 nhóm.

Ngày14/10: Hybrid full đủ4kernel/18round đã audit; hai ERROR do budget, không rerun từr1.
[Kết luận và 10 metrics](../papers/hybrid-full/feasibility-20261014.md).
ResponseRisk quyết định mới nhất chỉ kiểm localaugmentation trước; shared-response pilot hoãn.
26 là số nhóm đề xuất, 3 là nhóm có bằng chứng pilot tích cực; không nhóm nào đã đủ r50/R1/R2 proof.
