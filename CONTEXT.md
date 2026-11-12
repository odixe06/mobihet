# MobiHet — điểm vào dự án

Đọc theo thứ tự: **CONTEXT → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) →
[bàn giao sống](docs/operations/continue.md)**. Sau đó mở đúng tài liệu qua bảng dưới. Không nạp `archive/` mặc định.

## Bài toán và hợp đồng

Federated IDS V2X trên VeReMi NextGen v4: 5.343 xe nhận, 77 đầu vào, 16 lớp; mỗi xe giữ model riêng.
Mục tiêu: own-test tương lai của từng xe, nhất là xe ít mẫu/non-IID, trong ngân sách giao tiếp thấp.
Chỉ trao đổi thống kê cộng được và xác suất đầu ra; server dùng tổng toàn mạng. Không chia sẻ dòng local,
model, BN hoặc gradient. Schema không phụ thuộc kiến trúc; thí nghiệm full-scale cùng DAGSNet W1.

## Trạng thái hiện hành — kiểm 13/11/2026 02:45 UTC+7

- **Phương pháp chính = CBB** (owner chốt 10/11) = Core + generator KM16 w0 .5 + codec lite6 + Q 8-bit (QQ8) —
  [proposal](docs/research/proposal.md), cơ chế ở [method](docs/research/method.md). r50 **3 seed 41/42/43**: F1 macro **.9078 ± .0004**
  (hạng 3 sau APFed .9130, CANPerFL .9097; FedAvg .9048 hạng 4 — TB 3 seed), 265 B/round (13,3 KB/50 round không tính bootstrap),
  bootstrap 1,15 MB/xe, compute **6,09×F/dòng**. Core (.8991) là đối chứng. [Report 3 seed](runs/cbb-s41/report.md).
- **Mục tiêu owner (Proposal 2):** top 3 F1 ✓; byte không tính bootstrap < prototype (pFedES 0,14 MB) ✓; compute tại xe không tụt hạng.
- **Scope (owner 10–12/11):** ưu tiên **macro-F1**; guard p10/low256/TD/benign/absent report-only (vẫn audit đủ);
  chỉ trao thống kê/xác suất, không dòng thật, **không trao tham số** (mỗi xe model khác hoàn toàn được), giữ protocol 50r×1epoch,
  giữ đóng gộp phiên; **tạm không lượng tử hóa** thêm; experiment DAGSNet W1.
- **Cải thiện F1 trong hợp đồng hiện tại gần hết dư địa:** B13–B19, A1, N1, N2 đều FAIL (>40 biến thể trong ±.002) —
  bảng theo họ cơ chế ở [sổ tay xây dựng phương pháp](docs/research/method-building-brief.md) §4–5. Không mở lại họ đã đóng bằng tên mới.
- **Đang chạy — B20 giảm compute full-scale** ([design + kết quả](docs/research/screening-compute-b20.md)): SD r50 s42/s43 (5,32×F),
  SM r50 s42 (resume từ pilot r20 PASS_TRAJECTORY) + s43 (4,65×F). Luật r50: macro mỗi seed > .9048. Watcher nền tự collect/audit —
  job, log, lệnh theo dõi ở [continue](docs/operations/continue.md).
- Baseline 3 seed (`../fl_dagsnet_veremi_v2`) đã xong và audit; SCAFFOLD bị loại (phân kỳ). Privacy bootstrap **hoãn** (09/11).
  Mô hình trộn kiến trúc full-scale chưa chạy.

## Cần biết gì → mở ở đâu

| câu hỏi | nguồn chính |
|---|---|
| **Muốn cải thiện F1 / xây phương pháp mới: trần, đường học, họ nào đã đóng, hướng nào mở?** | [sổ tay xây dựng phương pháp](docs/research/method-building-brief.md) |
| Phương pháp chính (CBB) là gì, thành phần nào giữ/bỏ? | [proposal](docs/research/proposal.md), [cơ chế/code](docs/research/method.md) |
| Giảm compute tại xe: arm nào, luật, kết quả? | [B20](docs/research/screening-compute-b20.md), [phụ lục luật](docs/research/screening-compute-b20-addendum.md) |
| **Cơ chế X đã từng thử chưa, kết quả, doc nào?** (mọi study từ v1 → B20) | [index phương pháp đã thử](docs/research/tried-methods-index.md) |
| Generator / đề xuất ngoài generator đã thử gì, trạng thái? | [catalog generator](docs/research/generators.md), [catalog đề xuất](docs/research/improvement-proposals.md) |
| Băng thông từng gói, codec nào PASS/FAIL? | [screening-codec](docs/research/screening-codec.md) |
| Job nào sống, ai sở hữu file, phiên tới làm gì? | [bàn giao duy nhất](docs/operations/continue.md) |
| Chạy Kaggle, phục hồi, lấy artifact hoặc backup? | [vận hành](docs/operations/README.md), [công cụ](scripts/README.md) |
| Dataset/model/máy/runtime và cấu hình chung? | [knowledge](knowledge/README.md), [protocol bất biến](docs/research/protocol.md) |
| Bài rà/khảo sát 08–11/11 (module, macro-first, teacher source, cross-fit, audit lịch sử)? | [archive 13/11](archive/docs-20261113/README.md) — đã tổng hợp vào sổ tay |

## Quy tắc tránh làm sai khi tiếp phiên

**Scope macro-first 11/11:** B17 và các trial mới dùng macro làm utility gate, guard phụ report-only;
ngưỡng/tầng chi tiết khóa trong design. Verdict lịch sử giữ nguyên.

**Thang mở rộng tự động (owner 09/11, áp mọi ứng viên):** audit PASS + luật đã khóa PASS → push bậc kế ngay, không chờ owner duyệt.
1. Sim fold val r20 (200 xe) PASS **cả seed42 và seed43** (một seed PASS → chạy seed còn lại trước) → full-scale pilot r20 cosine50 seed42, so control cùng round.
2. Pilot full-scale r20 PASS → nối chính run đó tới r50 từ state r20 **và** push seed43 r50 cùng lúc.
3. r50 PASS cả hai seed theo luật r50 của phương pháp → nhận vào phương pháp chính ([proposal](docs/research/proposal.md)), làm nền cho sim kế.
Mỗi bậc: khóa luật trong tài liệu trước push, kiểm quota sống, ghi job vào continue. FAIL ở bất kỳ bậc → dừng nhánh đó, ghi lý do;
không tune, không kéo dài run FAIL để tìm win. Nhiều ứng viên cùng PASS → mỗi ứng viên lên riêng (không tự ghép combo).

Giữ protocol, split, source đã khóa; headline chính r50, pilot r20 chỉ là bậc trung gian của thang. Không chọn HP/best round qua official test. Các nghiên cứu đã đọc
nhiều lần official test, hai seed chưa là significance. Compile/cache là hạ tầng; không đổi bước/loss để tăng tốc.
`COMPLETE`/exit0/W&B finished không thay audit artifact. Resume phải có state/RNG cả hai rank cùng round, kiểm trước decode.
Không sửa file worker đang sở hữu hoặc bật watcher trùng. Không Git/DVC push, xóa artifact hay đổi global credential trong quyền hiện tại.

Dữ kiện bất biến → `knowledge/`; phương pháp/gate/kết quả → `docs/research/`; job sống → `continue`;
bài học tái sử dụng → hai lessons; hồ sơ đã đóng → archive có chỉ mục. Chỉ hub này dẫn tới bàn giao hiện hành.

[Bản đồ toàn bộ tài liệu](docs/README.md) · [Tổng quan repo](README.md).
