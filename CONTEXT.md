# MobiHet — điểm vào dự án

Đọc theo thứ tự: **CONTEXT → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) →
[bàn giao sống](docs/operations/continue.md)**. Sau đó mở đúng tài liệu qua bảng dưới. Không nạp `archive/` mặc định.

## Bài toán và hợp đồng

Federated IDS V2X trên VeReMi NextGen v4: 5.343 xe nhận, 77 đầu vào, 16 lớp; mỗi xe giữ model riêng.
Mục tiêu: own-test tương lai của từng xe, nhất là xe ít mẫu/non-IID, trong ngân sách giao tiếp thấp.
Chỉ trao đổi thống kê cộng được và xác suất đầu ra; server dùng tổng toàn mạng. Không chia sẻ dòng local,
model, BN hoặc gradient. Schema không phụ thuộc kiến trúc; thí nghiệm full-scale cùng DAGSNet W1.

## Trạng thái hiện hành — kiểm 10/11/2026 21:30

- **Phương pháp chính = CBB** (owner chốt 10/11) = Core + generator KM16 w0 .5 + codec lite6 + Q 8-bit (QQ8) —
  [proposal](docs/research/proposal.md), cơ chế ở [method](docs/research/method.md). r50 TB 2 seed: F1 macro **.9078** (hạng 3 sau
  APFed .9119, CANPerFL .9088), 265 B/round (13,3 KB/50 round không tính bootstrap), bootstrap 1,15 MB/xe. Core (.8991) là đối chứng.
  Đánh đổi đã chấp nhận: guard benign thấp hơn Core (~−.0015).
- **Mục tiêu owner (Proposal 2):** top 3 F1 ✓; byte không tính bootstrap < prototype (pFedES 0,14 MB) ✓; compute tại xe = Core 4,00×F
  (không tụt hạng). Bảng so sánh: trang Notion Proposal 2 của owner.
- Không job sống. Đóng: generator khác, b11, b12, G1/G2, gộp phiên, tăng compute, CBA/QQ4 — [catalog](docs/research/improvement-proposals.md).
- Owner 10/11: **tạm gác** chạy seed mới và model mới (BERT, model trộn); chưa có việc train kế tiếp được duyệt.
  Privacy bootstrap (ô 1 dòng) **hoãn** (09/11).

## Cần biết gì → mở ở đâu

| câu hỏi | nguồn chính |
|---|---|
| Phương pháp chính (CBB) là gì, thành phần nào giữ/bỏ? | [proposal](docs/research/proposal.md), [cơ chế/code](docs/research/method.md) |
| Generator nào đã thử, vì sao đóng băng? | [catalog generator](docs/research/generators.md), [diagnosis](docs/research/diagnosis.md) |
| Cải tiến tiếp theo (ngoài generator) là gì, trạng thái? | [catalog đề xuất](docs/research/improvement-proposals.md) |
| Băng thông từng gói, codec nào PASS/FAIL? | [screening-codec](docs/research/screening-codec.md) |
| Số đo, luật đã khóa và giới hạn bằng chứng? | [chỉ mục nghiên cứu](docs/research/README.md) |
| Job nào sống, ai sở hữu file, phiên tới làm gì? | [bàn giao duy nhất](docs/operations/continue.md) |
| Chạy Kaggle, phục hồi, lấy artifact hoặc backup? | [vận hành](docs/operations/README.md), [công cụ](scripts/README.md) |
| Dataset/model/máy/runtime và cấu hình chung? | [knowledge](knowledge/README.md), [protocol bất biến](docs/research/protocol.md) |

## Quy tắc tránh làm sai khi tiếp phiên

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
