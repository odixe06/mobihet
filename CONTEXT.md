# MobiHet — điểm vào dự án

Đọc theo thứ tự: **CONTEXT → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) →
[bàn giao sống](docs/operations/continue.md)**. Sau đó mở đúng tài liệu qua bảng dưới. Không nạp `archive/` mặc định.

## Bài toán và hợp đồng

Federated IDS V2X trên VeReMi NextGen v4: 5.343 xe nhận, 77 đầu vào, 16 lớp; mỗi xe giữ model riêng.
Mục tiêu: own-test tương lai của từng xe, nhất là xe ít mẫu/non-IID, trong ngân sách giao tiếp thấp.
Chỉ trao đổi thống kê cộng được và xác suất đầu ra; server dùng tổng toàn mạng. Không chia sẻ dòng local,
model, BN hoặc gradient. Schema không phụ thuộc kiến trúc; thí nghiệm full-scale cùng DAGSNet W1 (sim b4 trộn W1/W0.25 không sụp), **chưa chạy model trộn full-scale**.

## Trạng thái hiện hành — kiểm 10/11/2026 04:30

- **Mục tiêu owner (Proposal 2, 09/11):** hiệu năng **top 3** (F1 macro r50: APFed .9119, CANPerFL .9088, FedAvg .9042…);
  băng thông **không tính bootstrap** vẫn hơn nhóm prototype (pFedES 0,14 MB, TinyProto 0,17 MB, FedProto 1,64 MB / 50 round);
  compute tại xe không tụt hạng (Core 4,00×F, giữa TinyProto 3,99 và V2X-HtFL 4,11). Bảng so sánh: trang Notion Proposal 2 của owner.
- **Phương pháp chính = Core** (F1 .8991). **KM16 w0 .5 (codec gốc)** r50 F1 .9077/.9085 (hạng 3), recall macro .910 (cao nhất),
  FAIL riêng guard benign −.0015 so Core. **Owner 09/11: chưa chốt phương pháp chính — giảm băng thông trước rồi mới chốt.**
- **Đang chạy (push 10/11 00:15): combo codec r20** CBA (lite6+QQ8+LY0), CBB (lite6+QQ8) × seed 42/43 + C0 (Core sim) + CQ8 (Core+QQ8):
  luật ở [screening-codec](docs/research/screening-codec.md), kernel/watcher ở [continue](docs/operations/continue.md).
  Codec r20 đã PASS: CH6/CH8, CM8, CT8, QQ8, LY0/LY2, LT8, LT6 (bootstrap 3,61 → 1,15 MB; Q 1.088 → 319 B/round); QQ4 FAIL.
- Đóng: generator (KM16 w0 .5 đóng băng; G1 giảm dòng synthetic FAIL, G2 bỏ), b11, b12, hướng gộp phiên + tăng compute — chi tiết
  [catalog](docs/research/improvement-proposals.md).
- Backlog owner (**chưa chạy**): (1) 3 seed 41/42/43; (2) BERT thay DAGSNet; (3) mô hình trộn 25 % mỗi loại; báo băng thông bỏ phần trước train.
- Quyền: ưu tiên nối phiên; dùng quota dư ngay, kiểm quota/slot thật trước push. **PASS → tự lên bậc scale kế** (thang dưới).
  Privacy bootstrap (ô 1 dòng) owner 09/11 **hoãn**.

## Cần biết gì → mở ở đâu

| câu hỏi | nguồn chính |
|---|---|
| Phương pháp nào được chọn, thành phần nào giữ/bỏ? | [proposal](docs/research/proposal.md), [cơ chế/code](docs/research/method.md) |
| Generator nào đã thử, vì sao đóng băng? | [catalog generator](docs/research/generators.md), [diagnosis](docs/research/diagnosis.md) |
| Cải tiến tiếp theo (ngoài generator) là gì, trạng thái? | [catalog đề xuất](docs/research/improvement-proposals.md) |
| Băng thông hiện tại từng gói, codec nào đang thử? | [screening-codec](docs/research/screening-codec.md) |
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
