# CONTEXT — MobiHet

Đọc file này → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) →
[bàn giao sống](docs/operations/continue.md). Không nạp hoặc tìm kiếm `archive/` trong phiên thông thường.

## Bài toán và quyết định đang có hiệu lực

Federated IDS V2X, VeReMi NextGen v4: 5.343 xe, 16 lớp, 77 đầu vào. Mỗi xe giữ model riêng;
trao đổi thống kê gộp/output schema chung, không mẫu thật, feature/nhãn từng dòng hoặc private weights.
Nhánh hiện tại không proxy neural; Q là bảng xác suất theo lớp. Chưa có bảo đảm DP/secure aggregation.

- Ưu tiên own-test tương lai của từng xe, cải thiện xe ít mẫu/non-IID và giảm chênh chất lượng bằng cách nâng xe yếu.
- Tối ưu ba khâu: thông tin upload → server xử lý/downlink → client kết hợp local và teacher.
- Chưa ngưỡng byte cứng; báo bootstrap + upload/download và thời gian. Schema hỗ trợ khác kiến trúc;
  thí nghiệm hiện giữ DAGSNet W1 để cô lập cơ chế, chưa chứng minh kiến trúc trộn.
- Protocol chung 50 round; batch chẩn đoán M chỉ r1–20, vẫn cosine50, seed42/43. Không tự kéo dài hoặc chọn best test round.
- CLI global `dinhan25`, dùng credential theo process. W&B online. Không cần MCP JSON.

## Trạng thái — 01/11/2026

M0–M15 version1 đã collect và audit đủ20round; source/config/CM/Qlag/codec/snapshot PASS.
Kết quả và giới hạn tại [M20](docs/research/results-m20.md). Không kernel hoặc collector đang chạy.

Owner yêu cầu push Git/DVC, gọn context/archive và xóa local sau backup đã kiểm.
Trạng thái backup/cleanup duy nhất ở [bàn giao](docs/operations/continue.md).
Sau bảo trì: khóa đối chứng tách realKD/synthetic gate rồi mới cải tiến cơ chế tổng; chưa launch batch mới.

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Cơ chế hiện tại và cấu hình chung? | [method](docs/research/method.md), [protocol](docs/research/protocol.md) |
| Kết quả M và giới hạn bằng chứng? | [M20](docs/research/results-m20.md) |
| Nguyên nhân nào đã biết, cần kiểm gì tiếp? | [diagnosis](docs/research/diagnosis.md) |
| Job nào sống, ai sở hữu file, bước tiếp? | [continue](docs/operations/continue.md) |
| Cách chạy công cụ, backup và khôi phục? | [operations](docs/operations/README.md), [scripts](scripts/README.md) |
| Dataset, model, máy, image? | [knowledge](knowledge/README.md) |

Số đo ở kết quả; quyết định ở diagnosis/protocol; dữ kiện bất biến ở knowledge; trạng thái job chỉ ở continue.
