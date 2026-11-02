# CONTEXT — MobiHet

Đọc file này → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) →
[bàn giao sống](docs/operations/continue.md). Không nạp hoặc tìm kiếm `archive/` trong phiên thông thường.

## Bài toán và quyết định đang có hiệu lực

Federated IDS V2X, VeReMi NextGen v4: 5.343 xe, 16 lớp, 77 đầu vào. Mỗi xe giữ model riêng.
Hai yêu cầu của owner: **băng thông hạn chế** và **mỗi xe có thể dùng mô hình khác nhau** → chỉ trao đổi thống kê dữ liệu gộp
và xác suất đầu ra 16 lớp; không dòng dữ liệu, nhãn từng dòng, trọng số, BN, gradient. Chưa có DP/secure aggregation.

- Ưu tiên own-test tương lai từng xe, nâng xe ít mẫu/non-IID (không kéo xe mạnh xuống). Báo đủ byte hai chiều gồm bootstrap.
- Thí nghiệm giữ DAGSNet W1 cho mọi xe để cô lập cơ chế; trộn kiến trúc chưa chạy (owner 02/11: chưa cần).
- Protocol chung 50 round ([protocol](docs/research/protocol.md)); batch chẩn đoán 20 round, cosine50, seed 42/43.
  Không tự kéo dài, không chọn best round/HP bằng official test. 1 epoch/round trên dữ liệu local như repo benchmark;
  bước chỉ synthetic là cơ chế phương pháp được phép (owner 02/11).
- Kaggle: credential theo process, giữ global login; cache npy dùng chung qua kernel public `odixe0502/mobihet-v4npy-cache`.
  W&B online. Commit không có trailer `Co-Authored-By`.

## Trạng thái — 02/11/2026 17:00

M→V đã đóng audit. [R20/V20](docs/research/results-r20-v20.md): độ ồn phiên đã đo; realKD hại timeDelay/lớp vắng cả khi
teacher cố định (margin chỉ thắng nhờ bảo vệ teacher); val sạch xác nhận augmentation copula là nguồn lợi chính,
input-target nâng xe yếu, realKD không còn lợi pooled. Bootstrap chiếm 97% byte và nén được ~70%.

[Đề xuất phương pháp](docs/research/proposal.md): copula bootstrap nén + class Q trên synthetic (~1 MB/xe/20 round),
tầng input-bank tùy chọn, không realKD. **W20 đang chạy** (10 run: non-target KD, step floor, bootstrap nén) —
[thiết kế](docs/research/screening-w20.md). Backup DVC N/O/P/Q đang đẩy; xóa local N/O/P/Q sau verify (owner đã cho phép).
Job, owner file và bước tiếp chỉ ở [bàn giao](docs/operations/continue.md).

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Phương pháp đề xuất, byte, thành phần nào đã chứng minh? | [proposal](docs/research/proposal.md) |
| Nguyên nhân đã biết, đang kiểm gì? | [diagnosis](docs/research/diagnosis.md), [W20](docs/research/screening-w20.md) |
| Kết quả mới và giới hạn bằng chứng? | [R20/V20](docs/research/results-r20-v20.md), [Q20](docs/research/results-q20.md), [P20](docs/research/results-p20.md), [benchmark r20](docs/research/benchmark-r20-reference.md) |
| Cơ chế code/cấu hình chung? | [method](docs/research/method.md), [protocol](docs/research/protocol.md) |
| Job nào sống, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Chạy công cụ, Kaggle, backup/khôi phục? | [operations](docs/operations/README.md), [scripts](scripts/README.md), [backup](docs/operations/backup.md) |
| Dataset, model, máy, image? | [knowledge](knowledge/README.md) |

Số đo ở kết quả; quyết định ở diagnosis/proposal/protocol; dữ kiện bất biến ở knowledge; trạng thái job chỉ ở continue.
