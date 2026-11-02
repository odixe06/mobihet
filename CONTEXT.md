# CONTEXT — MobiHet

Đọc file này → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) →
[bàn giao sống](docs/operations/continue.md). Không nạp hoặc tìm kiếm `archive/` trong phiên thông thường.

## Bài toán và quyết định đang có hiệu lực

Federated IDS V2X, VeReMi NextGen v4: 5.343 xe, 16 lớp, 77 đầu vào. Mỗi xe giữ model riêng.
Hai yêu cầu của owner: **băng thông hạn chế** và **mỗi xe có thể dùng mô hình khác nhau**; chuẩn FL: **không chia sẻ dữ liệu local**
→ chỉ trao đổi thống kê gộp và xác suất đầu ra 16 lớp, server chỉ dùng tổng toàn mạng (tương thích secure aggregation);
không dòng dữ liệu, nhãn từng dòng, trọng số, BN, gradient. Chưa có DP/secure aggregation trong thí nghiệm.

- Ưu tiên own-test tương lai từng xe, nâng xe ít mẫu/non-IID (không kéo xe mạnh xuống). Báo đủ byte hai chiều gồm bootstrap.
- Thí nghiệm giữ DAGSNet W1 cho mọi xe để cô lập cơ chế; trộn kiến trúc chưa chạy (owner 02/11: chưa cần).
- Protocol chung 50 round ([protocol](docs/research/protocol.md)); batch chẩn đoán 20 round, cosine50, seed 42/43.
  Không tự kéo dài, không chọn best round/HP bằng official test. 1 epoch/round trên dữ liệu local như repo benchmark;
  bước chỉ synthetic là cơ chế phương pháp được phép (owner 02/11).
- Kaggle: credential theo process, giữ global login; cache npy dùng chung qua kernel public `odixe0502/mobihet-v4npy-cache`.
  W&B online. Commit không có trailer `Co-Authored-By`.

## Trạng thái — 02/11/2026 22:30

M→W đã đóng audit. [W20](docs/research/results-w20.md): step floor synthetic nâng mạnh xe nhỏ/lớp vắng (giữ); bootstrap nén
3,31 → 1,02 MB giữ utility (giữ); non-target KD trên dòng thật cắt feedback, pooled tốt nhất nhưng còn đè lớp vắng;
synthetic hard+NCKD chỉ dời ngưỡng timeDelay/benign (loại).

[Đề xuất phương pháp](docs/research/proposal.md): copula bootstrap nén + class Q trên synthetic + step floor (~1,02 MB/xe/20 round),
tầng input-bank và realKD non-target chờ **X20** ([thiết kế](docs/research/screening-x20.md): core, bank, nckd_local, full).
N/O/P/Q đã DVC; R/V/W chưa DVC (OneDrive thiếu chỗ). Job, owner file và bước tiếp chỉ ở [bàn giao](docs/operations/continue.md).

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Phương pháp đề xuất, byte, thành phần nào đã chứng minh? | [proposal](docs/research/proposal.md) |
| Nguyên nhân đã biết, đang kiểm gì? | [diagnosis](docs/research/diagnosis.md), [X20](docs/research/screening-x20.md) |
| Kết quả mới và giới hạn bằng chứng? | [W20](docs/research/results-w20.md), [R20/V20](docs/research/results-r20-v20.md), [Q20](docs/research/results-q20.md), [P20](docs/research/results-p20.md), [benchmark r20](docs/research/benchmark-r20-reference.md) |
| Cơ chế code/cấu hình chung? | [method](docs/research/method.md), [protocol](docs/research/protocol.md) |
| Job nào sống, bước tiếp, việc tay? | [continue](docs/operations/continue.md) |
| Chạy công cụ, Kaggle, backup/khôi phục? | [operations](docs/operations/README.md), [scripts](scripts/README.md), [backup](docs/operations/backup.md) |
| Dataset, model, máy, image? | [knowledge](knowledge/README.md) |

Số đo ở kết quả; quyết định ở diagnosis/proposal/protocol; dữ kiện bất biến ở knowledge; trạng thái job chỉ ở continue.
