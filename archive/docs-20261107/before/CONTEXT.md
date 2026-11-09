> Snapshot 07/11/2026; tài liệu hiện hành: [CONTEXT.md](../../../CONTEXT.md). Không đọc mặc định.

# CONTEXT — MobiHet

Đọc file này → [LESSONS](../../../LESSONS.md) → [bài học vận hành](../../../docs/operations/lessons.md) →
[bàn giao sống](../../../docs/operations/continue.md). Không nạp hoặc tìm kiếm `archive/` trong phiên thông thường.

## Bài toán và quyết định đang có hiệu lực

Federated IDS V2X, VeReMi NextGen v4: 5.343 xe, 16 lớp, 77 đầu vào. Mỗi xe giữ model riêng.
Hai yêu cầu của owner: **băng thông hạn chế** và **mỗi xe có thể dùng mô hình khác nhau**; chuẩn FL: **không chia sẻ dữ liệu local**
→ chỉ trao đổi thống kê gộp và xác suất đầu ra 16 lớp, server chỉ dùng tổng toàn mạng (tương thích secure aggregation);
không dòng dữ liệu, nhãn từng dòng, trọng số, BN, gradient. Chưa có DP/secure aggregation trong thí nghiệm.

- Ưu tiên own-test tương lai từng xe, nâng xe ít mẫu/non-IID (không kéo xe mạnh xuống). Báo đủ byte hai chiều gồm bootstrap.
- Thí nghiệm giữ DAGSNet W1 cho mọi xe để cô lập cơ chế; trộn kiến trúc chưa chạy (owner 02/11: chưa cần).
- Protocol chung 50 round ([protocol](../../../docs/research/protocol.md)); batch chẩn đoán 20 round, cosine50, seed 42/43.
  Không tự kéo dài, không chọn best round/HP bằng official test. 1 epoch/round trên dữ liệu local như repo benchmark;
  bước chỉ synthetic là cơ chế phương pháp được phép (owner 02/11).
- Kaggle: credential theo process, giữ global login; cache npy dùng chung qua kernel public `odixe0502/mobihet-v4npy-cache`.
  W&B online. Commit không có trailer `Co-Authored-By`.

## Trạng thái — kiểm lại 07/11/2026 02:41, F50 đóng 4/4, phương pháp = Core, Y50 đang chuyển phiên

M→X đã đóng audit. [X20](../research/results-x20.md): floor + bootstrap nén cộng thẳng (core); input-bank qua gate
(p10 .717, low256 .801) nhưng lợi thu hẹp theo round; NCKD giới hạn lớp xe có xóa hại lớp vắng nhưng vẫn tốn timeDelay,
combo full trượt gate → không realKD.

[Đề xuất phương pháp](../../../docs/research/proposal.md): copula bootstrap nén + class Q trên synthetic + step floor + input-bank
(~2,33 MB/xe/20 round; lõi không bank 1,02 MB). **[F50](../../../docs/research/results-f50.md) đã đóng 4 run r50, bank trượt luật cả hai seed → Core**: 50 round bank s42/s43 + core s42/s43, checkpoint mỗi round
([thiết kế](../research/screening-f50.md)). [Rà pipeline](../research/pipeline-review.md): ứng viên cải tiến, hướng đã loại. **[Y50](../../docs-20261109/research/screening-y50.md): oracle đã audit r50 (macro .917581, generator_worth_it); prior FAIL r20; Q/floor đang chuyển checkpoint sang tài khoản còn >3h để nối r50** 3 ứng viên Core (oracle, prior, bỏ Q/floor); job sống ở [continue](../../../docs/operations/continue.md).
Owner03/11 chọn **Core** làm mục tiêu đề xuất; [C10](../research/results-c10.md) đủ 4/4 audit 10 round/seed42. **[L50](../../../docs/research/results-l50.md) đã hoàn tất 50 round, audit PASS**: Core lossless CDF (zlib-6 uint16) đạt macro F1 0.899129 (F2 Core 0.899160), P10 0.752754; tổng communication budget 50 round giảm còn 691.466 B/xe (−34,45%). Toàn bộ kernel L50 đã đóng.
Generator mới: standardlatent đã preflight2seed và auditT4r2/2seed/resume2rank/cachePASS; chưa có utilityr50. [GateT4](../../docs-20261109/research/screening-standard-t4.md).
Forest+quantile Torchport8tests/preflight2seed/GPUfidelityPASS; dispatcherđãmởngay. Owner02:39ưu tiênresume nhưng **cho thửnghiệmkhảthingaytrênquota dư**, thaylệnhhoãntoànbộđến07:00; batchdàichờreset. [GateforestT4](../../docs-20261109/research/screening-forest-t4.md).
N/O/P/Q và C/F/L/R/V/W/X đã DVC và lên cloud (05/11, [backup](../../../docs/operations/backup.md)). Job, owner file và bước tiếp chỉ ở [bàn giao](../../../docs/operations/continue.md).

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Phương pháp đề xuất, byte, thành phần nào đã chứng minh? | [proposal](../../../docs/research/proposal.md), [rà pipeline](../research/pipeline-review.md) |
| Nguyên nhân đã biết, đang kiểm gì? | [diagnosis](../../../docs/research/diagnosis.md), [generator đang thử](../research/screening-standard-copula.md) |
| Kết quả mới và giới hạn bằng chứng? | [F50](../../../docs/research/results-f50.md), [L50](../../../docs/research/results-l50.md), [C10](../research/results-c10.md), [X20](../research/results-x20.md), [W20](../research/results-w20.md), [benchmark r20](../../../docs/research/benchmark-r20-reference.md) |
| Cơ chế code/cấu hình chung? | [method](../../../docs/research/method.md), [protocol](../../../docs/research/protocol.md) |
| Job nào sống, bước tiếp, việc tay? | [continue](../../../docs/operations/continue.md) |
| Chạy công cụ, Kaggle, backup/khôi phục? | [operations](../../../docs/operations/README.md), [scripts](../../../scripts/README.md), [backup](../../../docs/operations/backup.md) |
| Dataset, model, máy, image? | [knowledge](../../../knowledge/README.md) |

Số đo ở kết quả; quyết định ở diagnosis/proposal/protocol; dữ kiện bất biến ở knowledge; trạng thái job chỉ ở continue.
