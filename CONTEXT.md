# MobiHet — điểm vào dự án

Đọc CONTEXT → [LESSONS](LESSONS.md) → [bài học vận hành](docs/operations/lessons.md) → [bàn giao sống](docs/operations/continue.md).
Không nạp `archive/` mặc định.

Federated IDS V2X16lớp trên VeReMi NextGen v4:5343xe nhận,77đầu vào, mỗi xe giữ model riêng.
Chỉ trao thống kê cộng được/xác suất; server gộp toàn mạng; không trao dòng thật/model/gradient/BN.

## Trạng thái kiểm16/11/2026

**Phương pháp chính=T3, gọi MobiHet**: copula KM16+lite6+Q8, liều synthetic theo cỡ xe/pha lịch,
S64→64/M64→32/L48→16 từr26,floor16. [Cơ chế](docs/research/method.md), [quyết định](docs/research/proposal.md).
DAGSNet đủr50 cả3seed41/42/43;BERT đủr50 cả3seed, CM+state2rankPASS.
B20/B21 đã đóng; không nối lại từ trạng thái waiter cũ. Job hiện hành chỉ ở [continue](docs/operations/continue.md).

Sau reboot16/11: owner yêu cầu hoàn tất, không cần dừng/reboot; DVCpush lại chờmount và pFedES41p4 đãpush/watch1200s do repoBERT sở hữu.
Phiên16/11: owner ưu tiên pull/audit đa luồng, đồng bộ tài liệu hai repo và Notion Proposal2, rồi push Git+DVC.
Không xóa artifact, không đổi protocol/source/HP đã khóa. Chưa có việc tay bắt buộc.

## Quyết định đang áp dụng

- Macro-F1 là primary;guard p10/low256/TD/benign/absent report-only, vẫn audit đủ. Headline50round;không chọn best round/HP qua test.
- CBB là control cũ, không dùng kết quả CBB làm số liệu T3. Baseline DAGSNet3seed đã audit;SCAFFOLD loại.
- Mixed-model full-scale chưa chạy;privacy bootstrap hoãn;chưa triển khai SecAgg/DP.
- Không mở lại họ cơ chế đã FAIL bằng tên mới. Luật/thang sim→pilot→r50 trong design đã khóa; việc collect hiện tại không tự launch training.
- COMPLETE/exit0 không thay artifact audit;resume cần checkpoint/RNG cả2rank cùng round. Không sửa file worker đang sở hữu.

## Cần biết → mở

| Câu hỏi | Nguồn |
|---|---|
| Job nào sống, ai sở hữu, bước kế/việc tay? | [continue duy nhất](docs/operations/continue.md) |
| T3 là gì, vì sao chọn? | [method](docs/research/method.md), [proposal](docs/research/proposal.md), [B21](docs/research/screening-dose-b21.md) |
| Muốn cải thiện phương pháp, họ nào đóng/mở? | [sổ tay](docs/research/method-building-brief.md), [index đã thử](docs/research/tried-methods-index.md) |
| Generator/codec/đề xuất nào đã thử? | [generators](docs/research/generators.md), [codec](docs/research/screening-codec.md), [catalog](docs/research/improvement-proposals.md) |
| Dataset/model/máy và protocol chung? | [knowledge](knowledge/README.md), [protocol](docs/research/protocol.md) |
| Chạy/collect/backup thế nào? | [operations](docs/operations/README.md), [scripts](scripts/README.md), [backup](docs/operations/backup.md) |

Kết quả/phương pháp→docs/research;job sống→continue;dữ kiện bất biến→knowledge;bài học→hai lessons.
[Bản đồ tài liệu](docs/README.md) · [README](README.md).
