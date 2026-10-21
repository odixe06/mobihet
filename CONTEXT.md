# CONTEXT — MobiHet / federated IDS V2X

Đọc CONTEXT → [LESSONS](LESSONS.md) → [handoff hiện tại](docs/operations/continue.md).
[Mục lục](docs/README.md) · [Phương pháp](docs/proposals.md) · [Index file](docs/file-index.md).

## Bài toán

Xe học IDS cộng tác qua RSU với giao tiếp hữu hạn; private model độc lập, thiết kế hỗ trợ client khác kiến trúc.
Giữ 66 feature, dataset/partition/scaler gốc; mọi model đang chạy là DAGSNet. Không raw/perturbed record,
per-record activation/gradient rời client. Heterogeneous R2 thực nghiệm và contact/deadline R1 chưa chứng minh.

## Trạng thái và quyết định hiện hành — 20/10/2026

- Vét cải tiến đã chốt: `gkd + la + kdpx`, proxy q3_zl PASS cả seed42/43; 2-bit loại.
  [Kết quả/cổng đã audit](docs/methods/full-drift/results.md), [thiết kế](docs/methods/full-drift/design.md).
- Chạy protocol cuối **seed42 trước**: hai arm q3_zl la_kdpx và đối chứng gkd; 20/50/100 client; 50 round × 1 epoch
  local toàn train80%, cosineT50; chấm toàn test20% mỗi round, headline r50. Không chọn HP/checkpoint bằng official test.
- Train/chấm tách: session20c cap4 round, 50/100c cap2. Pull/audit mọi checkpoint; một round/notebook chấm,
  2 GPU và account khác. Evaluator có heartbeat tiến triển ⇒ train tiếp song song. Seed43 chờ seed42 hoàn tất.
- r24 zc/zi crash trước round1: Gaussian hash cứng pilot fixture trong final-only runtime; đã pull/audit, không checkpoint.
  Owner dừng bốn kernel còn lại; API xác nhận CANCEL_ACKNOWLEDGED. Source r25 sửa, kiểm local PASS;
  đã **push đủ sáu kernel r25**, kiểm startup ổn định. [Live](docs/operations/continue.md), [pipeline](docs/operations/final-pipeline.md).
- Giữ phương pháp/dataset/kiến trúc; chỉ tối ưu code tương đương. `--fast` đã được owner chấp thuận từ r19,
  chỉ so đối chứng cùng `--fast`. Communication cost đo trong harness; FLOPs estimate và walltime báo riêng.
- `QUEUED` = chờ GPU, không lỗi. `RUNNING`/`COMPLETE`/W&B finished không chứng minh SGD hoặc kết quả khoa học;
  cần exact kernel/version, log/heartbeat và terminal artifact audit. Log SSE dùng `kernels logs --follow`.
- Freeze mọi source/notebook đã PUSHED và 40 module legacy; sửa bằng tag mới. Resume từ committed round đã audit,
  không từ W&B summary, không quay lại r1 khi có checkpoint hợp lệ. Không nối adaptation FAIL y nguyên.
- Chỉ push **kernel Kaggle** trong phiên này; không Git/DVC push/backup/cleanup. Credentials account-scoped,
  không đổi global login/ghi secret vào docs; inline W&B chỉ ignored private notebook0600 theo quyền đã có.
- Khi không rõ yêu cầu khoa học/budget, hỏi owner trước. Việc phải làm bằng tay: báo exact bước và lỗi quan sát.

## Phiên sau

1. Mở [continue.md](docs/operations/continue.md): exact IDs, source tag, health, quota, lệnh.
2. Kiểm sáu kernel r25; QUEUED chờ, RUNNING phải có heartbeat tăng. Lỗi ⇒ pull/audit trước recovery.
3. Terminal train ⇒ pull/audit mọi round; mở chấm từng round khác account. Chấm healthy ⇒ nối train.
4. Báo riêng train r50 và eval đủ r1–50; cập nhật costs/10metrics từ artifact đã audit.

## Cần biết → mở

| câu hỏi | tài liệu |
|---|---|
| Tiếp tục ngay, kernel/checkpoint/quota/manual action? | [Handoff](docs/operations/continue.md) |
| Train/chấm tách, timing, source/tests, cost? | [Pipeline cuối](docs/operations/final-pipeline.md) |
| Kết quả nào PASS/FAIL, vì sao chọn phương pháp? | [Results](docs/methods/full-drift/results.md), [design](docs/methods/full-drift/design.md), [danh mục](docs/proposals.md) |
| Protocol, thông tin rời client, so sánh/cost? | [Protocol](docs/research/protocol.md), [luồng FL](docs/research/federated-information-flow.md), [so sánh](docs/research/comparison-fl-dagsnet-veremi.md), [cost](docs/research/computational-cost.md) |
| Dataset/kiến trúc cố định? | [Knowledge](knowledge/README.md) |
| Account/quota, lỗi kernel, layout, Git/DVC? | [Kaggle](docs/operations/kaggle.md), [sổ lỗi](docs/operations/notebook-errors.md), [layout](docs/operations/project-layout.md), [backup](docs/operations/backup.md), [artifact index](docs/operations/artifact-index.md) |
| Quyết định/kết quả lịch sử? | [Archive](docs/archive/README.md), [CONTEXT trước gọn tài liệu](docs/archive/operations/context-before-final-recovery-2026-10-20.md), [bàn giao17–20/10](docs/archive/operations/continuation-2026-10-17.md) |

Hub giữ trạng thái/quyết định/route. Số đo và commands ở docs; dữ kiện cố định ở knowledge; procedure dùng lại ở skills.
