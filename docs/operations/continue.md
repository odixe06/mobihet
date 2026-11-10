# Bàn giao sống — kiểm 10/11/2026 21:30 UTC+7

Đọc [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [operations lessons](lessons.md) trước. Đây là **bàn giao hiện hành duy nhất**;
state/log/receipt thật ưu tiên hơn tài liệu. Lịch sử 09–10/11 (codec study, combo r20/r50, waiter):
[snapshot 10/11](../../archive/handoffs/2026-11-10-codec-combo.md); trước đó [snapshot 09/11](../../archive/handoffs/2026-11-09-km-lite-b9-b10.md).

## Vừa xong

- Combo codec r50 (4 arm-seed) xong, pull + audit 21:02: **CBB PASS 2 seed, CQ8 PASS 2 seed**
  (`scripts/analyse_codec_combo50.py` → `runs/codec-combo50/{audit,verdict-r50}.json`; bảng ở
  [screening-codec](../research/screening-codec.md) mục "Kết quả combo r50").
- **Owner 10/11 chốt CBB (KM16 + lite6 + QQ8) làm phương pháp chính** → [proposal](../research/proposal.md), [method](../research/method.md).
- Tài liệu đồng bộ theo CBB (CONTEXT, proposal, method, generators, improvement-proposals, km50, km-lite, codec, research README).

## Trạng thái worker và Kaggle

Không job sống. Watcher `runs/codec-combo50/watch.sh` (PID 6315) và waiter `analyse-after-done.sh` đã kết thúc (`DONE`, `ANALYSED`).

## Việc tiếp

Chưa có việc train được duyệt. Owner 10/11 tạm gác seed mới và model mới. Chờ owner giao việc.

## Quyền và ràng buộc hiện hành

- Owner 09/11: **PASS → tự chạy bậc scale kế** (thang ở CONTEXT) cho ứng viên mới khi có.
- Credential theo process (`kaggle_as.py`), global login giữ nguyên; không nhúng W&B.
- Git/DVC push chỉ khi owner yêu cầu; không xóa artifact.
- Owner loại: trao model/proxy DAGSNet. Compute tại xe không tụt hạng (Core 4,00×F; ngưỡng V2X-HtFL 4,11×F).

## Backup

10/11 tối: Git `main` đã push; DVC đủ đến codec-combo20/50 (in sync OneDrive). Chi tiết: [backup](backup.md).

## Việc tay

Không có.
