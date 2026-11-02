# Continue — bàn giao sống, 03/11/2026 — X20 đóng; chờ duyệt 50 round

Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → [vận hành](lessons.md).
Bàn giao trước (P/Q/R/V chi tiết, callback cũ): [snapshot](../../archive/handoffs/2026-11-02-1700-before-w20-launch.md).

## Quyền owner đang có (02/11)

- Sau W20: chạy các kiểm nghiệm cần thiết cho phương pháp cuối → X20 (đã xong). 50 round chưa được duyệt.
- Step floor hợp lệ khi bước thêm chỉ synthetic; 1 epoch trên dữ liệu local giữ như repo benchmark.
- Chuẩn FL: không chia sẻ dữ liệu local; server chỉ dùng tổng toàn mạng ([proposal §Quyền riêng tư](../research/proposal.md#quyền-riêng-tư)).
- Không đổi protocol/HP; không chọn round/HP bằng official test; không thêm run ngoài batch đã khóa.
- Credential Kaggle process-scoped, không đổi global login. W&B online, key chỉ trong notebook private. Commit không có trailer `Co-Authored-By`.

## Job sống

Không có. X20 đóng 03/11 02:50: 8/8 COMPLETE, audit PASS (`runs/x-study/closure.json`, log `runs/x-study/pipeline.log`).
Artifact `artifacts/x20/` (cùng r20/v20/w20) chưa DVC.

## Bước tiếp

1. **Cần owner duyệt:** batch 50 round theo [protocol](../research/protocol.md), 2 seed: core+bank (phương pháp) và core
   (đo bank ở 50 round) cùng baseline benchmark; tùy chọn ô trộn kiến trúc. 20 round ≈ 3,75 h train + 0,35 h eval/diag
   → 50 round ≈ 10 h/kernel, sát giới hạn 12 h: cần checkpoint/resume hoặc tách session trước khi push. Quota reset 07/11.
   Quy tắc giữ bank: [diagnosis §Tiếp theo](../research/diagnosis.md#tiếp-theo).
2. Backup R/V/W/X khi OneDrive còn chỗ (~30 GB+ cần).

## Kết quả mới nhất

[X20](../research/results-x20.md): 8/8 audit PASS. core PASS, bank PASS (sát ngưỡng, lợi thu hẹp theo round),
nckd_local xác nhận cơ chế lớp vắng, full trượt gate → phương pháp = core + bank, không realKD.

## Việc tay cho owner

- Duyệt batch 50 round (quota/thời gian).
- Backup R/V/W/X: giải phóng hoặc mở rộng OneDrive; sau N/O/P/Q còn ~15 GB.
