# Continue — bàn giao sống, 30/10/2026

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này. Sửa tại chỗ.

## Sàng lọc 10 round — xong (30/10 ~06:45 giờ VN)

Kết quả + cổng: [screening-r10](../research/screening-r10.md). S1 FAIL sát ngưỡng, S3 FAIL, **S4 `none` tốt nhất** (r10 0,8679, 0,50 × FLOPs S0, 544 B/round).
Kernel `mh-s0..s4` đều COMPLETE (S0 dừng budget r9).

## Bước tiếp

1. ~~Collect r10, áp cổng~~ (xong). Chờ owner chọn arm nền cho D/E (S1, S3 FAIL).
2. Arm nền ⇒ D1 (mixup), D2 (BN inversion), E1–E3 (distill) — mỗi lần một yếu tố, 10 round.
3. Arm cuối 50 round; so baseline repo `fl_dagsnet_veremi_v2`.

## Việc tay (owner)

- (nếu chưa) dừng `trietbackup/fl2-mobihet` trên Kaggle.
