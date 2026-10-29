# Continue — bàn giao sống, 30/10/2026

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này. Sửa tại chỗ.

## Đang chạy — sàng lọc 10 round (W&B `mobihet-v4`, SHA `6c3536d`, budget 4 h)

Full 5.343 xe, lịch LR 50 round, chỉ r1–r10 (`--until 10`), seed 42, `group_rows` 4.096, `expandable_segments`.

| kernel | account | arm |
|---|---|---|
| `mh-s0-dml` | odixe0502 | S0 `dml` (đối chứng, arm v1) |
| `mh-s1-srv` | odixe0502 | S1 `srv`, ensemble theo n_{k,c} |
| `mh-s2-srvclient` | minhtriethihi | S2 `srv`, `agg_soft=client` (n_k) |
| `mh-s3-cheap` | minhtran0601 | S3 `cheap` |
| `mh-s4-none` | trietbackup | S4 `none` |

Smoke local (100 xe, W = 1, compile, r3; chỉ kiểm chạy được, không dùng làm cổng): private dml 0,801 / srv **0,828** / cheap 0,815 /
none 0,755 (r3, 30 xe sau sửa NaN); proxy dml 0,823 / srv 0,772 / cheap 0,408; byte lên dml 133 KB / srv 43 KB / cheap 119 KB / none 0,5 KB.

## Bước tiếp

1. Collect r10 năm arm ⇒ áp cổng [improvement-plan §3](../research/improvement-plan.md) (S1/S3 noninferior vs S0 + FLOPs ≤ 0,7×).
2. Arm thắng ⇒ D1 (mixup), D2 (BN inversion), E1–E3 (distill) — mỗi lần một yếu tố, 10 round.
3. Arm cuối 50 round; so baseline repo `fl_dagsnet_veremi_v2`.

## Việc tay (owner)

- (nếu chưa) dừng `trietbackup/fl2-mobihet` trên Kaggle.
