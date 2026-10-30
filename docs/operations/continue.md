# Continue — bàn giao sống, 30/10/2026

**File duy nhất cho "đang ở đâu, làm gì tiếp".** Đọc [CONTEXT](../../CONTEXT.md) → [LESSONS](../../LESSONS.md) → trang này. Sửa tại chỗ.

## Sàng lọc 10 round — xong (30/10 ~06:45 giờ VN)

Kết quả + cổng: [screening-r10](../research/screening-r10.md). S1 FAIL sát ngưỡng, S3 FAIL, **S4 `none` tốt nhất** (r10 0,8679, 0,50 × FLOPs S0, 544 B/round).
Kernel `mh-s0..s4` đều COMPLETE (S0 dừng budget r9).

## Xong — sàng lọc bộ sinh synthetic trên S4 `none` (kết quả + cổng: [screening-r10](../research/screening-r10.md) §G; G4 copula +0,0053, trần oracle +0,0123, G0 off −0,0182)

<!-- bảng push giữ để tra kernel -->
### Push (push 30/10 ~07:30 VN, SHA `f86e137`, budget 4 h)

Owner 30/10: nền = S4 `none`, chạy cả 5 song song. hp `synth` ([methods/mobihet.py](../../methods/mobihet.py) docstring). Đối chứng = S4
(`gauss`, r10 0,8679, p10 r10 0,5671). Một yếu tố/arm, 10 round, full K.

| kernel | account | synth | câu hỏi / cổng (r10 vs S4) |
|---|---|---|---|
| `mh-g0-off` | catbaochau | off | synthetic có giúp không (mô tả) |
| `mh-g1-oracle` | catbaochau | oracle (dòng thật client khác, không triển khai được) | trần: **G1 − S4 < +0,005 ⇒ dừng mảng bộ sinh** |
| `mh-g2-short` | nguyenminhtran060113 | short (ưu tiên lớp xe thiếu) | gain Δ ≥ +0,010, p10 không giảm > 0,010 |
| `mh-g3-mix` | annhannh331 | mix (FedMix) | như G2 |
| `mh-g4-copula` | annhtnt | copula (mở lại v1) | như G2 |

Smoke local 100 xe r2 (chỉ kiểm chạy): gauss 0,838 / off 0,815 / oracle 0,847 / short 0,837 / mix 0,835 / copula 0,844.
Owner 30/10: không proxy vẫn giữ được model khác nhau mỗi xe (chỉ trao đổi xác suất 16×16 + thống kê input) — xét về phương pháp,
không cần chạy kiểm chứng; nếu cần: arm trộn độ rộng W theo xe.

## Bước tiếp

1. ~~Collect r10, áp cổng~~ (xong). Chờ owner chọn arm nền cho D/E (S1, S3 FAIL).
2. Arm nền ⇒ D1 (mixup), D2 (BN inversion), E1–E3 (distill) — mỗi lần một yếu tố, 10 round.
3. Arm cuối 50 round; so baseline repo `fl_dagsnet_veremi_v2`.

## Việc tay (owner)

- (nếu chưa) dừng `trietbackup/fl2-mobihet` trên Kaggle.
