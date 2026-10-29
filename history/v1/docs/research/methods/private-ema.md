# Private EMA + BN tính lại

**Trạng thái (17/10):** Có lợi cho Gaussian không mask; hại DML và Gaussian-m4b.

## Cơ chế
EMA trọng số private (half-life 2 epoch), BN tính lại trên 65.536 dòng train local cố định; chấm như role riêng.

## Trao đổi
0 byte; recipe local (không phải lợi FL).

## Bằng chứng (audited)
Pilot r20: Gaussian +0,035–0,039; DML/Hybrid proxy EMA −0,028…−0,038. Full ep6: Gaussian +0,024, swappos +0,036;
DML private −0,026…−0,047 mọi arm (s42, s43); **Gaussian-m4b −0,010 (s42), −0,021 (s43)**.

## Kết luận
Không dùng cho DML; không ghép với m4b. Thiết kế EMA-backbone + head fit riêng: [ema-deployment-recipe](../../methods/ema-deployment-recipe/design.md) (chưa triển khai).

## Nguồn
Nội bộ: [AntiMem](../../methods/anti-memorization/results.md), [FullDrift](../../methods/full-drift/results.md).
