# m4b — đặt 0 khối 13 feature receiver/vị trí

**Trạng thái (17/10):** PASS seed 42 + 43, horizon ep15 PASS; headline.

## Cơ chế
Đặt 0 (sau z-score) 9 feature noise/profile receiver + 4 vị trí tuyệt đối, cho train/synthetic/eval, private và proxy.
Bỏ shortcut "vân tay receiver": ở full mỗi epoch ≈4.200 bước/client nên private/proxy học thuộc receiver của xe.
Không xóa cột, giữ 66 vị trí input; **có thay đổi thông tin đầu vào** (không gọi là full66 nguyên trạng).

## Trao đổi
0 byte thêm (R1); chỉ phía client, mọi kiến trúc (R2).

## Bằng chứng (audited, group)
| so sánh | private gp | proxy | |
|---|---|---|---|
| s42 ep6 vs `fd_dml` | +0,036 (0,7010) | +0,052 | PASS |
| s43 ep6 vs `fd_dml` | +0,031 (0,6940) | +0,039 | PASS |
| s42 ep15 vs `fd_dml@e15` | +0,036 (0,7323) | +0,086 (0,851) | PASS horizon |
| Gaussian-m4b s42 vs Gaussian | +0,067 | — | PASS |
Benign proxy không sụp theo epoch (0,76–0,78 tới ep15). Pilot r20 (100 bước/round) từng FAIL ở proxy ⇒ chỉ lộ ở full.
Cộng được: + q4 (±0,000), + `gpool` (+0,016 s42 / +0,024 s43).

## Việc tiếp / dấu hiệu sai
FL control (vòng 8–10) PASS hai seed: DML-m4b hơn proxy không gộp +0,043/+0,040. Horizon seed 43 ep15 PASS (private
+0,025, proxy +0,063, 17/10). Full r50 m4b chưa chạy.

## Nguồn
Nội bộ: [AntiMem](../../methods/anti-memorization/results.md), [FullDrift vòng 3/5/6](../../methods/full-drift/results.md), [LESSONS](../../../LESSONS.md).
