# swappos — tráo khối receiver + đặt 0 vị trí

**Trạng thái (17/10):** PASS s42 + s43 (DML); Gaussian full +0,052.

## Cơ chế
Với xác suất 0,5, thay khối 9 feature receiver của dòng bằng khối của một receiver khác **cùng client**; 4 vị trí = 0.
Thay nuisance thay vì xóa hẳn (khác m4b).

## Trao đổi
0 byte/contact; chỉ phía client.

## Bằng chứng (audited)
- DML s42 ep6: private gp +0,033, proxy +0,049; s43: private +0,034, proxy +0,031 (PASS cả hai seed).
- Gaussian full s42 ep6: +0,052; private EMA thêm +0,036. Pilot r20: DML private +0,019/+0,025, Gaussian +0,016–0,018.
- Không hơn m4b ở DML (0,6985 so 0,7010 s42).

## Việc tiếp
Gaussian swappos s43 + đối chứng Gaussian s43 (vòng 8 `r`). Biến thể hai view + JS: [response-consistency](response-consistency.md).

## Nguồn
Nội bộ: [AntiMem](../../methods/anti-memorization/results.md), [FullDrift](../../methods/full-drift/results.md).
