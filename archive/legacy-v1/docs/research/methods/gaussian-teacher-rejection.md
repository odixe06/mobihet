# B1 — Bỏ mẫu Gaussian mà proxy gộp bác bỏ

**Trạng thái (17/10 ~23:10Z):** s42 ep6 PASS (+0,0108, tệ nhất +0,012); s43 ep6 +0,0082 (benign +0,005, tệ nhất +0,018) — FAIL cổng gain
đã khóa nhưng cùng chiều; **horizon ep15 s42 PASS** (0,7455: +0,0132 vs m4b@e15, +0,0113 vs gpool@e15; test 0,7599; tệ nhất −0,004/−0,008) ([kết quả](../../methods/full-drift/results.md)). Trước đó: vòng 13 chạy `fd_dml_m4b_gpool_rej` (s42 ep6 từ đầu,
[design §Vòng 13](../../methods/full-drift/design.md)). Owner duyệt 17/10 ~16Z (đề xuất B, nhánh lọc). Nhánh B2 (hỗn hợp
moment) = [P2](gaussian-mixture-modes.md).

## Vấn đề (bằng chứng)
Đo local 17/10: proxy gpool s42 ep6 chỉ đồng ý 53 % mẫu Gaussian gộp (benign 25 %, positionMirroring 3 %, timeDelay 0 %), private
khớp 78 % ⇒ nhiều mẫu nằm ở vùng mà mô hình quần thể gán lớp khác; private học nhãn cứng ở đó có thể làm nhòe ranh giới.

## Cơ chế
Từ round 2, proxy snapshot nhận được đầu epoch (eval, no-grad) chấm 256 mẫu; mẫu có argmax ≠ nhãn bị bỏ khỏi CE. Mẫu được
nhận của lớp c mang đúng trọng lượng của lớp c (rejection sampling trong lớp); lớp không còn mẫu nào trong bước thì mất phần
đó. Round 1 = gpool (bit-exact, test `r13`). 0 byte, R2 không đổi. Log tỷ lệ nhận tổng và từng lớp.

## Dấu hiệu sai
Δ private gp < +0,010 so gpool; lợi chỉ ở lớp proxy vốn mạnh còn lớp proxy yếu (timeDelay) mất mẫu và giảm.

## Khác bản đã FAIL
[Density ratio V2-C](../../archive/methods/gaussian-density-ratio/README.md) học thêm một DAGSNet phân biệt real/Gaussian để
đánh trọng số (ESS thấp, FAIL). B1 không học mạng mới, không đo độ "thật" của mẫu: chỉ lọc nhất quán nhãn bằng proxy gộp
sẵn có. Rủi ro chung: lọc theo teacher yếu ở lớp khó.
