# Teacher snapshot gộp cố định cho private (`frz`) + Gaussian KL tới snapshot (`gkd`)

**Trạng thái (18/10 ~04:45Z): PASS cả ba arm ở ep15 s42 (luật head; chờ owner xác nhận luật + seed 43).**
`m4b_frz` 0,7690 (+0,037 vs m4b gp; test 0,7728); `gpool_frz` 0,7717 (+0,038 vs gpool gp; test 0,7795); **`gpool_frz_gkd` 0,8030 (+0,069 vs
gpool gp, +0,031 vs gpool_frz; benign +0,200, tệ nhất +0,128; test 0,8046)**; proxy −0,011…−0,028 ([kết quả](../../methods/full-drift/results.md),
[design §Vòng 15](../../methods/full-drift/design.md)). Đề xuất từ chẩn đoán probe 14–17 (owner 17/10 ~20Z cho phép đề xuất + thử).

## Vấn đề (bằng chứng)
Probe 14/16: backbone private (kể cả train không FL) refit Linear cuối trên dữ liệu gộp → 0,81–0,84 vs private gp 0,73 ⇒ tầng quyết định là
nút thắt. Probe 15: trong một epoch local (2–5k bước), private đạt đỉnh sớm rồi trôi về local (`cont` 0,758 → 0,727) vì teacher DML là bản
proxy cũng học local; khởi tạo private bằng proxy rồi chỉ CE local mất 0,08.

## Cơ chế
KD private ← proxy gộp **đóng băng suốt epoch** (snapshot vừa nhận); bản proxy vẫn train DML và được gộp như cũ. `gkd`: mẫu Gaussian gộp
(gpool) học KL tới snapshot thay nhãn cứng (trọng số KD thích nghi như KD trên dòng thật). Không KD private trước snapshot đầu (warm).
Private học prior toàn cục từ teacher ⇒ luật triển khai head (không cộng log π_g − log π_k lần nữa).

## R1 / R2 / mô hình
0 byte thêm (`frz`); `gkd` dùng moment gộp đã có (C1 142.752 B/client một lần). Chỉ KD logit + mẫu input-space ⇒ private tùy kiến trúc.
Không đổi DAGSNet.

## Kỳ vọng / dấu hiệu sai
Probe 15 (1 epoch từ ep15): kdfix head +0,043, cov_gauss head +0,065, client tệ nhất +0,03 / +0,11. Cổng ep15 gain + worst so control gp.
Sai nếu từ đầu tới ep15 head ≤ control + 0,010 (lợi probe chỉ là trạng thái tạm). Tiền lệ: `kdcap_frz` vòng 8 FAIL do cap KD + luật gp.
