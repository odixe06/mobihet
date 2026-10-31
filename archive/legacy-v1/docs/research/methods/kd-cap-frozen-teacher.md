# KD cap + frozen teacher — cách private hấp thụ teacher

**Trạng thái (17/10 ~14:00Z):** FAIL cả hai seed (kdcap, frz); vòng 11 `kd2` (+0,001) và `kdpx` (−0,008, proxy +0,031) FAIL ⇒ hướng trọng số KD đã cạn.

## Cơ chế
Source dùng `w = 1/stopgrad(CE_priv + CE_proxy)` không trần thực dụng: CE giảm 1→0,1 ⇒ hệ số tăng 10× dù teacher có thể
đang học thuộc receiver (giả thuyết, chưa đo). A: `w_cap = min(w, 1)`. B: private học teacher = proxy gộp đầu round
(eval, đóng băng) thay proxy đang train; proxy vẫn học KL từ private. Plan: tắt KD private khi chưa có snapshot gộp
(ep1); control `wu` = A + cùng warm-up.

## Trao đổi
0 byte/contact thêm. B thêm 1 bản proxy (≈1,59 MB RAM) + 1 forward/batch.

## Bằng chứng
Audited ([kết quả vòng 8–10](../../methods/full-drift/results.md)), private gp ep6, s42 / s43:
- `kdcap` vs m4b −0,028 / −0,025 (client tệ nhất −0,061 / −0,052) ⇒ FAIL; **proxy head +0,024 / +0,033**.
- `kdcap_frz` vs `wu` −0,087 / −0,088 ⇒ FAIL; private **head** 0,7135 / 0,6978 (> gp m4b 0,7010 / 0,6940).
- `wu` vs `kdcap` +0,000 / −0,004 (warm-up không đáng kể).
- Baseline m4b (rerun y, bit-exact vòng 3): `w_raw` median 2,39 → 4,14, cap_hit 0,955 → 1,0, ‖w∇KL‖/‖∇CE‖ 0,79 → 1,23.
Kết luận: KD mạnh giúp private (giả thuyết "w quá lớn" bị bác cho client); KL proxy←private mạnh hại proxy.
Frozen teacher làm private gần prior toàn cục nên luật gp sửa hai lần; muốn dùng phải khóa rule mới trước (vd head).

Sơ bộ W&B r2 (s42): cap chặn gần như mọi bước (`w` median 2,6–3,4, cap_hit 0,95–1,00);
frz r1 = wu r1 bit-exact; frz r2 private **head 0,584** > wu 0,549 nhưng **gp 0,244**: teacher gộp mang prior toàn cục,
luật gp (khóa) cộng `log π_g − log π_k` nên sửa quá tay. Nếu giữ hướng này cần calibration riêng (split riêng, khóa trước).
Log mới: median/p95/max `w_raw`, tỷ lệ chạm trần, KL hai chiều, probe 512 dòng train (‖∇CE‖, ‖∇KL‖, tỷ số).
Rerun baseline `fd_dml_m4b` (vòng 10 `y`, anhtruong1909) cho log của baseline chưa cap.

## Cổng (khóa [design §Vòng 8](../../methods/full-drift/design.md))
`kdcap` vs `fd_dml_m4b`; `kdcap_frz` vs `kdcap_wu` (cô lập nguồn teacher): private gp Δ ≥ +0,010, benign ≥ −0,005,
client tệ nhất ≥ −0,010. Giả thuyết yếu nếu cap_hit median < 0,5.

## Nguồn
FedKD (hệ số theo độ đúng) [arXiv:2108.13323](https://arxiv.org/abs/2108.13323); FML [arXiv:2006.16765](https://arxiv.org/abs/2006.16765).
Teacher đóng băng là adaptation của dự án. [Plan §3.2, §4.2](../../archive/research/algorithm-improvement-plan-2026-10-17.md).

## Vòng 11 (audited 17/10 ~14:00Z, s42, so `gpool`)
`gpool_kd2` (KD private×2) private gp +0,001 ⇒ FAIL; `gpool_kdpx` (chỉ cap KL proxy←private) private −0,008, proxy head
+0,031 ⇒ FAIL. Teacher tốt hơn không làm private tốt hơn; private đã bão hòa KD trên dòng local.
