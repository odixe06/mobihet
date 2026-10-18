# Response consistency — hai view receiver + JS

**Trạng thái (17/10 ~04:40Z):** FAIL (JS không thêm, s42) — dừng.

## Cơ chế
View a giữ khối 9 feature receiver, view b lấy nguyên khối của receiver khác cùng client; 4 vị trí = 0 cả hai.
`0,5[CE_a + CE_b] + λ·JS(p_a, p_b)` cho private; KL DML trung bình hai view; control λ=0 cùng hai forward.
Bản 13/10 (ResponseRisk) đề xuất chia sẻ trọng số consistency qua RSU — không mở kênh scalar đó.

## Trao đổi
0 byte; compute ≈2× (batch đôi).

## Bằng chứng
Audited ([kết quả vòng 8–10](../../methods/full-drift/results.md)): `cons` vs `cons0` private gp +0,003 (benign −0,003) ⇒ FAIL; `cons` vs m4b −0,001.
 ResponseRisk: fidelity probe PASS, chưa train. Giả định chưa kiểm: thay khối receiver không đổi nhãn.

## Cổng
`cons` vs `cons0` gain (cô lập JS); `cons` vs `fd_dml_m4b` mô tả (thua m4b ⇒ không ưu tiên).

## Nguồn
Ross et al., *Right for the Right Reasons*, IJCAI 2017 [link](https://www.ijcai.org/proceedings/2017/371);
*R-Drop*, NeurIPS 2021 [paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf).
Nội bộ: [ResponseRisk](../../archive/research/new-fl-primary-sources-2026-10-13.md), [hồ sơ](../../methods/receiver-response-consistency/README.md), [plan §4.5](../../archive/research/algorithm-improvement-plan-2026-10-17.md).
