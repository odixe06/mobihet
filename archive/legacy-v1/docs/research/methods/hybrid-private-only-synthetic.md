# Hybrid private-only — Gaussian dạy riêng private trên nền DML-m4b

**Trạng thái (17/10 ~19:05Z):** PASS ba cổng ep6 hai seed nhưng **horizon ep15 FAIL** (vs m4b +0,054 ep1 → +0,016 ep6 → +0,002 ep15; test ep15 0,7486 vs 0,7478; proxy thấp hơn −0,007) — lợi là tăng tốc đầu, không tăng mức cuối. gw2 FAIL. Owner 17/10 ~19:50Z: **giữ headline DML-m4b + gpool**. [Kết quả lc](../../methods/full-drift/results.md).

## Cơ chế
Giữ proxy và lịch tổng hợp DML-m4b; thêm CE trên 256 mẫu Gaussian **chỉ cho private** (hard label). Proxy chỉ học real;
KD chỉ trên dòng real. Hybrid cũ (proxy cũng học synthetic) pilot r20 hai seed hơn từng thành phần nhưng interaction âm,
full phẳng (proxy phẳng từ r10) ⇒ owner dừng.

## Trao đổi
Proxy + bootstrap moment (giai đoạn B: q4 ≈417 kB + moment53 ≈190 kB).

## Thiết kế kiểm
`gpool` (moment gộp), `gown` (moment riêng), `replay` (256 dòng local cân bằng lớp, cùng forward). Cổng: gpool phải hơn
`fd_dml_m4b`, hơn gown và hơn replay (mỗi cái ≥ +0,010, benign ≥ −0,005).

## Bằng chứng
Audited ([kết quả vòng 8–10](../../methods/full-drift/results.md)), private gp ep6 s42 / s43: `gpool` 0,7174 / 0,7175;
vs `fd_dml_m4b` +0,016 / +0,024 (benign +0,003 / −0,000, client tệ nhất −0,003 / +0,018); vs `gown` +0,030 / +0,030;
vs `replay` +0,033 / +0,032. Proxy head ±0,003. Byte thêm 4,2 MiB/6 epoch (+1,1 %). Lợi đến từ moment gộp của client
khác (không phải thêm CE hay cân bằng lớp local).
 Hybrid cũ: [pilot](../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/results/report.md).

## Nguồn
[Plan §4.4](../../archive/research/algorithm-improvement-plan-2026-10-17.md); [hồ sơ Hybrid](../../methods/gaussian-proxy-hybrid/README.md).
