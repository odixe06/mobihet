# P20 — tri thức có lợi không đồng đều giữa các xe

Chẩn đoán mô tả sau khi đọc officialtest; so cùng seed với C2. Không dùng test chọn method cho từng xe.
F1 mỗi xe tính lớp có mặt; dấu delta dùng tolerance1e-12. Hai seed không là significance.

| cell | xe tăng s42/s43 | xe giảm s42/s43 | xe giảm cả2seed | xe<256 giảm cả2seed |
|---|---|---|---:|---:|
| C0 real | 555/496 | 4668/4720 | 4363 | 404 |
| C1 augment | 2207/2240 | 2852/2810 | 1612 | 145 |
| C3 gate | 2703/2599 | 2339/2437 | 1244 | 128 |
| C4 margin | 2785/2783 | 2244/2247 | 1059 | 122 |
| C5 strength | 2667/2643 | 2364/2381 | 1199 | 124 |
| C6 bank_class | 1434/1439 | 3732/3717 | 2928 | 228 |
| C7 bank_input | 2014/1986 | 3114/3151 | 2203 | 136 |
| C8 proxy_bank | 1975/1935 | 3170/3210 | 2224 | 146 |
| C9 proxy_stream | 1274/1256 | 3877/3889 | 3179 | 184 |
| C10 proxy_hybrid | 2575 | 2600 | thiếu | thiếu |
| C11 bank_hard | 1961/1721 | 3220/3460 | 2176 | 168 |
| C12 project_augment | 2221/2220 | 2847/2839 | 1607 | 172 |
| C13 project_class | 2479/2412 | 2535/2592 | 1281 | 126 |
| C14 proxy_weights_bank | 1945/1917 | 3188/3230 | 2251 | 143 |
| C15 real_no_la | 523/520 | 4691/4713 | 4379 | 403 |

C10 chỉ seed42, không gọi stable gain. Số lượng xe không cân theo testrows hoặc độ lớn delta; xem mean/p10/absent/perclass cùng lúc.
Không có cơ chế chung đã chứng minh tăng mọi xe; personalized reception cần validation riêng, không oracle theo test.
Nguồn: `runs/p-study/analysis/client-transfer.json`; [phân tích chính](results-p20.md).
