# Chẩn đoán — nguyên nhân đã biết và phép thử tiếp (02/11/2026, sau W20)

Mục tiêu: own-test từng xe, nâng xe ít mẫu/non-IID, băng thông thấp, schema dùng được cho mô hình khác nhau.
Hướng phương pháp: [proposal](proposal.md). Số đo: [W20](results-w20.md), [R20/V20](results-r20-v20.md), [Q20](results-q20.md), [P20](results-p20.md).
Lịch sử M/N/O/P chi tiết đã archive; bài học còn hiệu lực ở [LESSONS](../../LESSONS.md).

## Nguyên nhân đã xác lập

| hiện tượng | cơ chế | bằng chứng |
|---|---|---|
| Synthetic copula là nguồn lợi chính | thống kê toàn mạng lấp lớp vắng/ít mẫu tại xe | val sạch: CE→augment +.0226 pooled, lớp vắng 0→.76 |
| realKD hại timeDelay/lớp vắng | phần target-class của KL(Q[y]‖p) kéo xác suất lớp thật về Q[y]_y; Q[13]≈(.74 timeDelay, .25 benign) | replay cùng teacher −.05 timeDelay; val sạch −.077 |
| Hại realKD lớn gấp đôi khi live | vòng feedback: học sinh nhầm hơn → uplink → Q[13,13] r20 .739 (noReal) → .581 (binary) | Q live vs replay R |
| Margin chỉ "tốt" khi live | giảm trọng số realKD ở lớp margin thấp → teacher ít nhiễm; khi teacher cố định margin ≤ binary | R8/R9 vs R6/R7, R10/R11 |
| timeDelay khó tổng quát | benign↔timeDelay gần như không để lại mâu thuẫn trong đặc trưng; có cả ở CE không KD | V0R recall train .886 → val .267 |
| Xe nhỏ thiếu tối ưu | 1 epoch = 2–4 bước/round; thêm bước chỉ synthetic tới 16 → xe<256 +.018–.027, lớp vắng +.12–.14 | W20 step_floor |
| Feedback realKD đến từ phần target-class | NCKD giữ Q[13,13] .742 = noReal; live = replay | W20 live/replay_real_nckd |
| KD dòng thật còn đè lớp vắng | NCKD khớp phân bố lớp sai theo Q[y]; lớp vắng có khối lượng teacher nhỏ → logit bị ép trên input local; lỗi lớp vắng dồn sang lớp xe có .153→.192 | W20 (giả thuyết, X20 kiểm) |
| Đổi loss synthetic chỉ dời ngưỡng timeDelay/benign | hard+NCKD: recall +.06, precision −.09, benign→timeDelay ×1,85, Q[13,13] giảm | W20 synth_nckd |
| Chênh từng xe phần lớn là ồn | cùng công thức/seed, ~250 xe dao động >.05 mỗi chiều giữa hai phiên | repeat R0–R3 |
| 97% byte là bootstrap | histogram dày fp32 + moment Gaussian không dùng; nén còn 1,02 MB, utility giữ | W20 boot_compact |

## Đang kiểm — X20

| giả thuyết | ô | đọc kết quả |
|---|---|---|
| floor + nén cộng được | core | low256/p10 trong spread W6/W7, pooled trong spread stream |
| input-bank còn lợi khi có floor | bank | p10/low256 so core |
| hại lớp vắng do NCKD ép logit lớp vắng | live_real_nckd_local | absent so Q6/Q7 và W2/W3 |
| realKD không-hại có cộng vào combo | full | chỉ đọc nếu nckd_local qua gate |

Thiết kế/dấu hiệu bác bỏ: [screening-x20](screening-x20.md). Trạng thái job: [continue](../operations/continue.md).

## Sau X20

1. Combo chọn theo [proposal](proposal.md#sau-x20--quy-tắc-chọn), 2 seed × 50 round cùng baseline cho bảng benchmark.
2. Nếu xe yếu vẫn xa APFed ở 50 round: bộ nhớ truy vấn/bank tích lũy cùng ngân sách byte, reliability từ validation local.
3. Ô trộn kiến trúc (≥2 kiến trúc client) trước khi tuyên bố hỗ trợ heterogeneous.
4. Bảo mật: tuyên bố "tương thích secure aggregation, chưa DP"; byte SecAgg ước tính ở proposal.
5. timeDelay: đặc trưng thời gian/ngữ cảnh ở repo features — ngoài phạm vi cơ chế FL.

Quy tắc đọc: mọi hiệu so với độ ồn phiên ([R20/V20 §1](results-r20-v20.md#1-độ-ồn-giữa-các-phiên-cùng-công-thức-cùng-seed));
không cộng gain các ô; không chọn round/HP bằng official test; hai seed không phải significance.
