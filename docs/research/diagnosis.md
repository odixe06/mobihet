# Chẩn đoán — nguyên nhân đã biết và phép thử tiếp (02/11/2026)

Mục tiêu: own-test từng xe, nâng xe ít mẫu/non-IID, băng thông thấp, schema dùng được cho mô hình khác nhau.
Hướng phương pháp: [proposal](proposal.md). Số đo: [R20/V20](results-r20-v20.md), [Q20](results-q20.md), [P20](results-p20.md).
Lịch sử M/N/O/P chi tiết đã archive; bài học còn hiệu lực ở [LESSONS](../../LESSONS.md).

## Nguyên nhân đã xác lập

| hiện tượng | cơ chế | bằng chứng |
|---|---|---|
| Synthetic copula là nguồn lợi chính | thống kê toàn mạng lấp lớp vắng/ít mẫu tại xe | val sạch: CE→augment +.0226 pooled, lớp vắng 0→.76 |
| realKD hại timeDelay/lớp vắng | phần target-class của KL(Q[y]‖p) kéo xác suất lớp thật về Q[y]_y; Q[13]≈(.74 timeDelay, .25 benign) | replay cùng teacher −.05 timeDelay; val sạch −.077 |
| Hại realKD lớn gấp đôi khi live | vòng feedback: học sinh nhầm hơn → uplink → Q[13,13] r20 .739 (noReal) → .581 (binary) | Q live vs replay R |
| Margin chỉ "tốt" khi live | giảm trọng số realKD ở lớp margin thấp → teacher ít nhiễm; khi teacher cố định margin ≤ binary | R8/R9 vs R6/R7, R10/R11 |
| timeDelay khó tổng quát | benign↔timeDelay gần như không để lại mâu thuẫn trong đặc trưng; có cả ở CE không KD | V0R recall train .886 → val .267 |
| Xe nhỏ còn thiếu tối ưu | 1 epoch = 2–4 bước/round; r15→20 mean xe<256 còn +.03–.05 | mọi batch |
| Chênh từng xe phần lớn là ồn | cùng công thức/seed, ~250 xe dao động >.05 mỗi chiều giữa hai phiên | repeat R0–R3 |
| 97% byte là bootstrap | histogram dày fp32 + moment Gaussian không dùng | đo offline |

## Đang kiểm — W20

| giả thuyết | ô | đọc kết quả |
|---|---|---|
| bỏ phần target-class giữ lợi other15 mà không hại timeDelay | replay_real_nckd, live_real_nckd | so R4–R7, Q6/Q7/Q10/Q11; xem Q[13,13] live |
| nhãn cứng + non-target cho synthetic giúp timeDelay | synth_nckd | so ba phiên stream mỗi seed |
| xe ít bước thiếu tối ưu | step_floor | low256/p10 vượt spread .003/.006 |
| downlink bootstrap nén giữ utility | boot_compact | mọi chỉ số trong spread |

Thiết kế/dấu hiệu bác bỏ: [screening-w20](screening-w20.md). Trạng thái job: [continue](../operations/continue.md).

## Sau W20

1. Combo các thành phần qua gate, 2 seed × 20 round; rồi 50 round cùng baseline cho bảng benchmark.
2. Nếu xe yếu vẫn xa APFed: bộ nhớ truy vấn/bank tích lũy cùng ngân sách byte, reliability từ validation local.
3. Ô trộn kiến trúc (≥2 kiến trúc client) trước khi tuyên bố hỗ trợ heterogeneous.
4. timeDelay: đặc trưng thời gian/ngữ cảnh ở repo features — ngoài phạm vi cơ chế FL.

Quy tắc đọc: mọi hiệu so với độ ồn phiên ([R20/V20 §1](results-r20-v20.md#1-độ-ồn-giữa-các-phiên-cùng-công-thức-cùng-seed));
không cộng gain các ô; không chọn round/HP bằng official test; hai seed không phải significance.
