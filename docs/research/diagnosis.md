# Chẩn đoán — nguyên nhân đã biết và phép thử tiếp (03/11/2026, sau X20)

Mục tiêu: own-test từng xe, nâng xe ít mẫu/non-IID, băng thông thấp, schema dùng được cho mô hình khác nhau.
Hướng phương pháp: [proposal](proposal.md). Số đo: [X20](results-x20.md), [W20](results-w20.md), [R20/V20](results-r20-v20.md), [Q20](results-q20.md), [P20](results-p20.md).
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
| KD dòng thật đè lớp vắng | NCKD khớp phân bố lớp sai theo Q[y]; lớp vắng có khối lượng teacher nhỏ → logit bị ép trên input local | W20; **X20 xác nhận**: giới hạn lớp xe có → absent +.037–.039 so W2/W3, xe ≥4096 +.12–.15 |
| KD dòng thật luôn tốn timeDelay | NCKD trên hàng timeDelay dạy phân bố lớp sai ≈ benign (Q[13,benign] .24); benign có ở mọi xe nên không giới hạn được | X20 nckd_local: timeDelay −.009/−.011, td→benign +.02 |
| Input-bank chủ yếu tăng tốc | target theo input cho xe nhỏ/vừa; lợi so core r10 +.011–.020 → r20 +.004–.007 low256 | X20 bank |
| Đổi loss synthetic chỉ dời ngưỡng timeDelay/benign | hard+NCKD: recall +.06, precision −.09, benign→timeDelay ×1,85, Q[13,13] giảm | W20 synth_nckd |
| Chênh từng xe phần lớn là ồn | cùng công thức/seed, ~250 xe dao động >.05 mỗi chiều giữa hai phiên | repeat R0–R3 |
| 97% byte là bootstrap | histogram dày fp32 + moment Gaussian không dùng; nén còn 1,02 MB, utility giữ | W20 boot_compact |

## Kết luận X20 ([kết quả](results-x20.md))

| giả thuyết | kết quả |
|---|---|
| floor + nén cộng được | PASS — core trùng W6/W7 |
| input-bank còn lợi khi có floor | PASS sát ngưỡng (seed42 low256 +.0037), lợi thu hẹp theo round |
| hại lớp vắng do NCKD ép logit lớp vắng | xác nhận |
| realKD không-hại cộng vào combo | không — full trượt low256 seed43 (−.0032), timeDelay −.008/−.009, rare −.007 |

## Tiếp theo

1. 50 round theo [protocol](protocol.md), 2 seed: core+bank (phương pháp) và core (bank còn lợi ở 50 round?) cùng baseline.
   Cần owner duyệt quota (≈10 h/kernel). Quyết định giữ bank: low256/p10 r50 core+bank so core vượt spread ở cả hai seed.
2. Nếu xe yếu vẫn xa APFed ở 50 round: bộ nhớ truy vấn/bank tích lũy cùng ngân sách byte, reliability từ validation local.
3. Ô trộn kiến trúc (≥2 kiến trúc client) trước khi tuyên bố hỗ trợ heterogeneous.
4. Bảo mật: tuyên bố "tương thích secure aggregation, chưa DP"; byte SecAgg ước tính ở proposal.
5. timeDelay: đặc trưng thời gian/ngữ cảnh ở repo features — ngoài phạm vi cơ chế FL; X20 thêm bằng chứng mọi KD dòng thật
   dạy lại sự nhầm benign/timeDelay của teacher.

Quy tắc đọc: mọi hiệu so với độ ồn phiên ([R20/V20 §1](results-r20-v20.md#1-độ-ồn-giữa-các-phiên-cùng-công-thức-cùng-seed));
không cộng gain các ô; không chọn round/HP bằng official test; hai seed không phải significance.
