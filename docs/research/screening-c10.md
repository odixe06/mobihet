# C10 — screening nén MobiHET Core

Owner 03/11/2026: Core làm mục tiêu đề xuất, giữ performance với ít byte hơn; Kaggle 10–15 round,
seed42 duy nhất. Chọn **10 round**, cosine50, DAGSNet W1/5343 xe/1 epoch/floor16 như Core.
Báo riêng thông thường và đường tương thích SecAgg. Không đổi run F50 hoặc protocol chung.

## Bằng chứng → cơ chế → dấu hiệu bác bỏ

Core hiện bootstrap up271031/down729408 B +1088 B/round: 1011319 B ở r10, 1022199 B ở r20.
Bootstrap chiếm97,9% tổng r20. Q-only dù bỏ hết cũng tiết kiệm tối đa2,13% ở r20.

- Lossless CDF: CDF nhiều giá trị lặp; nén zlib và giải nén giữ nguyên tensor. Đo byte thật từ bootstrap toàn mạng của
  mỗi kernel. Đây là phương án ưu tiên nếu giữ performance tuyệt đối. Không cộng tỷ lệ nén fixture2xe vào headline5343xe.
- Histogram256→64: giảm số phần tử trao đổi, dựng lại score/covariance bằng64bin; đổi độ phân giải synthetic.
  Bác bỏ nếu performance/cohort giảm vượt gate. Không thay bằng cắt lát CDF cũ.
- Q downlink uint8: giữ đủ16×16 quan hệ lớp, round(p×255), decode rồi chuẩn hóa hàng; uplink vẫn fp16.
  Bác bỏ nếu teacher bị méo gây giảm chất lượng. Không suy argmax không đổi thành gradient/utility không đổi.
- Combo kiểm tương tác, không cộng gain từ hai ablation.

## Batch khóa trước push

| ô | bootstrap | Q downlink | câu hỏi |
|---|---|---|---|
| C0 core |256bin, compact hiện hành|fp16|control cùng session recipe|
| C1 boot64 |64bin, CDF uint16, mean/L fp16|fp16|nén bootstrap có giữ utility?|
| C2 q8 |như Core|uint8+chuẩn hóa|giữ utility softtarget?|
| C3 combo |như boot64|như q8|hai thay đổi cộng được?|

Q8 gửi body256B+header quy ước32B=288B, thay544B downlink. Không bỏ offdiagonal.
Không thay header/transport vật lý, chưa đo ACK/retransmission/crypto.
Bootstrap64 chưa có số đo sparse upload toàn5343xe: **tính bảo thủ uplink dày512576B**, down256320B.
Tổng bootstrap64=768896B; r10=779776B, combo777216B; so Core r10≈−22,9%/−23,1%.
Số byte là payload mô hình hóa; histogram sparse Core là mean đã đo, chưa wire transport end-to-end.

SecAgg: dùng vector cố định, histogram số nguyên và moment fixed-point cần thang/overflow riêng trước triển khai.
Không coi zlib/sparse upload từng xe là cộng được dưới SecAgg. Với accounting dày như proposal:
Core bootstrap2385280B, boot64=768896B; r10=2396160B so779776B (~−67,5%).
Downlink lossless được nén sau aggregate nên không cản cộng vector uplink. Chưa triển khai SecAgg/DP.
Q8 chỉ downlink nên không đổi phép gộp uplink hiện tại.

## Cổng đọc kết quả — r10 duy nhất

1. Đủ10round, LRcosine50, đủ5343xe; finite weights/BN/Q/loss, mọi xe có appliedstep;
   skip≤1% theo guard chung. ERROR/numerical failure phải đọc log, không tự gọi compression utilityFAIL.
2. So cùng-round Core: pooled F1 không giảm>.001; p10 không giảm>.006;
   meanF1 xe<256 không giảm>.003; timeDelayF1 không giảm>.0035; absentaccuracy không giảm>.01.
   Ngưỡng là screening thực dụng dựa spread cũ (absent .01 là ngưỡng bảo vệ đề xuất), không equivalence/significance.
   Báo cả delta âm nhỏ, đủ10metric, F1 từng lớp, low256, rare và số/độ lớn xe giảm.
3. Nhận ứng viên nghiên cứu nếu qua mọi gate và tổng byte hai chiều giảm; ưu tiên lossless khi giữ tensor.
   Không chọn HP/best round bằng test. r10/seed42 không chứng minh giữ performance r50.
4. Audit source overlay, packet Q8 thật/decoded target/teacher chain, CM→10metric, bootstrap codec byte/hash cả2rank.
   Snapshot nhỏ không phải fullprivatebank; chưa prediction→CM replay hoặc dựng lại aggregate uplink stream (không được lưu). Collector không tự tuyên bố utilityPASS.

## Local feasibility đã thực hiện

[scripts/probe_core_compression.py](../../scripts/probe_core_compression.py): 40snapshot X0/X1 (2seed đã có,
không thêm train43), Q8 max|delta|.0144024;0argmax đổi. zlib Q body trung bình522,6B >fp16body512B.
Không ưu tiên lossless Q packet nhỏ. Probe5arm×3round/seed42/2xe/173trainrow:
finite,skip0; lossless state bitexact control. Artifact `runs/core-compression/{q-codec,smoke}.json`.
Compiled local sm86 không chứng minh T4/fullnetwork/utility. Bootstrap fixture CDF630784→14894B,
**không ngoại suy tỷ lệ này sang toàn mạng**.

## Chạy và thu

[scripts/run_c10.py](../../scripts/run_c10.py): quota/slot scoped → notebook private/source overlay → push receipt →
poll đúng version1200s → pull (kể cảERROR) → [audit](../../scripts/audit_c10.py).
Cap3,3h/kernel (bao gồm setup/compile/eval/save), dự trù4×3,3h+15% quota; không tự giảm round khi thiếu.
Kết quả/audit tại `runs/c10-study/`, payload/model tại `artifacts/c10/`; manifest launch sinh trước push.
Nguồn mô tả lossless thông dụng: [RFC8878](https://www.rfc-editor.org/rfc/rfc8878.html).
Probe hiện dùng zlib có sẵn, không cài thêm package; tỷ lệ phụ thuộc dữ liệu/codec.
