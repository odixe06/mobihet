# Pilot P2 — kết quả thu 2026-10-11

Đã pull đủ **4 kernel /22 run ×20round**. `scripts/verify_p2.py` qua kiểm tra những bằng chứng còn giữ:
config, lịch, source so với dataset private tải hiện tại, fixture manifest, log2×T4,
summary đủarm, confusion tổng, support, tính nhất quán các metric đã lưu.
Chi tiết: `papers/p2/audit.json`; bảng tái tạo bằng `python scripts/report_p2.py`, `papers/p2/report.md`.

| arm | group gp s42/s43 | giữ lợi P1 | cổng số học |
|---|---|---|---|
| H2 diag B0 h6 | 0,5419 /0,5410 | 99% /96% | P2-a PASS |
| H2 full B0 h6 | 0,4643 /0,4608 | ≈0% | FAIL;0/20xe kích hoạt |
| H2 lr4 B0 h6 | 0,4642 /0,4607 | ≈0% | FAIL;0/20xe kích hoạt |
| H2 full B2 h6 | 0,5718 /0,5758 | 102% /103% | P2-a PASS |
| DML DAGSNet B0 h6 | 0,5069 /0,5025 | 27% /24% | FAIL;0/20xe nhận proxy cập nhật |
| DML DAGSNet B2 naive | 0,5696 /0,5667 | 67% /64% | FAIL |
| DML DAGSNet B2 h6 | 0,5900 /0,5879 | 80% /77% | P2-a PASS |
| DML DAGSNet B3 naive int8 | 0,6085 /0,6058 | 92% /88% | P2-a PASS |

DML B2 h6−naive **+0,0204/+0,0211**, qua cổng P2-b số học.
H2 diag B0 h6≈naive; cursor tiết kiệm byte (≈1,0 vs2,2kB/client/round), không tăng F1 rõ.
Đây là holdout nội bộ và hai seed; chưa phải protocol test50round hay kiểm định ý nghĩa thống kê.

## Nguyên nhân và giới hạn

- H2 lr4 chỉ nén chiều xuống; upload vẫn145.792B/client như full. B0 chưa hoàn tất đủ tri thức trong20round.
  Giảm download mà giữ upload lớn không giải quyết kích hoạt muộn.
- DML B0 không nhận proxy mới vẫn hơn local: hai mạng học tương hỗ cục bộ có lợi riêng;
  **không** được diễn giải phần tăng này là tri thức liên client.
- P2 không giữ weights, prediction hay confusion từng client. Không thể tái dựng mean macro-F1 độc lập,
  hoặc chạy lại inference từ artifact. Verifier chỉ xác nhận nhất quán phần bằng chứng hiện có.
- Worker không xuất hash source đã thực thi. Code tải lại từ input dataset khớp checkout;
  không nâng bằng chứng này thành hash source tại thời điểm chạy.
- Prior toàn cục trong evaluator có sẵn trước contact và không nằm trong `bytes_used` P2.
- H2 tính byte fp32/fp16 nhưng không lượng tử hóa giá trị thực qua payload; DML thiếu byte scale/int-buffer,
  và có thể aggregate chỉ1update, không mô phỏng cohort secure aggregation tối thiểu.

Vì vậy **PASS ở bảng là cổng trên metric đã lưu, chưa phải xác nhận đầy đủ R1/codec/secure aggregation**.
Không dùng P2 làm đối chứng trực tiếp cho driver sửa prior/codec. Không sửa notebook đã PUSHED.
Pilot mới phải lưu đủ bằng chứng, lượng tử hóa thật, prior đi qua kênh, control cùng driver.

## Kernel đã thu

- `khanhmay0304/mobihet-p2-h2-s42`
- `minhtriethihi/mobihet-p2-h2-s43`
- `catbaochau/mobihet-p2-dml-s42`
- `minhtrit06/mobihet-p2-dml-s43`

Artifact mới chưa DVC add/push; phiên này chưa có yêu cầu backup.
