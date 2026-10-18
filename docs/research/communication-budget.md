# Ngân sách giao tiếp R1 — con số thật, không giảm model để vừa budget

Tách từ đề xuất owner 17/10 (§1, §7). Owner 17/10 ~13–14Z: ưu tiên communication budget hạn chế; bỏ hướng giảm độ phức tạp model.

## Yêu cầu
- **R1:** contact RSU ngắn, budget hạn chế; có thể nhỉnh hơn mốc cũ nhưng không quá lớn. Mốc đối chiếu B0 ≈ 4.500 B, B2 ≈ 231.000 B;
  **chưa có cap mới được chốt**. Dự toán mặc định = tổng up + down/client/contact.
- **R2:** nhiều kiến trúc private cùng hoạt động; kịch bản kiểm trước mắt mọi client DAGSNet (chưa chứng minh R2 thực nghiệm).
- Cần owner chốt: cap tổng hai chiều hay từng chiều, deadline, application throughput/contact trace thực tế.

## Khoảng trống của baseline
[QPacket q4](methods/qproxy-quantization.md): 208.268 B/chiều ⇒ 416.536 B up + down/client/round, vượt B2 1,80× trước moment, prior,
ACK, retry, bảo mật. Moment gpool là bootstrap một lần: D = 66 → 145.968 B lên + 72.912 B xuống = 218.880 B/client (chưa prior);
2,78 + 1,39 MiB trong report là tổng 20 client. ⇒ P1–P3 là ứng viên chất lượng có đường kiểm R1, chưa phải phương án đã thỏa R1.
Không sửa bằng tăng số lần gộp/epoch, tự nâng cap, hay báo riêng download.

## So sánh payload (application payload/client, chưa đo radio)

| Phương án | Hai chiều | Tần suất | Đọc theo B2 |
|---|---:|---|---|
| DML proxy q4 | 416.536 B | Mỗi epoch | Vượt 1,80× trước overhead |
| DML proxy q2 ([C2](methods/qproxy-quantization.md)), dự toán | ≈ 219.024 B | Mỗi epoch | Rất sát; vòng 13 đo `len(packet)` + utility |
| Gaussian66 codec hiện tại | 218.880 B | Bootstrap | Gần B2; chưa prior/contact overhead |
| [Gaussian53 C1](methods/gaussian-moment53.md), dự toán | 142.752 B | Bootstrap | Có dư địa |
| [P4 ridge](methods/ridge-rff-teacher.md) M256 | 164.352 B | Bootstrap/map cố định | Có dư địa, chưa overhead/teacher gate |
| P4 ridge M128 | 49.408 B | Bootstrap | Capacity chưa biết |
| [P5](methods/residual-histogram-teacher.md) block J256 | 49.152 B | Mỗi epoch | Chưa dictionary/bootstrap |
| [P6](methods/synthetic-consensus.md) M256 FP32 | 32.768 B | Mỗi epoch | Chưa distribution bootstrap |

Không phương án nào tự đạt B0 một-contact cold start. Báo thêm tổng tải RSU khi mô phỏng 20/50/100 xe; không cộng nhầm broadcast
vật lý một lần với byte mỗi xe cần nhận.

## Thời gian tối thiểu t = 8B / R_application (throughput giả định, không phải PHY)

| Byte/contact | 1 Mbit/s | 5 Mbit/s | 10 Mbit/s |
|---:|---:|---:|---:|
| 4.500 | 0,036 s | 0,0072 s | 0,0036 s |
| 231.000 | 1,848 s | 0,3696 s | 0,1848 s |
| 416.536 | 3,332 s | 0,6665 s | 0,3332 s |

Cộng discovery/authentication, contention, RTT, cohort wait, secure aggregation, encode/decode/commit, retry. Server solve (P4),
factorization (P2), update (P5) phụ thuộc upload vừa nhận ⇒ tính latency trong cùng contact; không kịp thì không commit snapshot
mới, không tự giả định contact miễn phí hay lịch trễ một epoch mà vẫn báo cùng contract.

## Cổng R1 bắt buộc (khi mô phỏng contact)
Cùng trace/contact cap cho candidate và baseline; attempted/ACKed/wasted bytes, bootstrap, p50/p95 latency, tỷ lệ snapshot commit
trước deadline, snapshot age, client không nhận được tri thức. Snapshot thiếu/sai base không kích hoạt; model private vẫn suy luận
local. Secure sum mô phỏng không chứng minh q4/q2 packed transport cộng được dưới crypto. Trong một contact có thể chia frame,
snapshot chỉ active sau đủ frame/CRC/version và trước deadline; không dùng fragment qua nhiều epoch rồi vẫn nói mỗi epoch nhận
proxy gộp mới.
