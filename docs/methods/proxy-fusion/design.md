# ProxyFuse — triển khai kết hợp private và proxy DML

Đề xuất15/10 tối, owner duyệt kiểm nghiệm cùng ngày. Cổng dưới đây khóa **trước** khi chạy probe.

## Vấn đề và bằng chứng

Tri thức quần thể đã nằm trong proxy chung của DML nhưng private triển khai không hấp thụ:

| nguồn đã audit | proxy head F1 macro | private gp F1 macro | private head |
|---|---:|---:|---:|
| ClassTail ct_dml r20 pilot group, seed42/43 | 0,7735/0,7719 | 0,6584/0,6502 (target-mass) · 0,6215/0,6183 (true-class) | 0,5909/0,5856 |
| Full H1/H2 h_dml r20 official test (chỉ mô tả) | 0,7879 | 0,7324 | 0,6096 |
| Repo anh em NILM-FL r50 (kế thừa) | 0,823 | 0,696 | — |

Private KD chỉ trên x local nên không nhận hàm của proxy ở vùng input receiver khác; mọi kênh thống kê
vô hướng (ClassTail, class-mass, CCR) ≈ local. Chưa phương pháp nào trong 28 họ nhắm vào khâu hấp thụ/triển khai.

## Cơ chế

Mỗi xe đã giữ proxy chung (DML tải về mỗi round) và private của mình. Triển khai hậu kiểm, không train/byte thêm:

```text
b_k = log pi_g - log pi_k            # đúng prior của rule private gp đã audit
F0  = 0.5*(log_softmax(z_priv)+b_k) + 0.5*log_softmax(z_proxy)     # primary, beta=0,5 cố định
```

Proxy mang ranh giới đa receiver; private mang đặc thù local. Trung bình log-prob (geometric mean) đếm prior một lần.
Nếu lỗi hai model bổ sung nhau, F0 hơn cả hai. Nếu private không có tín hiệu bổ sung ở receiver lạ, F0 ≤ proxy.
Chi phí: một forward proxy thêm khi suy luận; private vẫn có thể khác kiến trúc (R2) vì kết hợp ở output.

## Probe khóa

- Weights: ct_dml r20 seed42/43 (ClassTail, đã audit), restore DVC MD5 PASS. Holdout pilot `group` (receiver chưa thấy,
  primary) và `old` (temporal, phụ); 65.536 dòng mỗi tập. Không đọc official test.
- Tái hiện bắt buộc: argmax private head/gp/gp_legacy và proxy head phải khớp exact predictions đã audit
  mọi client (cho phép ≤0,01% dòng do kernel fp16, báo exact số dòng lệch); vượt thì gate FAIL, không diễn giải F0.
- Primary F0 dùng `b_k` target-mass từ `log_pi_effective` đã lưu (rule gp audited của arm).
  Diagnostic, không thay primary: F0 với true-class Laplace prior; trung bình xác suất 0,5/0,5.
  Không quét β, không chọn rule theo holdout.
- Cổng PASS mỗi seed trên group, mean20client: F0 − max(proxy head, private gp, private gp_legacy) ≥ +0,010 và
  benign F1(F0) ≥ max benign ba rule đó − 0,005. Báo đủ10metrics/per-class, cả group và old.
- FAIL ⇒ ghi phạm vi: private không thêm tín hiệu ngoài proxy ở r20 pilot; không quét β/T để cứu.
  PASS ⇒ chưa là R1/R2 proof; bước tiếp do owner chốt (xác nhận full-data, rule triển khai).

## Giới hạn

Một snapshot r20, một recipe DML, hai seed; không chứng minh lợi khi proxy bị nén hoặc private khác kiến trúc.
Proxy chung tốt hơn private cũng nói rằng mốc mạnh cần báo kèm trong mọi so sánh triển khai.
