# P6 — Consensus dự đoán trên synthetic input chung, không proxy tham số

**Trạng thái (17/10 ~17:10Z):** chưa chạy — chỉ mở nếu ensemble private có teacher quality vượt hard-label/class-mean/own trên real
holdout. Đề xuất owner 17/10 (ưu tiên 6, thấp). Tiền lệ âm: [P1 Gaussian-logit](../../archive/methods/gaussian-logit-distillation/README.md)
(teacher local yếu), [V9 anchor residual](../../lessons/anchor-function-residual.md) (target gần class-mean dù coverage > 99 %),
Build7 KD peer không thêm gì.

## Cơ chế
Mỗi private dự đoán trên cùng bank synthetic (seed, version, nhãn sinh mẫu; một phần giữ cố định đo drift, một phần đổi theo lịch
khóa trước). Epoch r dùng consensus r−1; cuối epoch upload prediction cho bank; RSU chỉ mở tổng probability/count theo cohort.
Chuẩn hóa scale giữa kiến trúc (không blind-average logits). Giữ hard CE synthetic làm neo; không anchor từ record local; không
kernel nội suy broad-band kiểu V9. R2: chỉ cần input66/output16.

## R1
256 anchor × 16 lớp × FP32, up + down = 32.768 B/round trước counts/framing/SA và bootstrap phân phối. Không truyền x nếu hai bên
tái tạo cùng bank từ seed + version (nếu không deterministic xuyên môi trường thì phải trả byte bank). int8 lý tưởng 8.192 B
chưa kiểm utility; SA thường cần accumulator rộng hơn.

## Không lách cold start
Warm-start từ DML mạnh phải báo inherited model/bytes/epochs và có cold-start control. Lợi FL phải vượt own consensus và
class-mean cùng input/compute; entropy/target fit/coverage chỉ chẩn đoán.

## Nguồn
FedMD ([arXiv:1910.03581](https://arxiv.org/abs/1910.03581)) dùng public data; thay bằng synthetic là adaptation lớn.
