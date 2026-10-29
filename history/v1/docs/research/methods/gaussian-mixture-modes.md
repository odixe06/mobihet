# P2 — Nhiều mode trong cùng lớp (Gaussian hỗn hợp), giữ full covariance và vị trí gpool

**Trạng thái (17/10 ~17:10Z): không mở P2a** — hỗn hợp urban/highway không làm proxy nhận mẫu nhiều hơn (benign +0,007, dataReplay −0,001, positionMirroring +0,008, timeDelay 0; cần ≥ +0,05 ở 2/4 lớp) ([kết quả probe](../../methods/full-drift/results.md)). Trước đó: chỉ chẩn đoán trong kernel probe vòng 13 (moment urban/highway cho 4 lớp khó; proxy nhận mẫu
hỗn hợp vs 1 Gaussian). Mở arm P2a nếu hỗn hợp nhận nhiều hơn ≥ +0,05 tuyệt đối ở ≥ 2/4 lớp
([design §Vòng 13](../../methods/full-drift/design.md)). Đề xuất owner 17/10 (ưu tiên 2) = nhánh B2 của đề xuất B.

## Vấn đề
`gpool > gown/replay` xác nhận input statistics quần thể có ích, nhưng một mean/covariance mỗi lớp không biểu diễn phân phối
nhiều cụm (atom do zero-imputation, 4 scenario, trạng thái chuyển tiếp). Chưa có phép đo chứng minh multimodality là nguyên
nhân chính.

## Cơ chế
Chỉ thêm component cho tối đa 4 lớp khó {benign, dataReplay, positionMirroring, timeDelayAttack}; lớp khác giữ Gaussian cũ.
Mỗi component full covariance trên 53 tọa độ không bị m4b che; chèn 0 về x66. Không đổi feature/model private.
- **P2a — conditional moments cố định:** một phép chia mode khóa trước từ train (ưu tiên urban/highway; scenario chỉ làm strata,
  không là feature thứ 67, không dùng receiver ID).
- **P2b — federated EM:** chỉ mở nếu P2a không đủ nhưng chẩn đoán cho thấy cấu trúc rõ. Khởi tạo chung từ pooled stats, mọi client
  tính responsibility theo cùng snapshot, gộp sufficient statistics một lần cuối epoch. Giả thuyết khác, không tự nối P2a FAIL.

Client gửi tổng N_kcs = Σ r_ics, A_kcs = Σ r_ics x_i, B_kcs = Σ r_ics x_i x_iᵀ; RSU μ = A/N, Σ = B/N − μμᵀ sau gộp (không trung
bình covariance local). Component ID/version chung; không average component cùng số thứ tự của GMM fit độc lập. Giữ đúng 256
synthetic/batch, mỗi lớp cùng tổng mẫu như gpool, chia giữa mode theo mass quần thể; shrinkage về Gaussian lớp khi thiếu support;
component phải đủ client đóng góp (ngưỡng cohort không tạo DP).

## Controls
pooled mixture; own mixture; gpool một Gaussian; randomized-mode (tráo assignment dòng train trong lớp rồi tính lại moment, giữ
số dòng mỗi mode); matched forward/loss/prior. Likelihood, số mode sống, fidelity chỉ chẩn đoán.

## R1
Thêm 4 component trên d53 ≈ 4×(1+53+53·54/2)×4 = 23.760 B mỗi chiều FP32 (chưa overhead). P2a tĩnh cache được; P2b cập nhật mỗi
epoch phải trả byte mỗi epoch ⇒ dễ làm DML vượt B2. Không ghép EM với q2 rồi giấu phần vượt budget.

## Dấu hiệu sai
Chỉ likelihood/fidelity tốt hơn; tăng F1 < +0,010; lợi chỉ ở lớp dễ; pooled không hơn own và một Gaussian cùng compute.

## Tiền lệ / rào chắn
[Support V4](../../archive/methods/gaussian-support-constraints/README.md), [Geometry V6](../../lessons/geometry-primitive-moments.md):
fidelity/schema tốt không đủ utility ⇒ P2 đo mode có ích với classifier, không tái dựng primitive. K4 joint-energy dùng mixture
analytic làm base (FAIL vì lý do khác). Đề xuất v2 (10/10) từng loại "mixture học bằng EM" theo trả lời owner khi đó; đề xuất 17/10
của owner mở lại P2b có điều kiện.

## Nguồn
Federated mixture/EM: Wu et al. 2023 ([arXiv:2305.01068](https://arxiv.org/abs/2305.01068)); private tùy kiến trúc là adaptation.
