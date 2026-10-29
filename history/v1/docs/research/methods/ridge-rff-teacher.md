# P4 — Federated supervised random-feature ridge teacher

**Trạng thái (17/10 ~17:10Z): FAIL teacher gate** — pooled M256 group 0,378 (M1024 0,484, linear 0,264) ≪ 0,7174 ([kết quả probe](../../methods/full-drift/results.md)); không train private. Owner chọn **chỉ teacher-only** (kernel probe vòng 13, không train private). Mở train private
chỉ khi pooled M256 group F1 ≥ 0,7174 **và** pooled − own ≥ +0,010 **và** pooled M256 − linear ridge ≥ +0,010
([design §Vòng 13](../../methods/full-drift/design.md)). Đề xuất owner 17/10 (ưu tiên 4, hướng FL thay thế).
**Tiền lệ âm:** Build13 S13 — teacher ridge/RFF tại RSU, KD trên dữ liệu local: input − off +0,018/+0,019 nhưng input −
class-mean chỉ +0,0003/+0,0038 (lợi ngang làm mượt theo lớp), thua record replay −0,09/−0,08
([kế thừa §4](../../archive/legacy/kien-thuc-ke-thua.md)).

## Cơ chế
1. Khóa map φ(x66) qua m4b: `sqrt(2/M) cos(Ω x53 + b)` + intercept, Ω/b từ seed chung; không dùng hidden DAGSNet làm tọa độ chung.
2. Client tính G_k = Σ φφᵀ, H_k = Σ φ onehot(y)ᵀ trên training rows; class weights (nếu có) chung cả cohort.
3. RSU gộp G, H; giải (G + λI)W = H bằng solver ổn định; λ theo quy ước khóa.
4. Phát W; teacher t(x) = φ(x)ᵀW chạy local; private train CE_real + λ_KD KL(q_t ‖ p_k), teacher frozen.
5. Teacher tốt mà private thiếu support: ablation riêng thêm synthetic.

Là nghiệm ridge trên union dữ liệu cho map/objective cố định (FP32 wire làm mất bit-exact). Teacher cho score, không xác suất:
khóa cách chuyển score→prob (temperature fit trên calibration riêng), báo NLL/Brier. One-shot teacher + local distillation
(G/H của dữ liệu tĩnh chỉ gửi một lần); muốn cập nhật mỗi round thì biến thể streaming với delta dữ liệu mới.

Probe vòng 13 (khóa): σ = trung vị khoảng cách cặp trên 4.096 dòng train gộp; λ ∈ {1e-4, 1e-3, 1e-2}·tr(G)/d chọn trên old;
chấm group (argmax). Pooled M128/M256 (M1024 mô tả), own (cùng hệ số λ), linear ridge.

## R1 (FP32)
B_up = 4[d(d+1)/2 + 16d], B_down = 4·16d (d = M + 1 nếu có intercept). M128 ≈ 49 KB, M256 ≈ 164 KB tổng/client/bootstrap, chưa
counts/schema/header/ACK/SA. M128 chỉ là control capacity.

## Gates
Teacher-only (pooled vs own, linear, class-mean); private (pooled teacher vs own/no-peer, so gpool cùng epoch); codec (Gram PSD,
λ ổn định, solver dùng statistic đã decode). Teacher > private mà student không tăng thì FAIL mục tiêu.

## Rủi ro
Raw-input RFF kém feature pretrained; bandwidth sai thành gần hằng/quá khớp; least squares không tối ưu macro-F1; private vẫn
thiếu support; M256 thiếu capacity giống seeded subspace. Khác KME (chỉ mean để sinh mẫu), V9 (kernel-average collapse về
class-mean), BoundaryCells (bảng đếm tĩnh).

## Nguồn
Fed3R, ICML 2024 ([PMLR v235](https://proceedings.mlr.press/v235/fani-24a.html)) — dùng extractor pretrained, ở đây RFF trên
input gốc nên không gọi là tái lập. Rahimi–Recht, NIPS 2007 (random features).
