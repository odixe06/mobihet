# P5 — Federated teacher cộng các hàm theo input, cập nhật bằng histogram residual

**Trạng thái (17/10 ~17:10Z):** chưa chạy; P4 teacher-only FAIL (ridge M256 0,378) cho thấy teacher thống kê trên input thô kém xa DAGSNet ⇒ không mở nếu owner không yêu cầu. Đề xuất owner 17/10 (ưu tiên 5). Tiền lệ âm:
[BoundaryCells](boundary-cell-counts.md) (bảng đếm tĩnh, bốn adaptation FAIL); đề xuất v2 (10/10) từng loại tree/histogram
classifier theo trả lời owner khi đó.

## Cơ chế
Không proxy neural, không Gaussian. RSU học teacher additive F_c(x) = Σ_j a_jc φ_j(x) với dictionary bins/interaction công bố
trước. Mỗi round chỉ cập nhật block J basis functions khóa từ snapshot trước. Client gửi
G_kjc = Σ_i φ_j(x_i)(p_c(x_i) − 1[y_i = c]), H_kjc = Σ_i φ_j(x_i)² p_c(1 − p_c); RSU cập nhật a_jc ← a_jc − η G_jc/(H_jc + λ) với
shrinkage (xấp xỉ coordinate/diagonal Newton, bỏ cross-class/cross-basis Hessian; damping khóa qua fixture). Một epoch → một
packet → một teacher version; không hỏi xe nhiều lần trong một contact. Dictionary cần interaction; histogram 1D có thể quá yếu.
Giữ mask m4b; không receiver ID/metadata attacker.

## R1
Upload FP32 G/H = 2·J·16·4, download J·16·4; J = 256 ⇒ 49.152 B/round trước metadata/SA; dictionary + lịch block trả bootstrap.
Full 66 feature × 32 bins: upload 270.336 B.

## Controls / dừng
Static counts cùng dictionary; additive 1D vs interaction; pooled vs own G/H; teacher vs local DAGSNet; private KD vs no-peer.
Dừng nếu teacher chỉ là class prior/bảng yếu, dynamic không hơn static, hoặc private không giữ lợi.

## Nguồn
FedTree, MLSys 2023 — gộp histogram cho boosted trees; ở đây không áp nguyên bản (split nhiều tầng cần trao đổi phụ thuộc nhau).
