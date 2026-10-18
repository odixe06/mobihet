# C1 — Chỉ truyền marginal Gaussian trên 53 tọa độ còn dùng sau m4b

**Trạng thái (17/10 ~16:40Z):** không cần train. Với m4b, mẫu 66 chiều rồi đặt 0 13 tọa độ có cùng phân phối N(μ_S, Σ_SS) với
mẫu từ moment 53 chiều chèn 0 ⇒ utility giống gpool về phân phối; còn kiểm covariance + byte bằng unit test trước khi dùng
([design §Vòng 13](../../methods/full-drift/design.md)). Đề xuất owner 17/10 (§7.2, hỗ trợ R1, không tính là cải tiến accuracy).

## Cơ chế
Marginal Gaussian trên S là N(μ_S, Σ_SS); không cần tọa độ luôn bị mask. Giữ covariance đầy đủ trên S, chèn 0 về đúng vị trí x66.
Không phải rank4/8 Nyström, không đổi DAGSNet, không tái dựng primitive V6. Factor FP16, jitter, thứ tự Cholesky và RNG làm mẫu
cụ thể khác (không bit-exact), phân phối giữ nguyên trước lượng tử.

## R1
Codec moment hiện tại ([core/gaussian_moments.py](../../../core/gaussian_moments.py): sum/second moment FP32 lên, mean/factor FP16
xuống), D = 66: 145.968 B lên + 72.912 B xuống = 218.880 B/client/bootstrap. D = 53: 95.216 + 47.536 = **142.752 B** (−35 %), chưa
prior. (2,78 + 1,39 MiB trong report là tổng 20 client.)

## Cold start (đề xuất owner)
Không ghép miễn phí moment bootstrap với proxy exchange trong cùng contact. Lịch nhân quả: epoch đầu local-only, contact cuối epoch
gộp/phát moment, epoch sau mới dùng pooled synthetic và trao proxy; lịch này thay baseline preloaded-moments nên cần controls cùng lịch.
