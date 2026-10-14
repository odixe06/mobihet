# Tên thí nghiệm và quy tắc lưu trữ

V2–V10 là **đợt kiểm nghiệm**, không phải chuỗi phiên bản của một thuật toán. Một đợt có thể so nhiều cơ chế; đợt sau có thể dùng lại driver/codec của đợt trước. Không suy V4 thay thế V3 hoặc V3 thay thế V2.

| nhãn lịch sử | tên build hiện tại | nội dung |
|---|---|---|
| V2 | [countsketch-subspace-ratio](../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/README.md) | CountSketch, seeded subspace, density ratio |
| V3 | [proxy-carriers](../../archive/methods/proxy-compression/builds/proxy-carriers/README.md) | Coordinate EF, rolling subnetwork, orthogonal projection, PowerSGD |
| V4 | [support-mala](../../archive/methods/gaussian-sampling/builds/support-mala/README.md) | Support-conditioned Gaussian và MALA |
| V5 | [anchor-synthetic-books](../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/README.md) | Kernel anchors, gradient/latent synthetic books |
| V6 | [geometry-g59-g61](../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/README.md) | Geometry primitive G59/G61 |
| V7 | [nystrom-rank4-rank8](../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/README.md) | Covariance Nyström rank4/8 |
| V8 | [hybrid-pilot](../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/README.md) | Gaussian + DML hybrid pilot |
| V9 | [anchor-residual](../../archive/methods/anchor-function-residual/builds/anchor-residual/README.md) | Anchor-function residual teacher |
| V10 | [gaussian-base-energy](../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/README.md) | Gaussian-base joint energy/MH |

Khu làm việc chỉ giữ build còn train/pending hoặc thiết kế đã chốt. Build kết thúc chuyển `archive/methods/`; thiết kế/kết quả lịch sử chuyển `docs/archive/`; bài học dùng tiếp ở `LESSONS.md` và `docs/lessons/`.
H1/H2 full, Hybrid full và BoundaryCells giữ nguyên kernel ID, notebook tag và source/config đã push. Đổi tên thư mục không đổi thuật toán, cổng hoặc kế hoạch round.
Archive có thể còn là dependency của snapshot cũ. Những file đó không được xóa hoặc thay byte; facade tương thích giữ đủ source để audit/resume. Đóng build không có nghĩa phủ định toàn bộ họ phương pháp.

[Mục lục](../README.md) · [Danh mục đề xuất](../proposals.md) · [Archive](../archive/README.md).
