# Protocol so sánh — khớp `fl_dagsnet_veremi`

Cập nhật 2026-10-10. **Chủ dự án chốt §1 + §3 (Q2) và mức ngân sách B0 + B2 (Q1).** Còn mở: số contact/round (Q3).
Mục tiêu: số của MobiHet mới phải đặt được cạnh số của các bản dựng lại trong repo `odixe06/fl_dagsnet_veremi` (AFPHA, FD-IDS, NILM-FL,
PerFed-SKD, TinyProto, pFedES, …) mà không cần chú thích lệch protocol.

## 1. Phần chung với `fl_dagsnet_veremi` (đọc từ report/CONTEXT của repo đó, 2026-10-10)

| mục | giá trị | ghi chú |
|---|---|---|
| Dữ liệu | train 43.045.415 dòng (đã z-score), test 10.761.343 dòng (raw, áp scaler một lần) | [knowledge/dataset.md](../../knowledge/dataset.md) |
| Input | **66 feature gốc**, đúng thứ tự `knowledge/meta.json`, fp16 | không M4b, không P4/map (chốt 2026-10-10) |
| Partition | Dirichlet α=0,5; 20 / 50 / 100 client | không chia lại |
| Model | DAGSNet 395.024 tham số, 16 lớp, khởi tạo mặc định PyTorch, seed 42 | [knowledge/architecture.md](../../knowledge/architecture.md) |
| Lịch | **50 round × 1 epoch local** | |
| Batch | 512 / 512 / 256 (20c / 50c / 100c) | |
| Optimizer | Adam(W) wd 0, reset theo client mỗi round (AFPHA); LR **cosine theo round** 1e-3 → 1e-5, T=50, hằng trong round | lịch thống nhất 2026-09-13; **kiểm lại code hiện hành của repo anh em trước khi viết driver** |
| Ổn định | clip grad-norm 1,0; AMP fp16, loss fp32 | |
| Đánh giá | **mọi round, toàn bộ test**, 10 metric từ một confusion 16×16; headline = **round 50**, không chọn checkpoint | metric riêng client (mean) và metric model toàn cục nếu có |
| Seed | 42 (repo anh em chạy 1 seed) | MobiHet báo thêm seed 43 cho so sánh nội bộ |

## 2. Phần MobiHet thêm vào (khác repo anh em, phải báo kèm mọi số)

1. **Kênh truyền có deadline (R1).** Báo ở hai mức ngân sách B0 (≈4,5 kB/contact) và B2 (≈231 kB/contact) ([đề xuất §3](de-xuat-huong-moi.md#3-ngân-sách-contact-và-kích-thước-tri-thức)).
   - Mỗi round, mỗi xe có contact mô phỏng: ngân sách byte và `tau_true` lấy từ simulator archive ([kế thừa §6](../legacy/kien-thuc-ke-thua.md#6-kênh-truyền-và-mô-phỏng-contact)).
   - Xe không có contact hoặc chưa nhận đủ gói thì vẫn học local, dùng tri thức cũ.
   - Repo anh em giả định full participation, không giới hạn byte. Muốn tách được hiệu ứng kênh thì mỗi phương pháp phải chạy cả hai chế độ:
     **không ràng buộc** (khớp anh em) và **có ràng buộc**.
2. **Model khác kiến trúc (R2).** Chạy đồng nhất DAGSNet trước để so với repo anh em. Sau đó chạy kịch bản khác kiến trúc (H1/H2/H3 như `MobiHet_IDS.md` §4.3 cũ),
   báo theo từng kiến trúc.
3. **Riêng tư (R3, chốt Q4).** Assert trong preflight: không tensor dữ liệu `(x, y)` nào nằm trong payload. Ghi rõ loại tri thức đi trên kênh và số byte của từng loại.
   Thống kê/tham số đi qua secure aggregation mô phỏng: RSU chỉ dùng tổng nhiều client. DP là thí nghiệm độ nhạy sau.
   **Ngoại lệ được chốt 2026-10-11:** RSU được đọc từng codebook synthetic tối ưu từ nhiễu; cấm khởi tạo bằng record thật.
   "Dữ liệu" bị cấm ở đây là mẫu local thật; không giả định codebook riêng đã qua secure aggregation/DP.
4. **Mốc dưới local-only** theo đúng §1. Repo anh em chưa có mốc này.

## 3. Chọn phương pháp/HP mà không tune trên test (chốt 2026-10-10)

Repo anh em không chọn checkpoint, nhưng có sweep HP trên validation 2% (TinyProto chọn k). Đề xuất cho MobiHet:

- **Pilot/chọn cấu hình:** holdout nội bộ (`old` temporal 2% + `group` receiver chưa thấy, fixture B như archive). Cổng khóa trước khi chạy, hai seed.
- **Kết quả để so với anh em:** train lại theo đúng §1 trên toàn train80%, chấm test mọi round, headline round 50. Đã chấm thì không quay lại chỉnh.
- Báo đủ 10 metric, F1 theo lớp, benign F1, byte up/down theo loại tri thức, số contact hoàn tất, thời gian.
