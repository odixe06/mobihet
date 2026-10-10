# CONTEXT — MobiHet (hướng mới, không chia sẻ dữ liệu)

Đọc file này → **[LESSONS.md](LESSONS.md)** → [kiến thức kế thừa](docs/legacy/kien-thuc-ke-thua.md) → [đề xuất](docs/research/de-xuat-huong-moi.md).

IDS V2X cộng tác qua RSU. Bốn yêu cầu bắt buộc:
- **R1:** contact hữu hạn, mọi byte trả phí, có deadline thật.
- **R2:** mỗi xe model riêng, có thể khác kiến trúc.
- **R3:** không mẫu dữ liệu thô / record `(x, y)` nào rời xe.
- **R4:** 66 feature gốc, cùng protocol `odixe06/fl_dagsnet_veremi` để so sánh.

## Trạng thái — 2026-10-10

- **Hướng cũ (record cache, Build7–21) đã dừng** vì vi phạm R3.
  - Archive 3 = `odixe06/mobihet-archive-b9-b21` (đổi tên từ `odixe06/mobihet` 2026-10-10), local `../mobihet_archive`.
  - Kernel Kaggle cuối (Build20/21) đang được thu về archive; không push thêm kernel nào của hướng cũ.
  - Không xóa dữ liệu local archive khi chủ dự án chưa yêu cầu.
- Repo này mới khởi tạo. Hiện có: knowledge (chép từ archive, không phụ thuộc phương pháp), tổng hợp kế thừa, đề xuất H1–H6, nháp protocol.
  Chưa có driver FL.
- **Probe H2-T0-lite (local, MLP, 2026-10-10) PASS:** mẫu từ Gaussian hiệp phương sai đầy đủ theo lớp, gộp từ tổng cộng dồn của 19 client khác,
  +0,102/+0,106 macro-F1 so với replay cân bằng local (8/8). Điểm trung bình lớp −0,013/−0,022.
  Chỉ là chẩn đoán: chưa DAGSNet, chưa ngân sách/DP. [Kết quả](docs/research/probes/h2-input-stats.md).

## Quyết định hiện hành (chủ dự án, 2026-10-10)

- **Tiêu chuẩn FL:** không chia sẻ local data giữa các client dưới bất kỳ hình thức nào, kể cả trong đối chứng thí nghiệm.
- Dạng tri thức được phép thử: tham số khối model dùng chung, prototype/thống kê theo lớp, logit trên dữ liệu proxy, generator/dữ liệu tổng hợp.
  Thử tất cả được, nhưng phải tính đến kinh nghiệm cũ (kênh prototype trung bình lớp từng chỉ được ±0,006).
- Input **66 feature gốc**, không P4/map. Cấu hình khớp `fl_dagsnet_veremi` (DAGSNet, α=0,5, 20/50/100 client, 50 round × 1 epoch, cosine LR, test mọi round).
- DVC remote dùng chung với archive: `gdrive` = `/mnt/g/My Drive/NCKH/veremi_build/mobihet/dvc-storage`.
- Có thể phân tích dữ liệu client trong `../dataset/` (train z-score: `fl_client/alpha05/{20,50,100}_client/train/client_id=NNN/`; test raw: `centralized/test/`).

## Chờ chủ dự án chốt

Q1 mức ngân sách contact, Q2 protocol, Q3 số contact/round, Q4 mức riêng tư (DP/secure aggregation), Q5 nguồn proxy (nếu theo H5),
Q6 thứ tự S0→S1, Q7 tắt M4b. Chi tiết: [đề xuất §6](docs/research/de-xuat-huong-moi.md#6-câu-hỏi-cần-chủ-dự-án-chốt).

## Bước kế tiếp (sau khi chốt)

1. S0: driver theo protocol, dựng từ code `fl_dagsnet_veremi` + simulator archive; chạy mốc local-only 20c.
2. S1a (H2 offline: dữ liệu tổng hợp từ thống kê lớp) song song S1b (H1: sức chứa proxy).
3. Cổng khóa trước khi chạy, hai seed, mỗi arm đổi một yếu tố.

## Muốn biết → mở

| câu hỏi | tài liệu |
|---|---|
| Archive đã thử gì, kết quả, bài học | [kiến thức kế thừa](docs/legacy/kien-thuc-ke-thua.md), `archive3:docs/experiments/index.md` |
| Đề xuất, ngân sách byte | [de-xuat-huong-moi.md](docs/research/de-xuat-huong-moi.md) |
| Protocol so sánh | [protocol.md](docs/research/protocol.md) |
| Dữ liệu / DAGSNet / máy local / Kaggle | [knowledge/](knowledge/README.md) |
| Backup, kéo dữ liệu archive | [backup.md](docs/operations/backup.md) |
