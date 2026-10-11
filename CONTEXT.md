# CONTEXT — MobiHet (hướng mới, không chia sẻ dữ liệu)

Đọc file này → **[LESSONS.md](LESSONS.md)** → [kết quả pilot P1](docs/research/probes/pilot-p1.md) → [kế hoạch pilot](docs/research/pilot-plan.md)
→ (khi cần nền) [kiến thức kế thừa](docs/legacy/kien-thuc-ke-thua.md), [đề xuất H1–H6](docs/research/de-xuat-huong-moi.md).

IDS V2X cộng tác qua RSU. Bốn yêu cầu bắt buộc:
- **R1:** contact hữu hạn, mọi byte trả phí, có deadline thật.
- **R2:** mỗi xe model riêng, có thể khác kiến trúc.
- **R3:** không mẫu dữ liệu thô / record `(x, y)` nào rời xe.
- **R4:** 66 feature gốc, cùng protocol `odixe06/fl_dagsnet_veremi` để so sánh.

## Trạng thái — bàn giao 2026-10-11 (chủ dự án tắt máy)

**Không có tiến trình nào đang chạy.** P2 đã khởi động trên local rồi bị dừng sau ~6 phút, chưa có kết quả; output rỗng đã xóa.
Không có kernel Kaggle nào của repo này.

- **Archive 3 đã đóng và xóa local** (2026-10-11).
  - Repo `odixe06/mobihet-archive-b9-b21`, tag `archive-3`, bundle `mobihet-archive-b9-b21.bundle`; mọi artifact trên DVC, đã kiểm thiếu 0.
  - B21 cuối: d3z PASS, Z66 và liều replay FAIL.
  - `~/.cache/mobihet` (39 GB scratch archive) đã xóa theo lệnh chủ dự án. Lấy lại: [backup.md](docs/operations/backup.md) §5.
- **Probe H2-T0-lite** (MLP, 2026-10-10): Gaussian full gộp +0,10 so với replay cân bằng. [Kết quả](docs/research/probes/h2-input-stats.md).
- **Pilot P1 XONG** (2026-10-11, local, 16 arm × 2 seed; [kết quả + phân tích](docs/research/probes/pilot-p1.md)).
  - Thiết lập: DAGSNet 20c, 2,5% dữ liệu mỗi client, 20 round, holdout `group`, quy tắc gp, không giới hạn kênh.
  - **Khả thi:**
    - H2 Gaussian lớp ở input (full +0,11, lr4 +0,09, diag +0,08 chỉ với 13 kB một lần; − Gaussian của chính xe +0,08 ⇒ lợi là tri thức quần thể);
    - H1 DML proxy (MLP64 +0,05, MLP256 +0,12, DAGSNet +0,16; proxy toàn cục 0,57 / 0,68 / 0,77).
  - Baseline: H4 FedProto +0,04. Mốc trên FedAvg +0,33 (không khác kiến trúc được).
  - **Trượt** (nguyên nhân trong kết quả + LESSONS): H3 LG +0,02 / FedGH −0,02, H5 DS-FL +0,075 (< H2) / FedGen +0,02.
  - Lỗi đánh giá đã thấy: `gp` sửa quá tay với DML. Từ nay DML báo cả `head`.
- **P2 (kênh) chưa chạy:** code [mh/p2.py](mh/p2.py) + [mh/channel.py](mh/channel.py) đã smoke test. Hàng đợi [scripts/p2_queue.txt](scripts/p2_queue.txt),
  cổng khóa ở [pilot-plan §5](docs/research/pilot-plan.md). Phải chạy trên **Kaggle** (xem quyết định 2026-10-11).

Code: [mh/pilot.py](mh/pilot.py) (9 phương pháp, một vòng FL chung), [mh/p2.py](mh/p2.py) (kênh), [mh/archs.py](mh/archs.py) (CNN1d/GRU, chưa dùng),
[scripts/report_pilot.py](scripts/report_pilot.py) (bảng + cổng). Artifact DVC: `papers/pilot/fixture.dvc` (fixture 153 MB), `papers/pilot/runs.dvc` (32 `result.json`).

## Quyết định hiện hành

**2026-10-10:**
- **Tiêu chuẩn FL:** không chia sẻ local data giữa các client dưới bất kỳ hình thức nào, kể cả trong đối chứng thí nghiệm.
- Dạng tri thức được phép: tham số khối model chung, prototype/thống kê theo lớp, logit trên dữ liệu proxy, generator/dữ liệu tổng hợp.
- Input **66 feature gốc**, không P4/map. Cấu hình khớp `fl_dagsnet_veremi` (DAGSNet, α=0,5, 20/50/100 client, 50 round × 1 epoch, cosine LR, test mọi round).
- Q1 ngân sách: báo nhiều mức, chính là B0 (≈4,5 kB/contact) và B2 (≈231 kB/contact); vẽ Pareto byte–F1.
- Q2 protocol: run chính như `fl_dagsnet_veremi` (headline r50); pilot chọn cấu hình trên holdout nội bộ, cổng khóa trước, hai seed.
- Q4 riêng tư: không dữ liệu thô rời xe + secure aggregation mô phỏng. DP để sau.
- DVC remote `gdrive` = `/mnt/g/My Drive/NCKH/veremi_build/mobihet/dvc-storage` (chung với archive). Được phân tích dữ liệu client trong `../dataset/`.

**2026-10-11:**
- Thử **mọi hướng H1–H5** (đã làm ở P1). Hướng trượt phải có nguyên nhân + bài học.
- **Mọi client dùng DAGSNet** để so với các phương pháp đã chạy full DAGSNet. Chưa chạy đa mô hình; chỉ ghi khả thi R2 về thiết kế.
- **Được commit + push** repo này ở các mốc. Commit không thêm attribution Claude (skill repo-backup).
- **Đồng ý** thử kết hợp **H1 + H2**, và chạy protocol đầy đủ cho hướng thắng.
- **Tiến trình nặng chạy trên Kaggle, không chạy ở máy local.** Local chỉ để smoke test, dựng fixture nhỏ, report.
- Xóa `~/.cache/mobihet` — đã làm.

## Còn mở

Q3 số contact/round (đang dùng 1 contact/xe/round), Q7 tắt M4b (đang tắt, đúng "66 feature gốc").
Q5 (proxy cho H5) và Q6 (thứ tự) không còn cần: H5 dừng, thứ tự theo P1.

## Bước kế tiếp (theo thứ tự)

1. **Đưa P2 lên Kaggle.**
   - Upload `papers/pilot/fixture` (153 MB) làm Kaggle dataset private.
   - Notebook chạy `python -m mh.p2 ...` theo `scripts/p2_queue.txt`: 11 arm × 2 seed, ~30 phút/run trên GPU cỡ 3050, 2 run/GPU song song trên 2×T4.
   - Dùng skill `kaggle-training-notebook` + `scripts/kaggle/push_kaggle.py`; image pin trong `knowledge/runtime.json`. Collect → `papers/p2/runs` → report.
2. **H1 + H2 kết hợp** (pilot, Kaggle):
   - private DAGSNet + proxy (MLP256 ở B2 / MLP64 ở B0) + mẫu Gaussian H2;
   - đối chứng là từng thành phần riêng của P1; cổng khóa trước (gợi ý: kết hợp − max(H1, H2) ≥ +0,02 cả hai seed).
   - Lưu ý `gp` với DML: báo cả `head`; cân nhắc ước prior hiệu dụng.
3. **Protocol đầy đủ trên Kaggle** cho hướng thắng + mốc local-only (50 round, 43 M dòng, test toàn bộ mỗi round, 20c trước), dựng từ driver NILM-FL
   của `fl_dagsnet_veremi` (`nilm_fl/papers/nilm-li-2024/proj/`; 2×T4, resume nhiều phiên, ~10–12 h/run). Chủ dự án đã đồng ý; báo quota trước khi push.

## Muốn biết → mở

| câu hỏi | tài liệu |
|---|---|
| Kết quả pilot, nguyên nhân hướng trượt | [pilot-p1.md](docs/research/probes/pilot-p1.md) |
| Thiết lập + cổng P1/P2 | [pilot-plan.md](docs/research/pilot-plan.md) |
| Archive đã thử gì, kết quả, bài học | [kiến thức kế thừa](docs/legacy/kien-thuc-ke-thua.md), `archive3:docs/experiments/index.md` |
| Đề xuất, ngân sách byte | [de-xuat-huong-moi.md](docs/research/de-xuat-huong-moi.md) |
| Protocol so sánh | [protocol.md](docs/research/protocol.md) |
| Dữ liệu / DAGSNet / máy local / Kaggle | [knowledge/](knowledge/README.md) |
| Backup, kéo dữ liệu archive | [backup.md](docs/operations/backup.md) |
