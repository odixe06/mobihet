# CONTEXT — MobiHet (hướng mới, không chia sẻ dữ liệu)

Đọc file này → **[LESSONS.md](LESSONS.md)** → [kết quả pilot P1](docs/research/probes/pilot-p1.md) → [kế hoạch pilot](docs/research/pilot-plan.md)
→ (khi cần nền) [kiến thức kế thừa](docs/legacy/kien-thuc-ke-thua.md), [đề xuất H1–H6](docs/research/de-xuat-huong-moi.md).

IDS V2X cộng tác qua RSU. Bốn yêu cầu bắt buộc:
- **R1:** contact hữu hạn, mọi byte trả phí, có deadline thật.
- **R2:** mỗi xe model riêng, có thể khác kiến trúc.
- **R3:** không mẫu dữ liệu local thật / record thật `(x, y)` nào rời xe; synthetic khởi tạo từ nhiễu được phép.
- **R4:** 66 feature gốc, cùng protocol `odixe06/fl_dagsnet_veremi` để so sánh.

## Trạng thái — phiên tiếp tục 2026-10-11

- **P2 đã thu đủ4kernel/22run ×20round.** Kiểm nhất quán bằng chứng lưu qua; chưa xác minh weights/prediction vì không được lưu.
  Prior/codec có giới hạn: [kết quả P2](docs/research/probes/pilot-p2.md). Các trạng thái RUNNING/chưa chạy bên dưới là bàn giao trước phiên này.
- **Đợt mới ngoài H1–H6:** [condensation + Fourier distribution sketch](docs/research/de-xuat-bo-sung-2026-10-11.md),
  [pilot S-a/S-b và cổng khóa](docs/research/pilot-new-fl.md). **Preflight local qua; S-a đã push2kernel v1 lúc12:39 UTC+7**.
  Kernel ID/quota/lệnh collect ở [new-fl/rebuild.md](papers/new-fl/rebuild.md); chưa có kết quả Kaggle xác minh.
- User xác nhận giữ DAGSNet, R2 ở thiết kế; local chỉ smoke. **Cho phép RSU đọc codebook tổng hợp**
  khởi tạo từ nhiễu; không raw record. Ngoại lệ Q4 này chỉ áp cho synthetic; thống kê vẫn gộp nhiều client.

### Bàn giao trước khi tiếp tục (lịch sử)

**Không có tiến trình local nào đang chạy.** P2 từng khởi động trên local rồi bị dừng sau ~6 phút (chưa có kết quả, output rỗng đã xóa).
**P2 đã push lên Kaggle 2026-10-11** (4 kernel, xem §P2 Kaggle). Phiên sau: thu kết quả → report → cổng.

- **Archive 3 đã đóng và xóa local** (2026-10-11).
  - Repo `odixe06/mobihet-archive-b9-b21`, tag `archive-3`, bundle `mobihet-archive-b9-b21.bundle`; mọi artifact trên DVC, đã kiểm thiếu 0.
  - B21 cuối: d3z PASS, Z66 và liều replay FAIL.
  - `~/.cache/mobihet` (39 GB scratch archive) đã xóa theo lệnh chủ dự án. Lấy lại: [backup.md](docs/operations/backup.md) §5.
- **Probe H2-T0-lite** (MLP, 2026-10-10): Gaussian full gộp +0,10 so với replay cân bằng. [Kết quả](docs/research/probes/h2-input-stats.md).
- **Pilot P1 XONG** (2026-10-11, local, 16 arm × 2 seed; [kết quả + phân tích](docs/research/probes/pilot-p1.md)).
  - Thiết lập: DAGSNet 20c, 2,5% dữ liệu mỗi client, 20 round, holdout `group`, quy tắc gp, không giới hạn kênh.
  - **Khả thi:**
    - H2 Gaussian lớp ở input (full +0,11, lr4 +0,09, diag +0,08 chỉ với 13 kB một lần; − Gaussian của chính xe +0,08 ⇒ lợi là tri thức quần thể);
    - H1 DML proxy (MLP64 +0,05, MLP256 +0,12, **DAGSNet +0,16** — chỉ proxy DAGSNet được dùng tiếp; proxy toàn cục 0,57 / 0,68 / 0,77).
  - Baseline: H4 FedProto +0,04. Mốc trên FedAvg +0,33 (không khác kiến trúc được).
  - **Trượt** (nguyên nhân trong kết quả + LESSONS): H3 LG +0,02 / FedGH −0,02, H5 DS-FL +0,075 (< H2) / FedGen +0,02.
  - Lỗi đánh giá đã thấy: `gp` sửa quá tay với DML. Từ nay DML báo cả `head`.
- **P2 (kênh) chưa chạy:** code [mh/p2.py](mh/p2.py) + [mh/channel.py](mh/channel.py) đã smoke test. Hàng đợi [scripts/p2_queue.txt](scripts/p2_queue.txt),
  cổng khóa ở [pilot-plan §5](docs/research/pilot-plan.md). Phải chạy trên **Kaggle** (xem quyết định 2026-10-11).

Code: [mh/pilot.py](mh/pilot.py) (9 phương pháp, một vòng FL chung), [mh/p2.py](mh/p2.py) (kênh), [mh/archs.py](mh/archs.py) (CNN1d/GRU, chưa dùng),
[scripts/report_pilot.py](scripts/report_pilot.py) (bảng + cổng). Artifact DVC: `papers/pilot/fixture.dvc` (fixture 153 MB), `papers/pilot/runs.dvc` (32 `result.json`).

## P2 trên Kaggle (push 2026-10-11 02:17–02:21 UTC, cả 4 RUNNING lúc 02:23 UTC)

| kernel | owner | seed | arm |
|---|---|---|---|
| `khanhmay0304/mobihet-p2-h2-s42` | khanhmay0304 | 42 | 7 arm H2 (B0: full/diag naive+h6, lr4 h6; B2: full naive+h6) |
| `minhtriethihi/mobihet-p2-h2-s43` | minhtriethihi | 43 | như trên |
| `catbaochau/mobihet-p2-dml-s42` | catbaochau | 42 | 4 arm H1 proxy DAGSNet int8 (B3 naive, B2 naive+h6, B0 h6) |
| `minhtrit06/mobihet-p2-dml-s43` | minhtrit06 | 43 | như trên |

- Input: dataset private `<owner>/mobihet-p2-src` (phẳng: code `mh/` + `meta.json` + fixture pilot). Notebook tự dựng lại cây thư mục và kiểm sha256 fixture.
- Mỗi kernel: 2×T4, 4 job song song (2 job/GPU). Ngừng nhận job mới sau 9 h. Output chỉ gồm `runs/<arm>_s<seed>/{result.json,log.txt}` + `summary_s<seed>.json`.
- Ước 1,5–3 h mỗi kernel (chưa đo trên T4). Quota trước khi push: 4 owner còn 20,9–22,9 h GPU (refresh 2026-10-17).
- Không có W&B: muốn xem tiến độ thì đợi COMPLETE rồi pull log. Kernel ERROR ngay đầu thường là assert 2×T4 → chạy lại y nguyên ở account khác trước khi sửa code.
- Nguồn: [scripts/kaggle/gen_p2.py](scripts/kaggle/gen_p2.py) → `papers/p2/notebook/<rid>/` (có marker `PUSHED`), push bằng [scripts/kaggle/push_nb.py](scripts/kaggle/push_nb.py).
- **Thu:**
  1. `python scripts/kaggle/collect_p2.py` (status + pull; copy vào `papers/p2/runs/`);
  2. `python scripts/report_p2.py > papers/p2/report.md`;
  3. cổng P2-a/P2-b ở [pilot-plan §5](docs/research/pilot-plan.md);
  4. `dvc add papers/p2/runs` + push;
  5. ghi kết quả vào `docs/research/probes/pilot-p2.md`.
- Kernel lỗi: đọc log (`papers/p2/kaggle/<rid>/<slug>.log`). Không sửa notebook đã PUSHED; sửa generator → thư mục rid mới.

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
- **Proxy của các phương án dùng proxy (H1, H1+H2) cũng là DAGSNet**, cấu hình đồng nhất cho mọi client.
  Hệ quả: 797 kB fp16 / 395 kB int8 mỗi chiều mỗi round. Ở B0 chỉ có thể truyền dần qua nhiều contact (H6); ở B2 cần ~2 contact mỗi chiều.
- Xóa `~/.cache/mobihet` — đã làm.

## Còn mở

Q3 số contact/round (đang dùng 1 contact/xe/round), Q7 tắt M4b (đang tắt, đúng "66 feature gốc").
Q5 (proxy cho H5) và Q6 (thứ tự) không còn cần: H5 dừng, thứ tự theo P1.

## Bước kế tiếp (theo thứ tự)

**Ưu tiên mới trong phiên 2026-10-11:** hoàn tất preflight → S-a hai seed trên Kaggle, quota thấp trước → verify → S-b nếu qua cổng.
Các mục H1+H2/full-scale dưới đây vẫn được đề xuất/đã đồng ý trước, chưa khởi chạy ở phiên này.

1. **Thu P2 từ Kaggle** (đã push, xem §P2 trên Kaggle) → report → cổng → ghi kết quả + bài học.
2. **H1 + H2 kết hợp** (pilot, Kaggle):
   - private DAGSNet + proxy **DAGSNet** + mẫu Gaussian H2;
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
