# Pilot S-a — condensation và Fourier distribution sketch

Thiết kế, lựa chọn cài đặt và cổng: [pilot-new-fl.md](../../docs/research/pilot-new-fl.md).
Nguồn gốc phương pháp + khác thuật toán paper: [đề xuất bổ sung](../../docs/research/de-xuat-bo-sung-2026-10-11.md).

## Đã kiểm local

- Math: additive moment/Fourier sums, kỳ vọng Fourier giải tích của Gaussian, optimization giảm loss,
  fp16 round-trip, frame bị deadline cắt không commit. Tolerance GEMM fp32: atol2e-6 cho tính cộng.
- Sáu arm, fixture512row/client và512row/holdout,3round; đủ weights/prediction/confusion/10metric.
- KME dừng sau r1, resume r2–3; tensor weights mọi round trùng bit với run liên tục.
- Verifier từ artifact thật: strict20model/round,395.024tham số, finite weights,
  predictions khớp confusion từng client, mean10metric/per-class/CSV, chain/hash config/source/fixture.
  Test sửa history đã bị từ chối.
- KME và DM: đủ200bước synthesis trên fixture smoke thật, hữu hạn. Peak RSS≤2,8GiB.
- Notebook metadata/image/private/source hashes; chạy đúng orchestration6job với worker mô phỏng.
  Local1GPU không xác nhận2×T4 hoặc thời gian Kaggle.

## Run history

| attempt | kernel | seed | push UTC+7 | trước push GPU free / reserved | trạng thái |
|---|---|---:|---|---|---|
| v1 | `minhtran0601/mobihet-sa-s42-20261011` |42|2026-10-11 12:39|15,1886h /0h|đã push, chưa thu |
| v1 | `dinhan25/mobihet-sa-s43-20261011` |43|2026-10-11 12:39|15,5158h /0h|đã push, chưa thu |

Hai private dataset đãready. Mỗi notebook chạy6arm,4job song song (2job/T4), không W&B.
Không nhận job mới sau2h; estimate cả notebook≤2h, chưa đo. Notebook yêu cầu đúng2×T4 trước train.
Không sửa notebook/input đã push. Source đã đóng băng trong `input/<owner>/source_manifest.json`;
worker xuất source+manifest trong output để audit source thật đã load.

## Thu và report

Probe local **chỉ truyền, không train**: `scripts/probe_new_fl_contacts.py`, `contact-probe.json`.
1contact/client/round,20round,32B header object+frame/ACK, cursor coherent;
histogram có byte, publication sau cohort≥5, commit availability sau tất cả contact để tránh lệch cid.
DM vòng nguồn/lớp dùng chu kỳ coprime19×16, không xếp hết một lớp trước các lớp khác.

| dạng tri thức | B0 median round nhận đầu | B2 median round nhận đầu | xe nhận tri thức r20 |
|---|---:|---:|---:|
| diag |4|2|20/20|
| KME |5,5|2|20/20|
| DM source/class object |3|2|20/20|

Đây là kết quả mô phỏng kích thước/availability, chưa phải codec/payload thực hoặc F1 dưới kênh;
không tính byte giao thức cryptographic secure aggregation. S-b phải kiểm bằng driver train/payload thực.

```bash
conda run --no-capture-output -n nckh python scripts/kaggle/collect_new_fl.py
conda run --no-capture-output -n nckh python scripts/report_new_fl.py
```

Collector: status→Range-resumable pull→executed notebook v1→source/hardware audit→verify từng arm.
Output giữ ở `kaggle/<rid>/runs/`; `runs/<arm>_s<seed>` là symlink tới output đã verify, không nhân đôi8GiB weights.
Chưa có kết quả S-a xác nhận, chưa mở S-b. Không protocol full-scale/test20% trong đợt này.
Không có yêu cầu backup/xóa artifact; dữ liệu mới chưa DVC add/push.
