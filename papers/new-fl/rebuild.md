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
| v1 | `minhtran0601/mobihet-sa-s42-20261011` |42|2026-10-11 12:39|15,1886h /0h|COMPLETE14:42 UTC+7;6arm ×20round đã verify14:43 |
| v1 | `dinhan25/mobihet-sa-s43-20261011` |43|2026-10-11 12:39|15,5158h /0h|COMPLETE;6arm ×20round đã verify |

Hai private dataset đãready. Mỗi notebook chạy6arm,4job song song (2job/T4), không W&B.
Đây là v1 đã push trước yêu cầu mới. **Notebook huấn luyện tạo tiếp theo bắt buộc W&B API key+online**,
theo [LESSONS.md](../../LESSONS.md); không sửa v1 đã chạy.
Không nhận job mới sau2h; estimate trước chạy cả notebook≤2h. Notebook yêu cầu đúng2×T4 trước train.
Worker log thực tế: seed42 `time` cuối3011,51s (50,19phút), seed43 3170,60s (52,84phút).
Seed42 từ push tới COMPLETE khoảng123phút; log không xác định nguyên nhân phần chờ ngoài worker, không gọi là thời gian train.
Quota đọc sau COMPLETE lúc14:43 UTC+7: minhtran0601 free14,351889h/reserved0, dinhan25 free14,634846h/reserved0;
chênh trước/sau đợt này lần lượt0,836733h và0,880941h GPU. Refresh17/10 UTC.
Snapshot RUNNING có thể khác số cuối: seed42 free13,679130h trước khi tăng lại14,351889h sau COMPLETE.
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
`collect_new_fl.py --watch` chờ và thu tự động; không verify lại kernel đã có marker đúng kernel/v1/source.
Read-only API probe13:10 UTC+7 xác nhận0file/0log khi seed42 đangRUNNING; chưa có telemetry từng arm.
CLI list file14:24 UTC+7 vẫn rỗng, exit0. Quota account-scoped đọc14:10 UTC+7:13,6791h GPU free,
reserved0; refresh17/10 UTC. Kernel đã chạy khoảng110phút lúc14:29; chưa biết round qua API.
Chủ dự án gửi log GUI lúc khoảng14:35 UTC+7: mốc2644,4s, `dm_gauss` r16 và `dm` r14;
loss hữu hạn, đang tiến. Đây là log người dùng gửi, chưa thay marker thu/verify; ETA train6–9phút từ mốc log.
Output giữ ở `kaggle/<rid>/runs/`; `runs/<arm>_s<seed>` là symlink tới output đã verify, không nhân đôi8GiB weights.
Mỗi seed pull309file/3.814MiB, không thiếu; executed notebook v1, source/fixture/hardware2×T4 đã kiểm.
Seed42 pull83s,46,1MiB/s; seed43 pull84s,45,5MiB/s. Collector exit0, peak tree RSS723MiB.
**12run đủ20round**, weights hữu hạn/strict-load, predictions→confusion→10metric từng private và hash chain qua.
Source `meta.json`/`dagsnet.py`/`metrics.py` dùng verifier được đối chiếu riêng với manifest đã chạy của cả hai kernel.
Report [report.md](report.md) + JSON/CSV/PNG/SVG đã dựng từ artifact; KME/DM trượt cổng, không mở S-b.
S-a mới lượng tử hóa giá trị và tính payload; chưa có wire codec/frame train. Codebook chỉ lưu tọa độ tổng hợp;
64B metadata/count trong dự toán không phải bằng chứng phát histogram local riêng. Prior real-count đi qua tổng cohort.
Không protocol full-scale/test20% trong đợt này.
Không có yêu cầu backup/xóa artifact; dữ liệu mới chưa DVC add/push.
