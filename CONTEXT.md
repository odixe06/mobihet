# CONTEXT — MobiHet (hướng mới, không chia sẻ dữ liệu)

Đọc file này → **[LESSONS.md](LESSONS.md)** → [kết quả pilot P1](docs/research/probes/pilot-p1.md) → [kế hoạch pilot](docs/research/pilot-plan.md)
→ (khi cần nền) [kiến thức kế thừa](docs/legacy/kien-thuc-ke-thua.md), [đề xuất H1–H6](docs/research/de-xuat-huong-moi.md).
Tìm file: [index theo nhu cầu](README.md) → [danh mục toàn dự án](docs/file-index.md).
Phiên đanghoạtđộng/collectors/reportwatch/sourcefrozen: [live2026-10-12](docs/operations/live-2026-10-12.md).

IDS V2X cộng tác qua RSU. Bốn yêu cầu bắt buộc:
- **R1:** contact hữu hạn, mọi byte trả phí, có deadline thật.
- **R2:** mỗi xe model riêng, có thể khác kiến trúc.
- **R3:** không mẫu dữ liệu local thật / record thật `(x, y)` nào rời xe; synthetic khởi tạo từ nhiễu được phép.
- **R4:** 66 feature gốc, cùng protocol `odixe06/fl_dagsnet_veremi` để so sánh.

**2026-10-12 backup/cleanup theo yêu cầu chủ dự án:** Git snapshot nghiên cứu và hồ sơ đãpush;
31/31 target DVC đủ trênG:,8.847 object verify thiếu0; workspace đãxóa, thu hồi155.99GiB local.
Không chờcloudsync. [Hồ sơ/phục hồi](docs/operations/backup.md).
Không collector local đang chạy; muốn collect/resume phải pull target cần thiết trước.
Quyền Git/DVC/delete chỉ cho đợt backup này, không tự mở run nghiên cứu mới.

## Trạng thái — triển khai kiểm khả thi V2 ngày2026-10-11

**Chủ dự án đã yêu cầu triển khai và kiểm khả thi V2**, thay phạm vi chỉ Markdown trước đó.
[Bộ đề xuất V2](docs/research/de-xuat-v2/README.md): proxy sketch + bù sai số; proxy trong cơ sở sinh từ seed;
DAGSNet học tỷ số mật độ để điều chỉnh Gaussian. [Pilot/cổng đã chốt](docs/research/pilot-v2.md),
[ledger triển khai](papers/v2/rebuild.md). Local math/16arm smoke/exact-resume PASS; **A hai seed ERROR nhưng đã audit xong**:
A đã thu/verify14arm:10arm đủ20round,4EF dừng3–6round; parent còn lỗi quét key trong notebook nền tảng.
**A small/medium FAIL hai seed, không rerun y nguyên.** [Kết quả/chẩn đoán](docs/research/probes/pilot-v2.md).
**B đã thu/audit cảhai seed:**512/own512 lỗi numerical r1, không commit;1024 đủ20round nhưng teacher gate FAIL.
Không rerun y nguyên B, không mở KD/ZO/contact. **C đãverify đủ8arm×20round, FAIL hai seed**:
FL−diag≈−0,0167/−0,0208, ESS≈0,007–0,024; không mở contact/rerun y nguyên.
[Live status](papers/v2/live-status.json) chỉ bằng chứng tiến độ W&B; [report](papers/v2/report.md) là artifact đãaudit.
**V3 đã push8kernel lúc2026-10-11 17:00UTC (2026-10-12 00:00UTC+7):**4carrier×2seed,
fp32/private+KD; [manifest](docs/research/pilot-v3.md), [ledger](papers/v3/rebuild.md).
48run W&B exact fingerprint; collector4family đang chờ terminal. Local math/23arm smoke và4carrier exact-resume PASS;
dense32 control+4representatives/resume PASS. Chưa kết quả F1 V3.
Chủ dự án chốt bổ sung **K2b soft target gộp**, **K3 IPC1/4, trần2×baseline**, **MALA cấu hình đề xuất**;
**V4 MALA/support đãpush4kernel lúc17:22UTC/00:22UTC+7**, localfidelity/math/5arm mỗi family,
MALA3exactresume/support2exactresume+inference PASS; [MALA](docs/research/pilot-v4-mala.md),
[support](docs/research/pilot-v4-support.md), [ledger](papers/v4/rebuild.md). ChưaF1 V4;20exactW&Bidentities.
**K2b/K3 đãchốt vàpush6kernelV5 lúc18:52UTC/01:52UTC+7**: bandwidthtrain-only,
80steptổngluânphiênclass/IPC1/4/refresh5round/2×baseline; [H](docs/research/pilot-v5-anchors.md),
[G/J](docs/research/pilot-v5-books.md), [ledger](papers/v5/rebuild.md).
23arm/6exactresumes/inference/full80localbookPASS;46exactW&Bidentities, chưaF1V5.
**V4support đủ10arm×20round đãaudit,pairedFAIL**: structural−diag−0,00212/−0,00824,
khôngcontact/rerun. [V4kếtquả](docs/research/probes/pilot-v4.md). MALAstatusRUNNING;19:11UTC đãcóW&Bidentity, diag tới r2; MALA chưa round,
khôngclaimhealthy tới khi cólog/artifact; khôngcầnGUI.
K5algebra/primitive mới [thiếtkế+boundedprobe](docs/research/v5-algebra-sources.md), chưacodephươngpháp/pilot;
không sửa source V3/V4 đãpush. Tự chờ/thu log, không cần chủ dự án kiểm GUI.
A cả small/medium; B FO512/1024 trước, ZO có điều kiện; C τ0,05/cap10/class-normalize/q-BN đã được xác nhận.
**Chốt bổ sung: mọi mô hình học được đều DAGSNet**, kể cả teacher/model phụ; loại tree/GMM học được khỏi đợt này.
R2 vẫn ở thiết kế. Train nặng chỉ Kaggle, W&B online bắt buộc; Git/DVC/push chỉ khi yêu cầu riêng.
[Bàn giao pilot trước đó](docs/operations/handoff-2026-10-11.md): kết quả, source/config, quota, artifact, bài học và việc còn lại.

- **P2 đã thu đủ4kernel/22run ×20round.** Kiểm nhất quán bằng chứng lưu qua; chưa xác minh weights/prediction vì không được lưu.
  Prior/codec có giới hạn: [kết quả P2](docs/research/probes/pilot-p2.md). Các trạng thái RUNNING/chưa chạy bên dưới là bàn giao trước phiên này.
- **Đợt mới ngoài H1–H6:** [condensation + Fourier distribution sketch](docs/research/de-xuat-bo-sung-2026-10-11.md),
  [pilot S-a/S-b và cổng khóa](docs/research/pilot-new-fl.md). **S-a hoàn tất và verify12run ×20round lúc14:43 UTC+7**.
  **KME/DM đều trượt cổng; S-b đóng.** [Kết quả + chẩn đoán](docs/research/probes/pilot-new-fl.md),
  [report đủ10metric + hình](papers/new-fl/report.md). Kernel/quota/source ở [rebuild](papers/new-fl/rebuild.md).
  Không còn train/collector hoạt động trong đợt này. S1 Nyström vẫn là thiết kế chưa pilot.
- User xác nhận giữ DAGSNet, R2 ở thiết kế; local chỉ smoke. **Cho phép RSU đọc codebook tổng hợp**
  khởi tạo từ nhiễu; không raw record. Ngoại lệ Q4 này chỉ áp cho synthetic; thống kê vẫn gộp nhiều client.
- **Notebook huấn luyện mới bắt buộc W&B API key + theo dõi online** (chốt trong phiên, xem LESSONS).
  Hai notebook S-a v1 đã PUSHED trước quyết định này, giữ source để thu/verify.

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
- **P2 (kế hoạch trước push; hiện đã thu/audit):** code [mh/p2.py](mh/p2.py) + [mh/channel.py](mh/channel.py) đã smoke test. Hàng đợi [scripts/p2_queue.txt](scripts/p2_queue.txt),
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
- P2 v1 đã chạy không có W&B: xem tiến độ qua log sau COMPLETE. Notebook tạo mới theo quy định W&B ở LESSONS.
  Kernel ERROR ngay đầu thường là assert 2×T4 → chạy lại y nguyên ở account khác trước khi sửa code.
- Nguồn: [scripts/kaggle/gen_p2.py](scripts/kaggle/gen_p2.py) → `papers/p2/notebook/<rid>/` (có marker `PUSHED`), push bằng [scripts/kaggle/push_nb.py](scripts/kaggle/push_nb.py).
- **Thu:**
  1. `python scripts/kaggle/collect_p2.py` (status + pull; copy vào `papers/p2/runs/`);
  2. `python scripts/report_p2.py > papers/p2/report.md`;
  3. cổng P2-a/P2-b ở [pilot-plan §5](docs/research/pilot-plan.md);
  4. backup chỉ khi chủ dự án yêu cầu riêng;
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
- **Quyết định mới nhất: không tự commit/push Git, DVC add/push/backup; chỉ làm khi chủ dự án yêu cầu rõ.** Thay quyền theo mốc trước đó.
- **Đồng ý** thử kết hợp **H1 + H2**, và chạy protocol đầy đủ cho hướng thắng.
- **Tiến trình nặng chạy trên Kaggle, không chạy ở máy local.** Local chỉ để smoke test, dựng fixture nhỏ, report.
- **Proxy của các phương án dùng proxy (H1, H1+H2) cũng là DAGSNet**, cấu hình đồng nhất cho mọi client.
  Hệ quả: 797 kB fp16 / 395 kB int8 mỗi chiều mỗi round. Ở B0 chỉ có thể truyền dần qua nhiều contact (H6); ở B2 cần ~2 contact mỗi chiều.
- Xóa `~/.cache/mobihet` — đã làm.

## Còn mở

Q3 số contact/round (đang dùng 1 contact/xe/round), Q7 tắt M4b (đang tắt, đúng "66 feature gốc").
Q5 (proxy cho H5) và Q6 (thứ tự) không còn cần: H5 dừng, thứ tự theo P1.

## Kế hoạch V2 và backlog

Đọc [ưu tiên V2](docs/research/de-xuat-v2/README.md) và [hợp đồng kiểm nghiệm](docs/research/de-xuat-v2/thiet-ke-chung.md)
trước chọn hướng code. V2-A được đề nghị kiểm đầu tiên; V2-C kiểm chất lượng riêng; V2-B rủi ro sức chứa.
Đã được yêu cầu triển khai; khóa cổng U theo [pilot V2](docs/research/pilot-v2.md).
Local smoke → Kaggle hai seed → pull/verify/report; chỉ hướng qua U mới kiểm contact. Chưa mở full-scale.

**Đợt S-a mới đã đóng:** KME/DM trượt cổng hai seed; giữ artifact/đối chứng, không tune/kéo dài để cứu.
S1 Nyström là phương án byte còn chưa pilot; nếu triển khai tiếp thì khóa thiết kế/cổng riêng và bắt buộc W&B online.
Các mục H1+H2/full-scale dưới đây vẫn được đề xuất/đã đồng ý trước, chưa khởi chạy ở phiên này.

1. **P2 đã thu/report/audit**; đọc [giới hạn](docs/research/probes/pilot-p2.md) trước dùng số cũ làm đối chứng.
2. **H1 + H2 kết hợp** (pilot, Kaggle):
   - private DAGSNet + proxy **DAGSNet** + mẫu Gaussian H2;
   - đối chứng là từng thành phần riêng của P1; cổng khóa trước (gợi ý: kết hợp − max(H1, H2) ≥ +0,02 cả hai seed).
   - Lưu ý `gp` với DML: báo cả `head`; cân nhắc ước prior hiệu dụng.
3. **Protocol đầy đủ trên Kaggle** cho hướng thắng + mốc local-only (50 round, 43 M dòng, test toàn bộ mỗi round, 20c trước), dựng từ driver NILM-FL
   của `fl_dagsnet_veremi` (`nilm_fl/papers/nilm-li-2024/proj/`; 2×T4, resume nhiều phiên, ~10–12 h/run). Chủ dự án đã đồng ý; báo quota trước khi push.

## Muốn biết → mở

| câu hỏi | tài liệu |
|---|---|
| Sau P1/P2/S-a còn hướng FL nào, giữ mọi learned model DAGSNet? | [Bộ đề xuất V2](docs/research/de-xuat-v2/README.md), [nguồn gốc](docs/research/de-xuat-v2/nguon-nghien-cuu.md) |
| Kết quả pilot, nguyên nhân hướng trượt | [pilot-p1.md](docs/research/probes/pilot-p1.md) |
| Thiết lập + cổng P1/P2 | [pilot-plan.md](docs/research/pilot-plan.md) |
| Archive đã thử gì, kết quả, bài học | [kiến thức kế thừa](docs/legacy/kien-thuc-ke-thua.md), `archive3:docs/experiments/index.md` |
| Đề xuất, ngân sách byte | [de-xuat-huong-moi.md](docs/research/de-xuat-huong-moi.md) |
| Protocol so sánh | [protocol.md](docs/research/protocol.md) |
| Dữ liệu / DAGSNet / máy local / Kaggle | [knowledge/](knowledge/README.md) |
| Backup, kéo dữ liệu archive | [backup.md](docs/operations/backup.md) |
