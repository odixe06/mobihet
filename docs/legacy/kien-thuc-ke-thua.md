# Kiến thức kế thừa từ MobiHet cũ (Build1–21)

Cập nhật 2026-10-10. Tài liệu này lọc từ ba repo archive những gì **còn dùng được** khi hướng nghiên cứu mới cấm chia sẻ dữ liệu thô.
Số liệu ở đây chép từ hồ sơ đã verify trong archive. Muốn xem chi tiết hoặc kiểm lại thì mở đúng file nguồn ghi ở từng mục.

## 0. Vì sao dừng hướng cũ, archive nằm ở đâu

Từ Build7 đến Build21, cơ chế cộng tác của MobiHet là **record cache**. Xe gửi mẫu feature đã nén kèm nhãn thật lên RSU. RSU giữ pool rồi phát
lại cho các xe khác, và xe replay các mẫu đó để học. Cơ chế này đạt điểm cao nhất (Build19 stackp: group GP c16 0,806/0,818), nhưng mẫu dữ liệu đã
rời khỏi xe. Như vậy nó vi phạm nguyên tắc nền của federated learning là không làm lộ dữ liệu. **Chủ dự án dừng hướng này ngày 2026-10-10.**

| repo (private) | nội dung | local |
|---|---|---|
| `odixe06/mobihet-archive-b1-b6` | Build1–6: prototype/latent, screening, leakage stage 0–1, method probe M1–M4 | — |
| `odixe06/mobihet-archive-b7-b8` | Build7–8: cache mẫu thật + head riêng | — |
| `odixe06/mobihet-archive-b9-b21` (archive 3, trước đây là `odixe06/mobihet`) | Build9–21, toàn bộ hồ sơ kết quả mọi build, LESSONS, CONTEXT cuối | local đã xóa 2026-10-11; tag `archive-3`, bundle `mobihet-archive-b9-b21.bundle` |

Trong tài liệu này, `archive3:<path>` là đường dẫn trong repo archive 3 (nhánh `session-20261004-leakage-probes`). Dữ liệu nặng nằm ở DVC remote
chung `gdrive` và lấy lại bằng `dvc pull <target>.dvc` ([cách làm](../operations/backup.md)).

**Không mang sang hướng mới:** mọi thành phần chỉ có nghĩa khi có record cache, gồm pool/cache RSU, replay peer, multi-contact để tải thêm record,
nén record (fp8, Huffman Z/Z66), ACK ledger, thứ tự transfer, liều replay, chọn mẫu gửi theo utility. Các phát hiện về kênh truyền vẫn giữ ở §6.
Kết quả của các build này vẫn có giá trị như **đối chứng trên** (upper reference không privacy-compliant), không dùng làm phương pháp.

## 1. Bài toán và bất biến

| # | bất biến | nguồn |
|---|---|---|
| R1 | Contact xe–RSU hữu hạn: chỉ truyền trong thời gian xe nằm trong vùng phủ, có deadline thật, mọi byte phải trả phí (header, ACK, frame dở không được commit). | `archive3:MobiHet_IDS.md` §5.3 |
| R2 | Mỗi xe giữ model riêng, có thể khác kiến trúc. Không averaging tham số của **model riêng** giữa các kiến trúc. | `archive3:LESSONS.md` §1 |
| R3 | **Mới (2026-10-10):** không mẫu dữ liệu thô hoặc record `(x, y)` nào rời xe, kể cả lên RSU. Chủ dự án cho phép thử: tham số của khối model dùng chung, prototype/thống kê theo lớp, logit trên dữ liệu proxy, generator/dữ liệu tổng hợp. | phiên 2026-10-10 |
| R4 | **Mới (2026-10-10):** input dùng **66 feature gốc**, cùng cấu hình với repo `odixe06/fl_dagsnet_veremi`, để so sánh trực tiếp. Không dùng P4 temporal summary và map feature (Build18–21). | phiên 2026-10-10 |

Học, chuẩn bị gói tin và tính toán diễn ra **ngoài contact**. Contact chỉ ràng buộc giao tiếp. Đây là thiết kế từ 2026-10-05, vẫn giữ.

## 2. Dữ liệu — sự thật đã đo

Chi tiết: [knowledge/dataset.md](../../knowledge/dataset.md), [knowledge/architecture.md](../../knowledge/architecture.md).

- VeReMi NextGen: 43.045.415 dòng train (80%), 10.761.343 dòng test (20%). Cắt theo thời gian trong từng lớp×scenario. **Không có validation riêng**.
- `train/` và partition FL **đã z-score**, còn `test/` thì **raw**. Áp `knowledge/scaler.json` đúng một lần lên test.
  Dấu hiệu kiểm: mean(`f_snd_spd`) trên test raw = 7,4493.
- Partition Dirichlet α=0,5 cho 20/50/100 client. α=0,5 tạo lệch tỉ lệ, không làm thiếu lớp: hầu như client nào cũng đủ 16 lớp. **Không chia lại.**
- Client = receiver unit, nhưng một receiver không thuộc độc quyền một client. Flow có thể trải trên nhiều client.
- File parquet của client **sắp theo lớp**. Không lấy "k dòng cuối" làm mẫu; luôn lấy mẫu ngẫu nhiên có seed.
- Khóa phiên đúng là `(scenario, source_run, receiver_id, sender_alias)`.
- Lệch lớp 41:1, nên luôn đọc `f1_macro`. Accuracy gần như vô nghĩa vì Sybil + benign chiếm 44,5% test.
- **Shortcut đã biết:**
  - Sybil có `f_first_in_session` và cụm 13 feature điền 0.
  - Benign lấy từ run Sybil (attacker=0), nên nhóm `rate` mạnh bất thường.
  - **Vân tay receiver:** trong mẫu train của một client, chỉ cần biết receiver là đoán đúng nhãn 80–90% (prior 24–52%).
    9 feature noise/profile của receiver có η² ≈ 0,8–0,95.
  - M4b (đặt 9 feature này + 4 feature vị trí tuyệt đối về 0) cho +0,02…+0,03 ở pilot và +0,074 ở full data (Build8).
  - Theo R4, **mặc định không bật M4b**. Có thể thử M4b như ablation sau.
- **Trần do bộ sinh dữ liệu** (đọc code VeReMi NextGen `acda994`):
  - timeDelay cộng cùng Δ vào cả send/rcvTime nên không có dấu hiệu trong nội dung tin. Centralized chỉ đạt F1 0,23 (76% nhầm sang benign).
  - positionMirroring giữ heading. suddenConstantSpeed có 15,2% dòng báo 0 giống zeroSpeed.
  - Đừng đổ lỗi cho phương pháp học ở các lớp này trước khi xem trần thông tin.
- Centralized DAGSNet (66 feature, round 5): test `f1_macro` **0,8532** — đối chứng trên không privacy-equivalent.

## 3. Đánh giá — bài học về protocol

- **Holdout chia ngẫu nhiên theo dòng lạc quan +0,09…+0,12** (Build2). Phải dùng temporal và/hoặc receiver-group holdout.
  71% dòng validation thuộc flow đã xuất hiện ở phần train hoặc calibration (Build5 bước 0).
- Archive dùng hai holdout từ train80%: `old` (2% cuối thời gian, toàn 20 client) và `group` (receiver chưa thấy).
  Đây là hai population khác nhau, không so giá trị của chúng với nhau. Fixture B (scaler fit trên train80% trừ holdout) dựng bằng `archive3:scripts/m_fixtures.py`.
  Leakage của scaler đã đo là không đáng kể (|Δ| ≤ 0,004).
- `gp` = prior correction `argmax(logits − (1−τ)·log π_i + log π_g)`: +0,025 so với head ở Build3. Cần histogram nhãn toàn cục (đếm theo lớp).
  Chốt **một** quy tắc quyết định chính trước khi chạy. Kiểu "thắng ở bất kỳ quy tắc nào" từng chọn nhầm (Build3 `t3l_p4`).
- Báo đủ 10 metric. Với bài toán single-label thì accuracy = precision/recall/F1 micro = recall_weighted.
- **Khóa cổng trước khi chạy.** Cặp seed (42/43), mỗi arm đổi đúng một yếu tố.
  - Thành phần mượn mà chưa đo riêng thì phải đo trước hoặc thiết kế 2×2. Ở M15, interleave đã che mất lợi của cơ chế mới.
  - Benign recall dao động mạnh giữa các cặp (trung vị |Δ| 0,047), nên guard dùng benign F1.
- **Pilot không chuyển nguyên lên full scale.** Lợi ở full scale thường chỉ còn khoảng một nửa (EMA, D2), có khi đổi dấu (Build9-P).
  Slope dương ở cửa sổ ngắn không dự báo được trôi về sau. Lợi kiểu "thêm tri thức" lại tích lũy theo cycle, nên cửa sổ ngắn đánh giá thấp nó (Build16 gh).
- **Probe offline chỉ là điều kiện cần**, ví dụ Build5 N3 offline có lợi nhưng online thua. Thống kê tri thức phải tính trên dữ liệu chưa train.
- Kernel API COMPLETE ≠ artifact hợp lệ. Phải pull rồi verify bằng **đúng source đã chạy**.
- Repo `fl_dagsnet_veremi` chấm **toàn bộ test mỗi round** và lấy round 50 làm headline (không chọn checkpoint). Muốn so trực tiếp thì phải theo cùng cách này
  ([protocol](../research/protocol.md)).

## 4. Bằng chứng về các kênh tri thức **không dùng dữ liệu thô** (đã thử)

Đây là phần quan trọng nhất với hướng mới: những gì đã thử và **không** đi qua record.

| kênh | build | kết quả | nguồn |
|---|---|---|---|
| Prototype trung bình lớp (kiểu FedProto, align λp=20) + mobility | Build2 | so với không trao đổi +0,045 (validation). **Test 20c r50 0,5436**, 50c 0,4448, 100c 0,3607. Trôi sau r10–20 | `archive3:docs/experiments/build2/README.md` §7, `overview.md` §2 |
| Sửa prototype: InfoNCE, cluster k-means, soft-label, hiệu chỉnh classifier bằng tâm, bank tách lớp, hard prototype, projection head | Build3–4 (T2/T4/T5/T8/P1–P4) | mọi biến thể nằm trong **±0,006**; T5 soft-label có hại (gp 0,416); P3 projection head −0,026 | `archive3:docs/experiments/overview.md` §0, §5 |
| Đặc trưng ảo từ mean + hiệp phương sai hạng thấp (kiểu CCVR) | Build5 N3 | **offline** encoder đóng băng: +0,0126 so với ft0 (18/20 xe), hiệp phương sai > nhiễu đẳng hướng, thống kê trên dữ liệu chưa học > trên dữ liệu train. **Online** khi encoder vẫn học: −0,0144/−0,0105 | `archive3:docs/experiments/build5/README.md` §7, §9 |
| Directional alignment, bỏ separation, record từ EMA | Build6 U0/U1/R1 | không qua cổng; U1 −0,03 gp | `archive3:docs/experiments/build6/results.md` |
| KD từ teacher peer | Build7 | teacher đúng 63–85%, **không thêm gì** so với head | `archive3:docs/experiments/build7/README.md` |
| Teacher ridge/RFF tại RSU, phát coefficient, KD trên dữ liệu local | Build13 S13 | input − off +0,018/+0,019 nhưng input − classmean chỉ +0,0003/+0,0038 ⇒ **lợi ngang làm mượt theo lớp**; thua record replay −0,09/−0,08 | `archive3:docs/experiments/build13/results.md` |
| Label smoothing head / chọn theo margin sender | Build13 H13/U13 | head + nhưng gp −; U13 −0,007…−0,010 | như trên |
| Hiệu chỉnh head khi khác kiến trúc | M2 | homo +0,035/+0,039, **hetero ≈ +0,004** (MLP64 quá yếu, chỉnh lớp cuối không cứu được backbone) | `archive3:docs/experiments/method_probe/README.md` |

**Chẩn đoán rút ra** (đã kiểm ở nhiều build):

1. **Nút thắt là ranh giới quyết định đa receiver.**
   - Xe sai khoảng 4% trên receiver của chính nó, nhưng khoảng 45% trên tập toàn cục.
   - Probe giữ backbone local thuần, chỉ fit lại Linear cuối trên khoảng 1.200 mẫu đa receiver: 0,57 → 0,71–0,76 (Build8).
   - M1 oracle: 32 mẫu thật/lớp từ client khác đã cho +0,07…+0,09 (khoảng 70% lợi ích tối đa).
2. **Trung bình lớp không chở được thông tin ranh giới.** Các lớp nhầm nhau (benign/timeDelay/positionMirroring/dataReplay) rải gần đều trên một khối lớp.
   Đẩy các tâm ra xa nhau không tách được khối đó.
3. **Thống kê trong latent bị cũ đi khi representation còn học.** Đó là lý do N3 offline có lợi mà online thua.
   Ngược lại, head triển khai **fit lại mỗi cycle trên backbone EMA đóng băng** là cơ chế thắng (D3).
   ⇒ Thống kê latent chỉ nên dùng cho calibration trên snapshot đóng băng, không dùng làm loss liên tục.
4. **Backbone yếu cần tri thức tác động lên toàn model**, không chỉ head (M2, M1-real: hetero accfull − local +0,044/+0,059).
5. **Chia sẻ model (không phải dữ liệu) trên đúng bộ dữ liệu này hiệu quả.** Repo `fl_dagsnet_veremi`, đồng nhất DAGSNet, không ràng buộc truyền, test 20c r50:
   AFPHA (FedAvg+FedProx) 0,775; FD-IDS 0,7655; NILM-FL (học tương hỗ private↔proxy, chỉ trung bình proxy) proxy 0,823 / private 0,696.
   Tức là tri thức đa receiver đi được qua tham số. Câu hỏi còn lại là làm sao đi qua được **contact hữu hạn** và **model khác kiến trúc**.

## 5. Kỹ thuật học cục bộ còn dùng được (không phụ thuộc kênh cộng tác)

| kỹ thuật | bằng chứng | điều kiện dùng |
|---|---|---|
| `gp` prior correction | +0,025 so với head (Build3); quy tắc chính từ Build3 trở đi | cần π_g = histogram nhãn đã giao (đếm, trả byte); không đọc prior của test |
| EMA trọng số của chính xe (T3/T3L), half-life 2 cycle | T3L gp r46–50 +0,037 so với ctrl (Build3); EMA + head fit riêng thắng ở full scale (Build11 D3) | EMA trung bình cả head thì hỏng (Build9-P −0,025). Head triển khai phải khớp backbone |
| Tính lại BN trên dữ liệu local cho model EMA (L) | offline +0,011/+0,010, Kaggle +0,006/+0,006, **lợi giảm dần** theo cycle | EMA trung bình cả running stats, không khớp với weights trung bình (kiểu SWA). Trước khi kết luận một phương pháp có hại, tính lại BN rồi chấm lại |
| Head triển khai fit lại mỗi cycle (16 bước) | Build11 D3 +0,028 group so với head; fit lâu hơn thì overfit (4.096 bước −0,017/−0,023) | trước đây fit trên cache; hướng mới phải fit trên nguồn hợp lệ (local hoặc tổng hợp) |
| LR cosine theo round 1e-3 → 1e-5 | các repo anh em chốt 2026-09-13: chặn đà trôi nhưng không kéo lại được đỉnh | lịch thống nhất để so sánh |
| AMP fp16 + GradScaler, guard skip theo cycle | guard theo session làm crash run dài (Build12 V2) | xét theo cycle/attempt/applied update |
| Không đóng băng BN, không tách optimizer khi chưa có đối chứng | Build12 N4/N5, probe gradient | thay đổi phải có đối chứng chỉ đúng nguyên nhân |

**Trôi do học thuộc cục bộ** có ở mọi cấu hình full data. CE local xuống 0,01–0,03 trong khi holdout đi xuống từ khoảng r10–20 (Build2–8).
Nguyên nhân là học thuộc cục bộ; bank prototype gộp lớp chỉ là triệu chứng.

**Input/feature** từng là đòn bẩy lớn nhất (P4 + map +0,12/+0,135, Build18), nhưng không dùng theo R4. Giữ làm hướng mở rộng sau khi đã so sánh xong.
Code: `archive3:scripts/ab_p4_fixture.py`, `map_lane_features.py`.

## 6. Kênh truyền và mô phỏng contact

Simulator: `archive3:papers/build20/notebook/b20_s42_stackpz/proj/mobility.py`. Hàm `contact_state`/`transfer` nằm trong `pilot.py` cùng thư mục.

- Mỗi (xe, round) có:
  - vị trí đều trên đĩa RSU bán kính R = 800 m, heading đều, tốc độ U[5, 20] m/s;
  - `tau_true` là thời gian rời đĩa; `tau_hat` ước lượng từ vị trí/vận tốc có nhiễu (8 m, 1,5 m/s);
  - rate ước lượng U[20, 80] kB/s, rate thật = rate ước lượng × U[0,75; 1,15]; guard 2 s.
  - Mọi số là hàm thuần của `(cfg, round, client)`, không đọc đồng hồ. (Build1 dùng wall-clock GPU nên hỏng.)
- **Ngân sách stress đã dùng** (Build9–21): `contact_scale` 0,01, `rate_scale` 0,2, `byte_cap` 32.768 B/contact.
  - Thực giao trung bình **≈ 4,7 KB/contact** (Build12 V1: 4.738.971 B / 1.000 contact), 1 contact/xe/cycle.
  - Lấp phần contact bỏ trống + bỏ ACK download: download ×1,63 (Build16).
  - Đây là **knob tạo stress** cho kênh record, không phải số đo vật lý. Với rate gốc (20–80 kB/s) và contact vài chục–vài trăm giây, ngân sách vật lý là cỡ MB/contact.
  - ⇒ **Hướng mới phải chốt lại ngân sách** ([câu hỏi Q1](../research/de-xuat-huong-moi.md#6-câu-hỏi-cần-chủ-dự-án-chốt)).
- **Quy tắc kế toán byte (giữ nguyên):**
  - Tính đủ header, schema/version, catalog/cursor, ACK, retransmission.
  - Deadline thật cắt frame thì trả phí byte đã truyền nhưng không commit frame đó.
  - Không có cache hit hay dữ liệu prior miễn phí.
  - Chỉ kích hoạt snapshot model/thống kê **đủ và coherent**, không ghép mảnh từ các version khác nhau (thiết kế Build13 §2).
- **Đo lại kênh trước khi thêm contact:** ACK chiếm 14% byte. 37% contact upload xong rồi bỏ trống phần còn lại.
  Chuyển transfer theo thứ tự cid tạo lệch nguồn (corr 0,93). Kiểm phân bố nguồn trước khi nói "đa dạng".
- Multi-contact M15: 8 đoạn/cycle, `tau_min` [1,5; 5,0], tỉ lệ urban theo client. Có code, nhưng hướng mới chưa cần.

## 7. Hạ tầng và vận hành

- **Kaggle:** 8 tài khoản × 2 phiên GPU = 16 slot; 2×T4 sm_75; image pin trong [knowledge/runtime.json](../../knowledge/runtime.json).
  - Push: `archive3:scripts/push_kaggle.py`. Pull: skill `kaggle-training-notebook/scripts/pull_kernel_output.py` (HTTP Range, tiếp tục được khi đứt). Không dùng `kaggle kernels output`.
  - Tiêu đề kernel ≤ 50 ký tự. Không sửa notebook đã PUSHED; mỗi attempt một snapshot.
  - Budget phiên lấy từ log thật, cộng dư ≥ 15%. Nạp checkpoint về CPU (`map_location='cpu'`).
  - Glob kết quả phải khớp đúng tên (`metrics_[0-9]*.json`). Kaggle không trùng bit giữa các account.
- **Máy local:** RAM 7,6 GiB, RTX 3050 4 GB ([knowledge/local-env.md](../../knowledge/local-env.md)).
  - Mỗi lần chỉ một job nặng, chạy qua `archive3:scripts/run_local_guarded.py`.
  - `conda run` nuốt stdout. Chờ tiến trình theo PID, không dùng `pgrep -f` (tự khớp chính lệnh chờ).
- **DVC/Git:**
  - Chỉ `dvc add` workspace đầy đủ. Workspace pull một phần mà add lại sẽ ghi đè manifest đủ.
  - `.git` có thể phình vì `tmp_obj_*` của lần push bị ngắt. Kiểm bằng `git count-objects -vH`.
  - Chỉ xóa local khi chủ dự án yêu cầu.
- **W&B:** key chỉ nhúng vào bản tạm khi push, không vào repo. Collector redact key trong log/notebook.

## 8. Code tái sử dụng

| thành phần | ở đâu | dùng lại thế nào |
|---|---|---|
| DAGSNet + registry kiến trúc (`dagsnet`, `mlp`, `mlp2`, `cnn1d`, `gru`) | `archive3:papers/build20/notebook/b20_s42_stackpz/proj/{model,pilot}.py` (`private_model`) | bộ kiến trúc khác loại cho R2 |
| 10 metric từ confusion | `.../proj/metrics.py` | giữ nguyên |
| Simulator contact | `.../proj/mobility.py` | giữ phần contact/rate/deadline, bỏ phần prototype item |
| Frame/fragment/deadline/resume exact | `.../proj/pilot.py` (`frame`, `transfer`, verify/resume) | tách phần kênh khỏi phần record |
| Fixture holdout B, prepare | `archive3:papers/build9_common/proj/prepare.py`, `scripts/m_fixtures.py` | chỉ khi dùng holdout nội bộ |
| Collector/verify theo đúng source đã chạy | `archive3:scripts/collect_b13.py` | mẫu cho collector mới |
| Driver FL đồng nhất 50 round, ckpt/verify/report | repo `odixe06/fl_dagsnet_veremi` (`fd_ids/papers/fd-ids-2025/proj/`, `nilm_fl`, `tinyproto`, `pfedes`) | **nền tốt nhất** cho code mới vì trùng protocol so sánh |

## 9. Số tham chiếu

Không đặt các cột khác protocol cạnh nhau để so.

**A. Test toàn cục, round 50, cùng protocol `fl_dagsnet_veremi`** (DAGSNet đồng nhất, 66 feature, không ràng buộc truyền, 1 seed):

| phương pháp | loại | 20c | 50c | 100c |
|---|---|---:|---:|---:|
| Centralized DAGSNet (round 5) | đối chứng trên | 0,8532 | | |
| NILM-FL proxy toàn cục w̄_r | học tương hỗ, trung bình proxy | 0,8228 | 0,7923 | 0,7568 |
| AFPHA (FedAvg + FedProx) | trung bình tham số | 0,7752 | 0,7162 | 0,6949 |
| FD-IDS (FedProx + KD) | trung bình tham số | 0,7655 | ≈0,68 (plateau) | ≈0,66 (plateau) |
| NILM-FL private w_s | học tương hỗ | 0,6963 | 0,6543 | 0,6111 |
| PerFed-SKD ω_m | cá nhân hóa | 0,6504 | 0,6901 | 0,6541 |
| TinyProto-FP | prototype thưa (HtFL) | 0,6212 | 0,4018 | 0,3435 |
| pFedES | proxy extractor chia sẻ (HtFL) | 0,4856 | 0,3652 | 0,2900 |
| MobiHet Build2 (prototype + mobility) | lịch sử | 0,5436 | 0,4448 | 0,3607 |

Nguồn: `fl_dagsnet_veremi/<method>/docs/report.md`. FD-IDS 50c/100c là trung bình 10 round cuối, cần đọc lại số round 50 trước khi trích.

**B. Holdout nội bộ archive (group GP), không cùng thang với A:**

| cấu hình | input | số | nguồn |
|---|---|---:|---|
| S13 off (gần local-only, recipe D3-AMP) c9–12 | 53 live (M4b) | 0,551/0,551 | Build13 |
| D3-AMP (record cache) c9–12 | 53 live | 0,662/0,647 | Build13 |
| B17 stack (record cache) c16 | 66 (M4b) | 0,7515/0,7424 | Build17 |
| B19 stackp (record cache) c16 | 126 (p4map) | 0,806/0,818 | Build19 |

**Thiếu:** chưa có **local-only** theo đúng protocol A. Đây là mốc dưới bắt buộc cho hướng mới ([đề xuất §5](../research/de-xuat-huong-moi.md#5-thứ-tự-và-kịch-bản-kiểm-chứng)).
