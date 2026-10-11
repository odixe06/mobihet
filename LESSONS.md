# LESSONS — điều bắt buộc nhớ khi mở phiên mới

Đọc ngay sau [CONTEXT.md](CONTEXT.md). Phần lớn bài học kế thừa từ archive 3 (`archive3:LESSONS.md`, 2026-10-10), đã lọc theo hướng mới.
Chi tiết và số gốc: [kiến thức kế thừa](docs/legacy/kien-thuc-ke-thua.md). Bài học bị bác bỏ thì sửa tại đây, không xóa im lặng.

## 1. Chủ dự án và cách làm việc

- Trả lời tiếng Việt, kiểu caveman (CLAUDE.md). Chưa chắc thì **hỏi trước**. Việc phải làm bằng tay thì liệt kê từng bước.
- Đề xuất phương pháp phải có: **bằng chứng vấn đề → cơ chế → dấu hiệu kiểm chứng có thể sai**. Không thấy dấu hiệu thì dừng hướng đó.
- Phương án mới phải kiểm nghiệm ngắn để xác định khả thi. Bất hợp lý thì chỉnh từ nguyên nhân hoặc bỏ; **không kéo dài run để tìm điểm thắng**.
- Làm tới mốc nào cập nhật docs tới đó. Trước khi compact phải lưu kernel ID, source/config, kết quả đã verify, quota và bước còn thiếu.
- **2026-10-11 (chủ dự án): tổng hợp tài liệu phải có index toàn dự án.** [README](README.md) định tuyến theo nhu cầu;
  [docs/file-index.md](docs/file-index.md) liệt kê từng file và vị trí artifact. Sau thêm/xóa/chuyển file, chạy
  `python3 scripts/update_file_index.py`; kiểm link trước bàn giao để phiên sau tìm đủ code/docs/notebook/skill.
- Không backup, không xóa artifact khi chưa được đồng ý trong phiên (commit/push: xem dòng 2026-10-11 bên dưới).
- Kaggle: agent push trực tiếp sau preflight. Không ngại quota cho kịch bản ngắn; run dài hoặc nhiều client thì hỏi trước.
- **2026-10-11 (chủ dự án): mọi notebook huấn luyện tạo mới bắt buộc tích hợp Weights & Biases (`wandb`) API key
  và theo dõi online trực tiếp.** Khởi tạo/kết nối W&B trước train, bật Internet, log tiến trình mỗi round
  (loss, LR, bước applied/skipped, thời gian; đủ10metric khi đánh giá). Thiếu key/kết nối ⇒ fail trước train,
  không âm thầm offline/disabled. Key chỉ nhúng vào bản push private tạm dưới `papers/<method>/private_push/`
  (đã gitignore) theo workflow đã cho phép, không vào Git/log/chat; kiểm `git check-ignore` trước nhúng key.
  Notebook đã PUSHED giữ source; không sửa/rerun chỉ để thêm W&B. Quy định áp dụng từ notebook mới tiếp theo.
- **2026-10-11: tiến trình nặng (train/pilot nhiều arm, protocol đầy đủ) chạy trên Kaggle, không chạy ở máy local.**
  Local chỉ để smoke test, dựng fixture nhỏ, report, collect.
- 2026-10-11: được commit + push repo này ở các mốc. Không thêm attribution Claude vào commit.
- 2026-10-11: mọi client dùng DAGSNet, **kể cả proxy** (H1, H1+H2): cấu hình đồng nhất cho toàn bộ client. Chưa chạy đa mô hình.
- 2026-10-11 (phiên tiếp): được đề xuất ngoài H1–H6. RSU được đọc **codebook synthetic từ nhiễu**; cấm raw-init/record thật.
  Đây là ngoại lệ Q4 cho synthetic, không phải tự động có secure aggregation hay DP. DAGSNet/local-smoke được xác nhận lại.
- **2026-10-10:**
  - R3 cấm mọi dạng gửi mẫu `(x, y)`, kể cả lên RSU. **Không client nào thấy local data của client khác** (tiêu chuẩn FL, chủ dự án nhắc lại),
    kể cả trong arm đối chứng/upper reference của thí nghiệm. Số record cache cũ chỉ còn là lịch sử, không chạy lại.
  - Input 66 feature gốc, cùng protocol `fl_dagsnet_veremi`.
  - Được thử mọi dạng tri thức không phải dữ liệu thô, nhưng phải tính đến kết quả cũ.

## 2. Dữ liệu

- `train/` và partition FL **đã z-score**, `test/` **raw**: áp `scaler.json` đúng một lần. Kiểm: mean(`f_snd_spd`) test raw = 7,4493.
- Không chia lại dataset. Partition α=0,5 (20/50/100 client) là lệch tỉ lệ, không làm thiếu lớp.
- File client **sắp theo lớp**: luôn lấy mẫu ngẫu nhiên có seed. Khóa phiên: `(scenario, source_run, receiver_id, sender_alias)`.
- Shortcut:
  - Sybil có `f_first_in_session` và cụm feature điền 0.
  - Benign lấy từ run Sybil.
  - Vân tay receiver (9 feature noise/profile, η² 0,8–0,95) giúp đoán nhãn 80–90% trong một client.
- Trần do bộ sinh dữ liệu: timeDelay không để lại dấu hiệu trong nội dung tin (centralized F1 0,23). Đọc generator trước khi đổ lỗi cho phương pháp.

## 3. Đánh giá

- Holdout chia ngẫu nhiên theo dòng lạc quan +0,09…+0,12. Dùng temporal/receiver-group. Test chỉ để báo theo protocol khóa, không dùng chọn HP.
- Khóa **một** quy tắc quyết định chính (`gp`) và cổng **trước** khi chạy. Cặp seed, một yếu tố/arm, đủ 10 metric, F1 theo lớp, benign F1.
- Thành phần mượn mà chưa đo riêng thì phải đo trước hoặc thiết kế 2×2 (M15: interleave che mất lợi của cơ chế mới).
- Pilot → full scale: lợi thường còn khoảng một nửa, có khi đổi dấu. Slope ngắn không dự báo trôi. Lợi kiểu thêm tri thức tích lũy theo cycle.
- Probe offline chỉ là điều kiện cần (N3 offline + / online −). Thống kê tri thức phải tính trên dữ liệu chưa train.
- Kernel COMPLETE ≠ artifact hợp lệ: pull rồi verify bằng **đúng source đã chạy**.

## 4. Phương pháp — điều đã biết

- **Prototype trung bình lớp** (FedProto-like): so với không trao đổi +0,045, nhưng mọi cách sửa (cluster, InfoNCE, soft-label, tách bank, hard proto, projection head) chỉ ±0,006.
  Trung bình lớp không chở ranh giới. Đừng đầu tư thêm vào biến đổi trung bình lớp.
- **Thống kê latent bị cũ đi khi backbone còn học:** CCVR-like N3 offline +0,013, online −0,014/−0,010.
  Dùng chúng cho head triển khai trên snapshot đóng băng, không làm loss liên tục.
- **Thống kê lớp bậc hai ở không gian input mang được tri thức đa client mà không lộ dòng nào** (H2-T0-lite 2026-10-10, MLP, 4 client × 2 seed):
  Gaussian full-cov gộp +0,10, diag +0,05, điểm trung bình −0,02 so với replay cân bằng local. Thông tin nằm ở hiệp phương sai, không ở tâm.
  Input không trôi theo representation, nên tránh được lỗi của N3 latent. Chưa xác nhận trên DAGSNet/ngân sách/DP.
  [Kết quả](docs/research/probes/h2-input-stats.md).
- **Pilot P1 (2026-10-11, DAGSNet 20c, 2,5% dữ liệu, 20 round; [kết quả](docs/research/probes/pilot-p1.md)):**
  - H2 xác nhận trên DAGSNet: Gaussian full gộp +0,11; Gaussian của **chính xe** chỉ +0,024. ⇒ lợi là tri thức quần thể, không phải làm mượt.
    Lợi tăng theo cấu trúc bậc hai giữ được: diag +0,08 (13 kB một lần), lr4 +0,09, full +0,11.
  - H1 (DML) tăng theo sức chứa proxy: MLP64 +0,05, MLP256 +0,12, DAGSNet +0,16. Proxy toàn cục còn cao hơn private (0,57 / 0,68 / 0,77).
  - **Tri thức chỉ chạm head thì vô dụng khi backbone học thuộc:** LG +0,02, FedGen +0,02, FedGH −0,02.
    Dấu hiệu nhận biết: loss local r20 vẫn ≈ 0,01 như `local`.
  - **Teacher yếu thua nhãn miễn phí:** DS-FL trên điểm Gaussian (+0,075) thua chính nhãn lớp của Gaussian (+0,107).
    Ensemble của 20 model local ~0,46 làm nhãn nhiễu đi.
  - **`gp` sửa quá tay khi model private bị chưng cất về model toàn cục** (DML): `head` tăng đều nhưng `gp` rơi từ r15, benign F1 −0,17.
    `gp` giả định prior của private là π_i. Phương pháp kéo về toàn cục phải ước prior hiệu dụng, hoặc báo cả `head`.
  - positionMirroring **không** chạm trần thông tin: FedAvg +0,26. Gaussian (H2) và DML private không học được lớp này (+0,01).
    Chỉ timeDelay có trần (centralized 0,23).
- **KD/teacher:** teacher peer (Build7) và ridge teacher RSU (S13) chỉ ngang làm mượt theo lớp.
  Mọi teacher cần cổng chất lượng + đối chứng class-mean.
- **S-a seed43 đã verify (2026-10-11):** fit Fourier giảm0,0129→4,23e-7 nhưng KME chỉ hơn
  Gaussian-target bank +0,0024, thua Gaussian diag liên tục −0,0442. DM fit mean20producer giảm mạnh
  nhưng hơn codebook chưa tối ưu chỉ +0,0036. Khớp mục tiêu synthesis không chứng minh tri thức có ích.
  Chưa suy nguyên nhân riêng bank/encoder từ đối chứng hiện có; chưa phủ định họ FedDM/DP-MERF.
  Chờ seed42 để ghi kết quả cặp, không tune/kéo dài run cứu cổng. [Thiết lập](docs/research/pilot-new-fl.md).
- **Khác kiến trúc:** chỉnh head không cứu được backbone yếu (M2). Tri thức phải tác động toàn model.
- **Nút thắt:** ranh giới quyết định đa receiver. Xe sai 4% trên receiver của mình, 45% toàn cục.
  32 mẫu thật/lớp từ client khác ≈ 70% lợi tối đa (M1, số lịch sử; dạng này vi phạm R3, không chạy lại).
- **Chia sẻ model đi được trên dữ liệu này:** FedAvg/FedProx 0,77, NILM-FL proxy 0,82 / private 0,70 (test 20c, `fl_dagsnet_veremi`).
- **Công thức local có lợi:** `gp` (+0,025); EMA + head triển khai fit riêng mỗi cycle; tính lại BN local cho model EMA (lợi chuyển tiếp, giảm dần).
  EMA trung bình cả head thì hỏng.
- Trôi do học thuộc cục bộ có ở mọi cấu hình full data (CE 0,01–0,03); LR cosine chặn đà trôi.
- Loss train FL **tăng** ở cuối lịch cosine (FedAvg 0,28 → 0,45, DML) không phải trôi.
  LR ~1e-5 thì model vừa tải xuống hầu như không khớp thêm vào dữ liệu local, nên loss trung bình ≈ loss của model tải xuống. Đọc điểm đánh giá, không đọc loss.

## 5. Kênh truyền

- **P2 audit 2026-10-11:** lr4 giảm download nhưng upload vẫn145.792B; B0 không kích hoạt tri thức trong20round.
  DML B0 không nhận proxy mới vẫn hơn local do mutual learning tại xe; không gọi đó là lợi FL.
  Prior miễn phí và byte giả định fp16 chưa lượng tử hóa làm cổng metric chưa đủ chứng minh R1.
  Pilot phải lưu prediction/confusion từng client và weights để kiểm lại mean metric; confusion tổng không đủ.
  [Bằng chứng và giới hạn](docs/research/probes/pilot-p2.md).

- Ngân sách stress archive (`contact_scale` 0,01, `rate_scale` 0,2, cap 32 KiB) ≈ 4,5–4,7 kB/contact. Đó là knob, không phải số đo vật lý; bỏ scale ≈ 2,2 MB/contact.
- Kế toán byte: header, ACK, retransmission. Deadline cắt frame thì trả phí nhưng không commit.
  Không có gì miễn phí. Chỉ kích hoạt snapshot coherent.
- Đo idle/ACK/lệch nguồn của kênh hiện có trước khi thêm contact (ACK 14%, 37% contact bỏ trống, lệch cid corr 0,93).

## 6. Vận hành

- Local: RAM 7,6 GiB, RTX 3050 4 GB. Job nặng **không** chạy local (chủ dự án 2026-10-11). Job vừa (smoke, fixture) chạy qua `run_local_guarded.py`.
- Đọc parquet bằng `pyarrow.dataset` phải tắt readahead (`batch_readahead=1, fragment_readahead=1`), nếu không RSS lên 4,6 GB.
  `centralized/test/` có lẫn `*.stats.json`, nên glob `*.parquet`.
  `conda run` nuốt stdout. Chờ theo PID, không `pgrep -f` (tự khớp chính lệnh chờ).
- RTX 3050: DAGSNet train ~20k dòng/s một job (bị giới hạn bởi launch). **Hai job song song** cho tổng ~30k dòng/s (×1,5), peak RAM ~4,5 GB.
  Sửa module mà hàng đợi đang dùng: chép ra bản sao, smoke test, rồi `mv` thay nguyên tử. Job sau đọc file mới, job đang chạy không bị ảnh hưởng.
- `xargs ... bash -c '...' arg0 arg1`: `$0` là arg đầu, nên `shift` chỉ bỏ `$1`. Đọc `$?` **trước** `$(date)` trong cùng lệnh echo.
- Kaggle:
  - 2×T4, image pin. Tiêu đề ≤ 50 ký tự. Không sửa notebook đã PUSHED.
  - Budget phiên = log thật + dư ≥ 15%. Nạp checkpoint về CPU.
  - Pull bằng `pull_kernel_output.py` (Range resume). Collector cần cả `python` lẫn `kaggle` trong PATH env `nckh`.
  - Kaggle không trùng bit giữa các account. Preflight phải chạy vòng arm của driver với đúng số cycle production (lỗi E14: arm 2 chạy 0 cycle).
- DVC:
  - Chỉ `dvc add` workspace đầy đủ.
  - `.git` phình vì `tmp_obj_*` sau push bị ngắt: kiểm `git count-objects -vH`.
  - Chỉ xóa local khi chủ dự án yêu cầu; không chờ cloud sync, đủ khi object đã lên G:.
