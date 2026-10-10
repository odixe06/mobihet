# LESSONS — điều bắt buộc nhớ khi mở phiên mới

Đọc ngay sau [CONTEXT.md](CONTEXT.md). Phần lớn bài học kế thừa từ archive 3 (`archive3:LESSONS.md`, 2026-10-10), đã lọc theo hướng mới.
Chi tiết và số gốc: [kiến thức kế thừa](docs/legacy/kien-thuc-ke-thua.md). Bài học bị bác bỏ thì sửa tại đây, không xóa im lặng.

## 1. Chủ dự án và cách làm việc

- Trả lời tiếng Việt, kiểu caveman (CLAUDE.md). Chưa chắc thì **hỏi trước**. Việc phải làm bằng tay thì liệt kê từng bước.
- Đề xuất phương pháp phải có: **bằng chứng vấn đề → cơ chế → dấu hiệu kiểm chứng có thể sai**. Không thấy dấu hiệu thì dừng hướng đó.
- Phương án mới phải kiểm nghiệm ngắn để xác định khả thi. Bất hợp lý thì chỉnh từ nguyên nhân hoặc bỏ; **không kéo dài run để tìm điểm thắng**.
- Làm tới mốc nào cập nhật docs tới đó. Trước khi compact phải lưu kernel ID, source/config, kết quả đã verify, quota và bước còn thiếu.
- Không commit/push git, không backup, không xóa artifact khi chưa được đồng ý trong phiên.
- Kaggle: agent push trực tiếp sau preflight. Không ngại quota cho kịch bản ngắn; run dài hoặc nhiều client thì hỏi trước.
- **2026-10-10:**
  - R3 cấm mọi dạng gửi mẫu `(x, y)`, kể cả lên RSU.
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
- **KD/teacher:** teacher peer (Build7) và ridge teacher RSU (S13) chỉ ngang làm mượt theo lớp.
  Mọi teacher cần cổng chất lượng + đối chứng class-mean.
- **Khác kiến trúc:** chỉnh head không cứu được backbone yếu (M2). Tri thức phải tác động toàn model.
- **Nút thắt:** ranh giới quyết định đa receiver. Xe sai 4% trên receiver của mình, 45% toàn cục.
  32 mẫu thật/lớp từ client khác ≈ 70% lợi tối đa (M1, chỉ còn là đối chứng trên).
- **Chia sẻ model đi được trên dữ liệu này:** FedAvg/FedProx 0,77, NILM-FL proxy 0,82 / private 0,70 (test 20c, `fl_dagsnet_veremi`).
- **Công thức local có lợi:** `gp` (+0,025); EMA + head triển khai fit riêng mỗi cycle; tính lại BN local cho model EMA (lợi chuyển tiếp, giảm dần).
  EMA trung bình cả head thì hỏng.
- Trôi do học thuộc cục bộ có ở mọi cấu hình full data (CE 0,01–0,03); LR cosine chặn đà trôi.

## 5. Kênh truyền

- Ngân sách stress archive (`contact_scale` 0,01, `rate_scale` 0,2, cap 32 KiB) ≈ 4,5–4,7 kB/contact. Đó là knob, không phải số đo vật lý; bỏ scale ≈ 2,2 MB/contact.
- Kế toán byte: header, ACK, retransmission. Deadline cắt frame thì trả phí nhưng không commit.
  Không có gì miễn phí. Chỉ kích hoạt snapshot coherent.
- Đo idle/ACK/lệch nguồn của kênh hiện có trước khi thêm contact (ACK 14%, 37% contact bỏ trống, lệch cid corr 0,93).

## 6. Vận hành

- Local: RAM 7,6 GiB, RTX 3050 4 GB. **Mỗi lần một job nặng**, chạy qua `run_local_guarded.py`.
  `conda run` nuốt stdout. Chờ theo PID, không `pgrep -f` (tự khớp chính lệnh chờ).
- Kaggle:
  - 2×T4, image pin. Tiêu đề ≤ 50 ký tự. Không sửa notebook đã PUSHED.
  - Budget phiên = log thật + dư ≥ 15%. Nạp checkpoint về CPU.
  - Pull bằng `pull_kernel_output.py` (Range resume). Collector cần cả `python` lẫn `kaggle` trong PATH env `nckh`.
  - Kaggle không trùng bit giữa các account. Preflight phải chạy vòng arm của driver với đúng số cycle production (lỗi E14: arm 2 chạy 0 cycle).
- DVC:
  - Chỉ `dvc add` workspace đầy đủ.
  - `.git` phình vì `tmp_obj_*` sau push bị ngắt: kiểm `git count-objects -vH`.
  - Chỉ xóa local khi chủ dự án yêu cầu; không chờ cloud sync, đủ khi object đã lên G:.
