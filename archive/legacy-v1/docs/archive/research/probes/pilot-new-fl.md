# Pilot S-a — Fourier sketch và condensation tĩnh

Hoàn tất, pull và verify ngày2026-10-11: **12run ×20round**, hai seed42/43 trên hai kernel private2×T4.
Sáu arm cùng fixture1.000.369dòng train,20private DAGSNet; holdout nội bộ `group`/`old`,65.536dòng/tập.
Local chỉ math/smoke/resume/report. Test20% chưa mở; R2 chỉ đánh giá thiết kế.

**KME và DM đều trượt cổng đã khóa.** Gaussian diag là mốc tốt nhất trong sáu arm đợt này.
S-a không giới hạn kênh; kết quả chưa chứng minh R1. Không mở S-b/full-scale cho hai adaptation này.

Nguồn số duy nhất: [report](../../methods/synthetic-distribution-matching/builds/static-synthetic-books/results/report.md),
[gates.json](../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/probes/gates.json),
[metrics-all.csv](../../methods/synthetic-distribution-matching/builds/static-synthetic-books/results/metrics-all.csv),
[per-class-final.csv](../../methods/synthetic-distribution-matching/builds/static-synthetic-books/results/per-class-final.csv).
Source/config/fixture/kernel/quota và mọi attempt: [rebuild](../../methods/synthetic-distribution-matching/builds/static-synthetic-books/results/rebuild.md).
Cổng khóa trước chạy: [pilot-new-fl.md](../pilot-new-fl.md).

## Bằng chứng và quyết định

- **KME:** hơn bank Gaussian-target chỉ +0,0019/+0,0024 macro-F1; thua Gaussian diag −0,0412/−0,0442.
  Cổng yêu cầu hơn `max(diag,kme_gauss)` ít nhất0,015 cả hai seed. Không qua.
- **DM:** so codebook chưa tối ưu −0,0002/+0,0036; chưa đạt0,01 cả hai seed.
  Gain so local của seed42 cũng chưa đạt0,03. Không qua.
- `head` giữ cùng chiều: hai nhánh mới vẫn thua diag. Chuyển quy tắc prior không giải thích hết khoảng cách.
- positionMirroring: KME so đối chứng −0,0005/+0,0001; DM +0,0019/−0,0033.
  Chưa sửa được lớp từng cho thấy Gaussian thiếu quan hệ phi tuyến.
- Benign guard không phải nguyên nhân chính trượt cổng. Synthesis fit giảm mạnh ở cả hai seed,
  nhưng lợi phân loại của bước tối ưu rất nhỏ. Đây không phải bằng chứng toàn bộ họ FedDM/DP-MERF thất bại.

## Chẩn đoán có giới hạn

KME dùng128điểm/lớp cố định; DM4điểm/lớp/producer, một encoder DAGSNet ngẫu nhiên đóng băng.
Loss train tổng r20 gần0,005–0,007, phù hợp giả thuyết private học thuộc bank tĩnh.
Chưa cô lập bank khỏi optimizer; DM còn khác diag về mixture nguồn và loại book của chính xe.
Không gán nguyên nhân duy nhất cho bank hoặc encoder.

Loss được lưu là CE real + CE synthetic. Tổng thấp cho thấy cả hai nhỏ;
tổng cao của diag chưa chứng minh CE real cao hoặc chống học thuộc thành công.
Chưa lưu phép đo Fourier độc lập/nearest-neighbor; fit trên target dùng tối ưu chưa chứng minh fidelity hay DP.

Stat/Fourier được tính tại client rồi gộp; codebook bắt đầu từ noise theo moment của chính client.
Không chọn record thật làm init/coreset/auxiliary batch của client khác.
RSU được đọc synthetic theo ngoại lệ đã chốt. Đây là simulation, chưa triển khai cryptographic secure aggregation hoặc DP.

## Kênh và bước còn lại

[Contact probe](../../../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/probes/contact-probe.json) chỉ mô phỏng kích thước, cursor và availability B0/B2;
không train với payload đã nhận. Không lấy probe này làm F1 dưới deadline thật.
S-b cần codec/payload/prior trả phí thực; hiện đóng vì S-a trượt cổng.

S1 Nyström covariance + object từng lớp vẫn là **thiết kế chưa pilot**, mở rộng H2/H6 để giảm upload,
không coi là họ FL mới hoặc hướng đã thắng. Xem [đề xuất S1](../de-xuat-bo-sung-2026-10-11.md#2-s1--sketch-covariance-cộng-được--phát-từng-lớp).
Không tune/kéo dài hai run tĩnh để cứu điểm. Nếu mở cơ chế khác, khóa đối chứng/cổng riêng trước chạy.

Notebook tạo tiếp theo bắt buộc W&B API key+online trước train, theo [LESSONS](../../../../LESSONS.md).
Hai notebook v1 đã chạy trước quyết định này giữ nguyên source. Artifact mới giữ local/private Kaggle;
Git/DVC chỉ thực hiện theo yêu cầu rõ của chủ dự án.
