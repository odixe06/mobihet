# EMA backbone + local BN + head triển khai

**Cải tiến recipe H1/H2, không đếm họ FL mới.** Chủ dự án chốt fresh pilot 2×2 và own controls;
không sửa continuation đang chạy.
[Dự báo/giới hạn lịch sử](../../docs/research/performance-feasibility-2026-10-13.md).

Dự báo vừa cho ổn định cuối run, chưa cơ sở hứa tăng lớn hơn Gaussian/DML.
Đề nghị half-life2round trên backbone, loại toàn `head` khỏi EMA.
Deployment DAGSNet: BN calibrate local, backbone đóng băng, head fit16bước riêng.
Fit chỉ own local/synthetic hợp lệ nếu nền H2, không peer raw cache lịch sử.
Fit dài từng overfit; không tăng bước theo metric.

Mỗi nền: local/plain, local/EMA, FL/plain, FL/EMA; own/plain, own/EMA.
Interaction=(FL_EMA−FL_plain)−(local_EMA−local_plain); FL−own cần còn dương.
Không gọi recipe local là lợi kênh FL. gp/benign guards khóa trước pilot;
resume cần EMA parameters, deployment head/BN và RNG/head-fit schedule.

**Đã chốt 13/10:** kiểm cả Gaussian và DML, cùng local/own controls. Tại r20 từng seed42/43:
FL+EMA hơn FL plain≥0,010; hơn own+EMA≥0,015; benign giảm≤0,005;
cost≤2× plain cùng nền. Interaction báo riêng, không bắt buộc dương để gọi recipe có lợi.
Half-life2/backbone-only/BN4096/head16/gp cũ chính như trên đã chốt;
source và notebook chưa triển khai, chưa kết quả.
