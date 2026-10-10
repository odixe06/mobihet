# MobiHet — federated IDS cho V2X, model khác loại, ngân sách contact, không chia sẻ dữ liệu

Phát hiện tấn công/misbehavior trên tin nhắn V2X (VeReMi NextGen, 16 lớp). Xe học cộng tác qua RSU, và mọi phương pháp phải giữ đủ bốn yêu cầu:

1. **R1 — contact hữu hạn:** giao tiếp xe–RSU phải nằm trong ngân sách contact.
2. **R2 — khác kiến trúc:** mỗi xe được dùng model riêng, kiến trúc có thể khác nhau.
3. **R3 — không chia sẻ dữ liệu:** không mẫu dữ liệu thô hoặc record nào rời khỏi xe.
4. **R4 — cùng cấu hình để so sánh:** dùng 66 feature gốc, cùng protocol với `odixe06/fl_dagsnet_veremi`.

Repo này bắt đầu ngày 2026-10-10, sau khi hướng record cache (Build7–21) bị dừng vì vi phạm R3.
Toàn bộ lịch sử nằm ở `odixe06/mobihet-archive-b9-b21` (cùng `-b1-b6`, `-b7-b8`).

| bắt đầu từ | |
|---|---|
| [CONTEXT.md](CONTEXT.md) | trạng thái, quyết định hiện hành |
| [LESSONS.md](LESSONS.md) | bài học bắt buộc |
| [docs/](docs/README.md) | kiến thức kế thừa, đề xuất, protocol, vận hành |
| [knowledge/](knowledge/README.md) | sự thật về dữ liệu, kiến trúc DAGSNet, máy, hạ tầng (không phụ thuộc phương pháp) |

Dữ liệu nặng do DVC quản lý (remote `gdrive`, chung với archive): `dvc pull <target>.dvc`.
