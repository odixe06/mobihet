# MobiHet — federated IDS cho V2X, model khác loại, ngân sách contact, không chia sẻ dữ liệu

Phát hiện tấn công/misbehavior trên tin nhắn V2X (VeReMi NextGen, 16 lớp). Xe học cộng tác qua RSU, và mọi phương pháp phải giữ đủ bốn yêu cầu:

1. **R1 — contact hữu hạn:** giao tiếp xe–RSU phải nằm trong ngân sách contact.
2. **R2 — khác kiến trúc:** mỗi xe được dùng model riêng, kiến trúc có thể khác nhau.
3. **R3 — không chia sẻ dữ liệu thật:** không raw record rời xe; synthetic khởi tạo từ nhiễu được phép.
4. **R4 — cùng cấu hình để so sánh:** dùng 66 feature gốc, cùng protocol với `odixe06/fl_dagsnet_veremi`.

Repo này bắt đầu ngày 2026-10-10, sau khi hướng record cache (Build7–21) bị dừng vì vi phạm R3.
Toàn bộ lịch sử nằm ở `odixe06/mobihet-archive-b9-b21` (cùng `-b1-b6`, `-b7-b8`).

| bắt đầu từ | |
|---|---|
| [CONTEXT.md](CONTEXT.md) | trạng thái, quyết định hiện hành |
| [LESSONS.md](LESSONS.md) | bài học bắt buộc |
| [docs/](docs/README.md) | kiến thức kế thừa, đề xuất, protocol, vận hành |
| [knowledge/](knowledge/README.md) | sự thật về dữ liệu, kiến trúc DAGSNet, máy, hạ tầng (không phụ thuộc phương pháp) |
| [Index toàn bộ file](docs/file-index.md) | tìm từng file, script, notebook, skill và vị trí artifact |

## Index theo nhu cầu

| cần tìm | mở |
|---|---|
| Trạng thái, quyết định, việc tiếp theo | [CONTEXT.md](CONTEXT.md) → [LESSONS.md](LESSONS.md) |
| Lập luận, protocol, kế hoạch và kết quả nghiên cứu | [docs/README.md](docs/README.md) |
| Dữ liệu, feature/class, scaler, DAGSNet, môi trường | [knowledge/README.md](knowledge/README.md) |
| FL H1–H5 và kênh P2 | [mh/pilot.py](mh/pilot.py), [mh/p2.py](mh/p2.py), [mh/channel.py](mh/channel.py) |
| Condensation/Fourier và driver có checkpoint | [mh/new_methods.py](mh/new_methods.py), [mh/new_fl.py](mh/new_fl.py) |
| Fixture và ngân sách contact | [build_pilot_fixture.py](scripts/build_pilot_fixture.py), [budget_tiers.py](scripts/budget_tiers.py), [probe_new_fl_contacts.py](scripts/probe_new_fl_contacts.py) |
| Dựng/push notebook P2, thu và verify | [gen_p2.py](scripts/kaggle/gen_p2.py), [push_nb.py](scripts/kaggle/push_nb.py), [collect_p2.py](scripts/kaggle/collect_p2.py), [verify_p2.py](scripts/verify_p2.py) |
| Dựng notebook S-a, thu và verify | [gen_new_fl.py](scripts/kaggle/gen_new_fl.py), [collect_new_fl.py](scripts/kaggle/collect_new_fl.py), [verify_new_fl.py](scripts/verify_new_fl.py) |
| Report P1/P2/S-a | [report_pilot.py](scripts/report_pilot.py), [report_p2.py](scripts/report_p2.py), [report_new_fl.py](scripts/report_new_fl.py) |
| Artifact, source/hash, kernel ID và run history | [P1](papers/pilot/report.md), [P2](papers/p2/report.md), [S-a ledger](papers/new-fl/rebuild.md); thư mục artifact trong [file-index](docs/file-index.md) |
| Local smoke và guard RAM/VRAM | [run_local_guarded.py](scripts/run_local_guarded.py), các `check_new_fl_*` trong [file-index](docs/file-index.md) |
| Kaggle, W&B, tài khoản, download/resume | [kaggle-training-notebook](.agents/skills/kaggle-training-notebook/SKILL.md), [W&B](.agents/skills/kaggle-training-notebook/references/wandb.md); yêu cầu hiện hành ở [LESSONS.md](LESSONS.md) |
| Backup/phục hồi Git–DVC và archive | [backup.md](docs/operations/backup.md), [repo-backup](.agents/skills/repo-backup/SKILL.md) |
| Từng file, kể cả helper/skill/notebook/log lưu trong Git | [docs/file-index.md](docs/file-index.md) |

Sau khi thêm/xóa/chuyển file hoặc tổng hợp tài liệu:

```bash
python3 scripts/update_file_index.py
rg --files
```

Index liệt kê file thuộc Git và file mới không bị ignore. Artifact nặng được định tuyến qua thư mục/manifest;
source/config đã chạy phải lấy từ ledger và snapshot, không suy từ code hiện tại.

Artifact có target `.dvc` do DVC quản lý (remote `gdrive`, chung với archive): `dvc pull <target>.dvc`.
Trạng thái backup artifact mới xem ledger; không suy đã DVC push từ việc có file local.
