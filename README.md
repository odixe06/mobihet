# MobiHet — Federated IDS trong V2X

[Mục lục tổng quan](docs/README.md) · [Tất cả đề xuất và trạng thái](docs/proposals.md) · [Index từng file](docs/file-index.md).

Mở phiên: [CONTEXT](CONTEXT.md) → [LESSONS](LESSONS.md) → [bàn giao hiện tại](docs/operations/reorganization-2026-10-14.md).

| cần tìm | mở |
|---|---|
| Căn cứ, cơ chế, thành công/thất bại từng phương pháp | [Danh mục đề xuất](docs/proposals.md), [hồ sơ](docs/methods/README.md) |
| Source, probes, calibration, pilot, checkpoint của một build | [methods/](methods/README.md), README của build |
| Kết quả và đủ10metrics | `docs/methods/<method>/builds/<build>/results/`; [H1/H2 cuối](docs/methods/shared-baselines/builds/federated-baselines-full/results/latest-rounds.md), [Hybrid cuối](docs/methods/gaussian-proxy-hybrid/builds/hybrid-full/results/latest-rounds.md) |
| Lỗi kernel và quy tắc trước tạo notebook | [Sổ lỗi](docs/operations/notebook-errors.md), [CONTEXT](CONTEXT.md) |
| Bài học khiến train chưa tốt | [LESSONS](LESSONS.md), lessons riêng phương pháp |
| Scripts dùng chung / scripts riêng | [scripts/](scripts/README.md) / `methods/<method>/builds/<build>/scripts/` |
| Package chung / module riêng build | [core/](core/README.md) / [phân loại và đường dẫn](docs/operations/package-ownership.md) |
| Logs | `methods/<method>/builds/<build>/logs/`; log chung tại [logs/operations/](logs/operations/README.md) |
| Git/DVC, alias, restore | [Cấu trúc](docs/operations/project-layout.md), [backup/restore](docs/operations/backup.md) |
| Lặp lại quy trình bằng skill | [Codex](.agents/skills/federated-experiment-workflow/SKILL.md), [Claude Code](.claude/skills/federated-experiment-workflow/SKILL.md) |

DAGSNet-only hiện tại; R2 khác kiến trúc ở thiết kế. Không raw/perturbed local records rời client.
Heavy training trên Kaggle; W&B online bắt buộc. Source/notebook đã PUSHED giữ hash.
Khu làm việc: core/methods/scripts. Phần đã kết thúc: [archive](archive/README.md), [docs/archive](docs/archive/README.md).
Bài học: [docs/lessons](docs/lessons/README.md). mh/papers chỉ facade tương thích; tên V2–V10 chỉ giữ để truy vết.
