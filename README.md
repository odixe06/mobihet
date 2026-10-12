# MobiHet — Federated IDS cho V2X

Xe học cộng tác qua RSU, contact hữu hạn, private model có thể khác kiến trúc, không chia sẻ record thật.
Input66feature VeReMi NextGen/16lớp; hiện kiểm learned modelsDAGSNet, R2 ở thiết kế.

| cần tìm | mở |
|---|---|
| Bắt đầu phiên / trạng thái | [CONTEXT](CONTEXT.md) → [LESSONS](LESSONS.md) |
| Đề xuất theo tên, cơ chế, code, kết quả, bài học | [proposals/](proposals/README.md) |
| Tổng số hướng / triển vọng | [24 hướng, 5 ưu tiên](proposals/status-summary.md) |
| Tổng hợp bằng chứng và việc cần kiểm tiếp | [evidence-summary](proposals/evidence-summary.md) |
| Từng file toàn dự án | [file-index](docs/file-index.md) |
| Protocol, hợp đồng pilot và vận hành | [docs/](docs/README.md) |
| Dữ liệu, scaler, DAGSNet, hạ tầng | [knowledge/](knowledge/README.md) |
| Kernel/output đang collect | [live](docs/operations/live-2026-10-12.md) |
| Artifact đã backup/xóa / khôi phục chọn lọc | [backup](docs/operations/backup.md) |

## Cấu trúc

```text
proposals/<tên-phương-pháp>/  README, bài học, thiết kế mới và kết quả theo phương pháp
docs/research/              protocol, manifest đã khóa, probe/hợp đồng các đợt
docs/operations/            quyết định, collect, restore, handoff
knowledge/                  facts dữ liệu, kiến trúc, máy
mh/                         source training đã chạy, giữ tên/path/hash
scripts/                    generator, probe, collector, verifier, report
papers/<đợt>/               notebook, report, ledger, DVC manifests, artifact
```

Mã H1/V2/V3/S-a là nhãn lịch sử; tra [proposals](proposals/README.md) bằng tên phương pháp.
Source/notebook/manifest/artifact giữ đường dẫn gốc để audit/resume/DVC.
Artifact bị ignore hoặc đã xóa local vẫn được chỉ rõ vị trí/manifest trong index.

Repo bắt đầu2026-10-10, thay hướng record-cache vi phạm yêu cầu không chia sẻ data.
Lịch sử: `odixe06/mobihet-archive-b9-b21` cùng archive b1-b6/b7-b8; phục hồi ở hồ sơ backup.
Không tự Git/DVC push hoặc cleanup; notebook mới bắt buộc W&B online.

Artifact đãbackup / lệnhpull: [chỉ mục artifact](docs/operations/artifact-index.md).
