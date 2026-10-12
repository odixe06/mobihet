# Synthetic book khớp gradient DAGSNet

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | V5-G/J/K3 |
| trạng thái | IPC1/4 đủhai seed đãaudit; qualityFAIL, cost2×PASS. |

## Cơ chế FL

Noise book IPC1/4; gradient matching so latent-mean, frozen/trained encoder;80 bước tổng round-robin lớp, refresh5round, giới hạn2×baseline.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [pilot-v5-books.md](../../docs/research/pilot-v5-books.md)
- [v3-knowledge-sources.md](../../docs/research/v3-knowledge-sources.md)
- [v5_books.py](../../mh/v5_books.py)
- [report.md](../../papers/v5/report.md)
- [rebuild.md](../../papers/v5/rebuild.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
