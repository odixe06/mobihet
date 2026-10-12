# Teacher từ kernel soft target trên anchor chung

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | V5-H/K2b |
| trạng thái | Hai seed đã audit trước cleanup; small/medium teacher FAIL. |

## Cơ chế FL

Xe tính kernel sum theo lớp trên Gaussian anchors chung; SecureSum tạo bảng soft target; teacher DAGSNet train bảng, không classifier bảng triển khai.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [pilot-v5-anchors.md](../../docs/research/pilot-v5-anchors.md)
- [v3-knowledge-sources.md](../../docs/research/v3-knowledge-sources.md)
- [v5_anchor.py](../../mh/v5_anchor.py)
- [report.md](../../papers/v5/report.md)
- [rebuild.md](../../papers/v5/rebuild.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.

[Chẩn đoán target thực tế](diagnostics.md): restored4resumes/MD5, coverage/entropy/CEfloor, khôngsolecause.
