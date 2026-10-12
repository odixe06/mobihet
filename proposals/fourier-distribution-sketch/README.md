# Fourier sketch phân phối lớp

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | S-a/KME |
| trạng thái | S-a paired FAIL; không mở S-b. |

## Cơ chế FL

SecureSum đặc trưng Fourier theo lớp; tối ưu synthetic bank từ noise để khớp sketch, xe train trên bank.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [de-xuat-bo-sung-2026-10-11.md](../../docs/research/de-xuat-bo-sung-2026-10-11.md)
- [pilot-new-fl.md](../../docs/research/pilot-new-fl.md)
- [pilot-new-fl.md](../../docs/research/probes/pilot-new-fl.md)
- [new_fl.py](../../mh/new_fl.py)
- [report.md](../../papers/new-fl/report.md)
- [rebuild.md](../../papers/new-fl/rebuild.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
