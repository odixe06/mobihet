# Synthetic book khớp latent mean

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | S-a/DM; V5 mean controls |
| trạng thái | S-a tĩnh FAIL; V5 refreshed/trained vẫn thuaGaussiancontinuous, auditđãđủ. |

## Cơ chế FL

Tối ưu noise book theo mean latent DAGSNet; so encoder đóng băng/encoder private đang học và book refresh. Chỉ book synthetic đi qua RSU.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [pilot-new-fl.md](../../docs/research/pilot-new-fl.md)
- [pilot-v5-books.md](../../docs/research/pilot-v5-books.md)
- [pilot-new-fl.md](../../docs/research/probes/pilot-new-fl.md)
- [new_fl.py](../../mh/new_fl.py)
- [v5_books.py](../../mh/v5_books.py)
- [report.md](../../papers/new-fl/report.md)
- [report.md](../../papers/v5/report.md)
- [rebuild.md](../../papers/v5/rebuild.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
