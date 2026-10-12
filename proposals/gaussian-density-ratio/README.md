# DAGSNet học tỷ số mật độ cho Gaussian

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | V2-C |
| trạng thái | Hai seed FAIL; không contact. |

## Cơ chế FL

DAGSNet NCE real làm trơn local tau0.05 so Gaussian; weight synthetic theo ratio clip10, chuẩn hóa lớp. Tham số tỷ số gộp, không gửi record.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [c-density-ratio.md](../../docs/research/de-xuat-v2/c-density-ratio.md)
- [pilot-v2.md](../../docs/research/pilot-v2.md)
- [pilot-v2.md](../../docs/research/probes/pilot-v2.md)
- `mh/v2_density.py` — vị trí lưu; kiểm index khi khôi phục artifact.
- [report.md](../../papers/v2/report.md)
- [rebuild.md](../../papers/v2/rebuild.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
