# Gaussian được vận chuyển bằng MALA

[Danh mục phương pháp](../README.md) · [Bài học](lessons.md)

| thuộc tính | nội dung |
|---|---|
| nhãn lịch sử | V4-M |
| trạng thái | Hai seed đãaudit; pairedFAIL. |

## Cơ chế FL

DAGSNet ratio gộp hướng dẫn MALA tại xe; Gaussian base, eta0.01,20burn-in rồi một transition/batch. Không gửi chain/record.

Raw record/record perturb không rời xe. Thống kê được gộp; book tối ưu từ noise thuộc ngoại lệ synthetic đã cho phép. Mọi learned model đang kiểm đều DAGSNet; khác kiến trúc chỉ ở thiết kế.

## Thiết kế, kết quả, code và artifact

- [v3-knowledge-sources.md](../../docs/research/v3-knowledge-sources.md)
- [pilot-v4-mala.md](../../docs/research/pilot-v4-mala.md)
- [pilot-v4.md](../../docs/research/probes/pilot-v4.md)
- [v4_mala.py](../../mh/v4_mala.py)
- [report.md](../../papers/v4/report.md)
- [rebuild.md](../../papers/v4/rebuild.md)

Artifact nặng có thể đã xóa local sau backup. Xem [hồ sơ restore](../../docs/operations/backup.md); manifest còn tại `papers/`. COMPLETE không thay marker audit.
