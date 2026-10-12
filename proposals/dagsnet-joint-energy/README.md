# Joint-energy bằng DAGSNet

[Danh mục](../README.md) · [Bài học](lessons.md)

Nhãn lịch sử: K4. Thiết kế chưa ready: bản không base measure bị bác bỏ về normalizability trước train.

## Cơ chế FL

DAGSNet tạo năng lượng chung feature/label; sinh input synthetic tại xe, tham số model được gộp.

## Tài liệu liên quan

- [v3-knowledge-sources.md](../../docs/research/v3-knowledge-sources.md)
- [probe_v3_energy.py](../../scripts/probe_v3_energy.py)
- [energy-probe.json](../../papers/v3/energy-probe.json)

Mọi learned model giữ DAGSNet; không raw record rời xe. Đây là thiết kế, chưa có bằng chứng F1/R1/R2 mới.
