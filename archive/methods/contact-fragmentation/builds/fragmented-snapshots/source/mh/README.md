# Module riêng — p2-deadline

Implementation thuộc build `p2-deadline`, phương pháp `contact-fragmentation`.
Mỗi file giữ nguyên byte source đã chốt; import qua alias `mh/<module>.py` tại repository root.

| module | dependency mh |
|---|---|
| [p2.py](fragmented_snapshot_driver.py) | `channel`, `pilot` |

[Căn cứ/cơ chế](../../../../../../../docs/archive/methods/contact-fragmentation/design.md) · [Build/kết quả/artifact](../../../../../../../docs/archive/methods/contact-fragmentation/builds/fragmented-snapshots/README.md) · [Package dùng chung và compatibility](../../../../../../../docs/operations/package-ownership.md).

Không sửa file này để nối checkpoint cũ; thay thuật toán cần source/tag mới và preflight.
