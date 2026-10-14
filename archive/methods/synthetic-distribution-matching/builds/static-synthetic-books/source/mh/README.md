# Module riêng — sa-static-books

Implementation thuộc build `sa-static-books`, phương pháp `synthetic-distribution-matching`.
Mỗi file giữ nguyên byte source đã chốt; import qua alias `mh/<module>.py` tại repository root.

| module | dependency mh |
|---|---|
| [new_methods.py](distribution_matching.py) | `pilot` |

[Căn cứ/cơ chế](../../../../../../../docs/archive/methods/synthetic-distribution-matching/design.md) · [Build/kết quả/artifact](../../../../../../../docs/archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/README.md) · [Package dùng chung và compatibility](../../../../../../../docs/operations/package-ownership.md).

Không sửa file này để nối checkpoint cũ; thay thuật toán cần source/tag mới và preflight.
