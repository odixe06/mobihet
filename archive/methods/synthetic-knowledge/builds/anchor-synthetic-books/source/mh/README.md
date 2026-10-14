# Module riêng — v5-anchor-gradient-books

Implementation thuộc build `v5-anchor-gradient-books`, phương pháp `synthetic-knowledge`.
Mỗi file giữ nguyên byte source đã chốt; import qua alias `mh/<module>.py` tại repository root.

| module | dependency mh |
|---|---|
| [v5_anchor.py](kernel_anchor_targets.py) | `pilot`, `v2_math`, `v2_methods` |
| [v5_books.py](gradient_latent_books.py) | `pilot`, `v2_math`, `v2_methods` |
| [v5_run.py](synthetic_books_driver.py) | `v2_methods`, `v2_run`, `v5_anchor`, `v5_books` |

[Căn cứ/cơ chế](../../../../../../../docs/archive/methods/synthetic-knowledge/design.md) · [Build/kết quả/artifact](../../../../../../../docs/archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/README.md) · [Package dùng chung và compatibility](../../../../../../../docs/operations/package-ownership.md).

Không sửa file này để nối checkpoint cũ; thay thuật toán cần source/tag mới và preflight.
