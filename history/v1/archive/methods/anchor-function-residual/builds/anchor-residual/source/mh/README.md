# Module riêng — v9-teacher-residual

Implementation thuộc build `v9-teacher-residual`, phương pháp `anchor-function-residual`.
Mỗi file giữ nguyên byte source đã chốt; import qua alias `mh/<module>.py` tại repository root.

| module | dependency mh |
|---|---|
| [v9_anchor.py](anchor_residual_teacher.py) | `pilot`, `teacher_base`, `v2_math`, `v2_methods` |
| [v9_run.py](anchor_residual_driver.py) | `teacher_driver`, `v6_run`, `v9_anchor` |

[Căn cứ/cơ chế](../../../../../../../docs/archive/methods/anchor-function-residual/design.md) · [Build/kết quả/artifact](../../../../../../../docs/archive/methods/anchor-function-residual/builds/anchor-residual/README.md) · [Package dùng chung và compatibility](../../../../../../../docs/operations/project-layout.md).

Không sửa file này để nối checkpoint cũ; thay thuật toán cần source/tag mới và preflight.
