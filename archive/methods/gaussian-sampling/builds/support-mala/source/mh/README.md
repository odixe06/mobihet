# Module riêng — v4-support-mala

Implementation thuộc build `v4-support-mala`, phương pháp `gaussian-sampling`.
Mỗi file giữ nguyên byte source đã chốt; import qua alias `mh/<module>.py` tại repository root.

| module | dependency mh |
|---|---|
| [v4_mala.py](mala_transport.py) | `pilot`, `v2_methods` |
| [v4_run.py](support_mala_driver.py) | `v2_methods`, `v2_run`, `v4_mala`, `v4_support` |
| [v4_support.py](support_conditioned_gaussian.py) | `pilot`, `v2_math`, `v2_methods`, `v4_schema` |

[Căn cứ/cơ chế](../../../../../../../docs/archive/methods/gaussian-sampling/design.md) · [Build/kết quả/artifact](../../../../../../../docs/archive/methods/gaussian-sampling/builds/support-mala/README.md) · [Package dùng chung và compatibility](../../../../../../../docs/operations/package-ownership.md).

Không sửa file này để nối checkpoint cũ; thay thuật toán cần source/tag mới và preflight.
