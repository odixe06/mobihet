# Module riêng — v7-rank4-rank8

Implementation thuộc build `v7-rank4-rank8`, phương pháp `covariance-nystrom`.
Mỗi file giữ nguyên byte source đã chốt; import qua alias `mh/<module>.py` tại repository root.

| module | dependency mh |
|---|---|
| [v7_run.py](covariance_sketch_driver.py) | `v2_run`, `v6_run`, `v7_nystrom` |

[Căn cứ/cơ chế](../../../../../../../docs/archive/methods/covariance-nystrom/design.md) · [Build/kết quả/artifact](../../../../../../../docs/archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/README.md) · [Package dùng chung và compatibility](../../../../../../../docs/operations/package-ownership.md).

Không sửa file này để nối checkpoint cũ; thay thuật toán cần source/tag mới và preflight.
