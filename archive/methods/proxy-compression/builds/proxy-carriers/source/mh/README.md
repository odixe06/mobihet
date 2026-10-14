# Module riêng — v3-carrier-comparison

Implementation thuộc build `v3-carrier-comparison`, phương pháp `proxy-compression`.
Mỗi file giữ nguyên byte source đã chốt; import qua alias `mh/<module>.py` tại repository root.

| module | dependency mh |
|---|---|
| [v3_dense.py](dense_proxy_control.py) | `v2_math`, `v2_methods` |
| [v3_math.py](proxy_carrier_codecs.py) | `v2_math` |
| [v3_methods.py](proxy_carrier_methods.py) | `v2_math`, `v2_methods`, `v3_math` |
| [v3_run.py](proxy_carrier_driver.py) | `v2_run`, `v3_dense`, `v3_math`, `v3_methods` |

[Căn cứ/cơ chế](../../../../../../../docs/archive/methods/proxy-compression/design.md) · [Build/kết quả/artifact](../../../../../../../docs/archive/methods/proxy-compression/builds/proxy-carriers/README.md) · [Package dùng chung và compatibility](../../../../../../../docs/operations/package-ownership.md).

Không sửa file này để nối checkpoint cũ; thay thuật toán cần source/tag mới và preflight.
