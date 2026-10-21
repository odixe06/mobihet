# Module riêng — v8-pilot

Implementation thuộc build `v8-pilot`, phương pháp `gaussian-proxy-hybrid`.
Mỗi file giữ nguyên byte source đã chốt; import qua alias `mh/<module>.py` tại repository root.

| module | dependency mh |
|---|---|
| [v8_hybrid.py](../../../../../../../methods/gaussian-proxy-hybrid/builds/hybrid-full/source/mh/hybrid_models.py) | `model_packet`, `pilot`, `v2_math`, `v2_methods`, `v7_nystrom` |
| [v8_run.py](hybrid_pilot_driver.py) | `v2_run`, `v6_run`, `v8_hybrid` |

[Căn cứ/cơ chế](../../../../../../../docs/methods/gaussian-proxy-hybrid/design.md) · [Build/kết quả/artifact](../../../../../../../docs/archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/README.md) · [Package dùng chung và compatibility](../../../../../../../docs/operations/project-layout.md).

Không sửa file này để nối checkpoint cũ; thay thuật toán cần source/tag mới và preflight.
