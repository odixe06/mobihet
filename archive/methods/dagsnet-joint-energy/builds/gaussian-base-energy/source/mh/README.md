# Module riêng — v10-gaussian-base-mh

Implementation thuộc build `v10-gaussian-base-mh`, phương pháp `dagsnet-joint-energy`.
Mỗi file giữ nguyên byte source đã chốt; import qua alias `mh/<module>.py` tại repository root.

| module | dependency mh |
|---|---|
| [v10_energy.py](gaussian_base_energy.py) | `pilot`, `teacher_base`, `v2_methods` |
| [v10_run.py](joint_energy_driver.py) | `teacher_driver`, `v10_energy`, `v6_run` |

[Căn cứ/cơ chế](../../../../../../../docs/archive/methods/dagsnet-joint-energy/design.md) · [Build/kết quả/artifact](../../../../../../../docs/archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/README.md) · [Package dùng chung và compatibility](../../../../../../../docs/operations/package-ownership.md).

Không sửa file này để nối checkpoint cũ; thay thuật toán cần source/tag mới và preflight.
