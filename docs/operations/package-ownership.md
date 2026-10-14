# Code chung, code build và tên cũ/mới

Code chung vật lý nằm trong `core/`; implementation riêng nằm trong build active hoặc archive.
`mh/ → .compat/mh/` chỉ là facade tương thích để snapshot cũ vẫn import và hash đúng 40 tên file.
Không có các version thuật toán độc lập V2/V3/V4 đang cùng chạy trong core.

[Tên đợt thí nghiệm](experiment-names.md) · [Layout](project-layout.md) · [Core](../../core/README.md).
## Chung toàn dự án

| tên cũ để truy vết | source canonical |
|---|---|
| `mh/__init__.py` | [__init__.py](../../core/__init__.py) |
| `mh/archs.py` | [legacy_model_factory.py](../../core/legacy_model_factory.py) |
| `mh/channel.py` | [rsu_contact.py](../../core/rsu_contact.py) |
| `mh/dagsnet.py` | [dagsnet.py](../../core/dagsnet.py) |
| `mh/metrics.py` | [classification_metrics.py](../../core/classification_metrics.py) |
| `mh/mobility.py` | [vehicle_mobility.py](../../core/vehicle_mobility.py) |
| `mh/model_packet.py` | [model_codec.py](../../core/model_codec.py) |
| `mh/new_fl.py` | [artifact_driver.py](../../core/artifact_driver.py) |
| `mh/pilot.py` | [pilot_runtime.py](../../core/pilot_runtime.py) |
| `mh/prepare.py` | [prepare_data.py](../../core/prepare_data.py) |
| `mh/teacher_base.py` | [teacher_training.py](../../core/teacher_training.py) |
| `mh/teacher_driver.py` | [teacher_driver.py](../../core/teacher_driver.py) |
| `mh/v2_math.py` | [vector_codecs.py](../../core/vector_codecs.py) |
| `mh/v2_methods.py` | [shared_prior_and_proxy.py](../../core/shared_prior_and_proxy.py) |
| `mh/v2_run.py` | [federated_driver.py](../../core/federated_driver.py) |
| `mh/v4_schema.py` | [feature_support_schema.py](../../core/feature_support_schema.py) |
| `mh/v6_run.py` | [expanded_fixture_runtime.py](../../core/expanded_fixture_runtime.py) |
| `mh/v7_basis.py` | [covariance_basis.py](../../core/covariance_basis.py) |
| `mh/v7_nystrom.py` | [gaussian_moments.py](../../core/gaussian_moments.py) |

## Riêng build đang làm

| tên cũ để truy vết | source canonical |
|---|---|
| `mh/v8_hybrid.py` | [hybrid_models.py](../../methods/gaussian-proxy-hybrid/builds/hybrid-full/source/mh/hybrid_models.py) |

## Implementation lưu trữ

| tên cũ để truy vết | source canonical |
|---|---|
| `mh/new_methods.py` | [distribution_matching.py](../../archive/methods/synthetic-distribution-matching/builds/static-synthetic-books/source/mh/distribution_matching.py) |
| `mh/p2.py` | [fragmented_snapshot_driver.py](../../archive/methods/contact-fragmentation/builds/fragmented-snapshots/source/mh/fragmented_snapshot_driver.py) |
| `mh/v10_energy.py` | [gaussian_base_energy.py](../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/source/mh/gaussian_base_energy.py) |
| `mh/v10_run.py` | [joint_energy_driver.py](../../archive/methods/dagsnet-joint-energy/builds/gaussian-base-energy/source/mh/joint_energy_driver.py) |
| `mh/v3_dense.py` | [dense_proxy_control.py](../../archive/methods/proxy-compression/builds/proxy-carriers/source/mh/dense_proxy_control.py) |
| `mh/v3_math.py` | [proxy_carrier_codecs.py](../../archive/methods/proxy-compression/builds/proxy-carriers/source/mh/proxy_carrier_codecs.py) |
| `mh/v3_methods.py` | [proxy_carrier_methods.py](../../archive/methods/proxy-compression/builds/proxy-carriers/source/mh/proxy_carrier_methods.py) |
| `mh/v3_run.py` | [proxy_carrier_driver.py](../../archive/methods/proxy-compression/builds/proxy-carriers/source/mh/proxy_carrier_driver.py) |
| `mh/v4_mala.py` | [mala_transport.py](../../archive/methods/gaussian-sampling/builds/support-mala/source/mh/mala_transport.py) |
| `mh/v4_run.py` | [support_mala_driver.py](../../archive/methods/gaussian-sampling/builds/support-mala/source/mh/support_mala_driver.py) |
| `mh/v4_support.py` | [support_conditioned_gaussian.py](../../archive/methods/gaussian-sampling/builds/support-mala/source/mh/support_conditioned_gaussian.py) |
| `mh/v5_anchor.py` | [kernel_anchor_targets.py](../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/source/mh/kernel_anchor_targets.py) |
| `mh/v5_books.py` | [gradient_latent_books.py](../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/source/mh/gradient_latent_books.py) |
| `mh/v5_run.py` | [synthetic_books_driver.py](../../archive/methods/synthetic-knowledge/builds/anchor-synthetic-books/source/mh/synthetic_books_driver.py) |
| `mh/v6_geometry.py` | [geometry_moments.py](../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/source/mh/geometry_moments.py) |
| `mh/v6_schema.py` | [geometry_feature_schema.py](../../archive/methods/geometry-primitive-moments/builds/geometry-g59-g61/source/mh/geometry_feature_schema.py) |
| `mh/v7_run.py` | [covariance_sketch_driver.py](../../archive/methods/covariance-nystrom/builds/nystrom-rank4-rank8/source/mh/covariance_sketch_driver.py) |
| `mh/v8_run.py` | [hybrid_pilot_driver.py](../../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/source/mh/hybrid_pilot_driver.py) |
| `mh/v9_anchor.py` | [anchor_residual_teacher.py](../../archive/methods/anchor-function-residual/builds/anchor-residual/source/mh/anchor_residual_teacher.py) |
| `mh/v9_run.py` | [anchor_residual_driver.py](../../archive/methods/anchor-function-residual/builds/anchor-residual/source/mh/anchor_residual_driver.py) |

19 module chung, 1 module Hybrid đang được full build dùng trực tiếp, 20 implementation riêng của build đã kết thúc.
Driver/common helpers có thể import một implementation lịch sử; source frozen vẫn cần dependency đó.
Đóng build không cấp quyền xóa file còn trong source fingerprint. Không đổi byte Python hoặc notebook đã PUSHED.

`pilot_runtime.py` giữ cùng độ sâu root như trước: `pilot.ROOT` vẫn repository root. Generator đọc byte qua
facade rồi dựng runtime cũ trên Kaggle; không cần symlink tại Kaggle. Namespace `scripts` tìm helper lịch sử
trong `.compat/scripts`; code helper mới dùng `scripts.project_paths.current_path` khi đọc đường dẫn cũ.

[Layout/ownership hiện tại](workspace-layout.json) · [Manifest nguồn trước chuyển](package-layout-2026-10-14.json) ·
[Kiểm import/hash/runtime](package-verification-2026-10-14.json) · [Source parent](package-resume-source-check-2026-10-14.json) ·
[Git/DVC/link/key](layout-verification-2026-10-14.json).

Kiểm: `conda run -n nckh python scripts/check_package_layout.py` rồi `python scripts/check_project_layout.py`.
