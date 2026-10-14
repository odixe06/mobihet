# Core — code dùng chung

19 module dùng cho model, data, metrics, RSU, codec và driver nhiều phương pháp. Tên file mô tả nhiệm vụ;
không dùng nhãn V2/V3/V4 làm tên file canonical.

| nhiệm vụ | source |
|---|---|
| DAGSNet | [dagsnet.py](dagsnet.py) |
| Chuẩn bị dữ liệu, pilot runtime | [prepare_data.py](prepare_data.py), [pilot_runtime.py](pilot_runtime.py) |
| 10 metrics | [classification_metrics.py](classification_metrics.py) |
| Giao tiếp RSU, mobility | [rsu_contact.py](rsu_contact.py), [vehicle_mobility.py](vehicle_mobility.py) |
| Model/vector codec | [model_codec.py](model_codec.py), [vector_codecs.py](vector_codecs.py) |
| Prior và proxy dùng chung | [shared_prior_and_proxy.py](shared_prior_and_proxy.py) |
| Gaussian moments và basis | [gaussian_moments.py](gaussian_moments.py), [covariance_basis.py](covariance_basis.py) |
| Driver, artifact I/O | [federated_driver.py](federated_driver.py), [artifact_driver.py](artifact_driver.py) |
| Teacher, fixture runtime | [teacher_training.py](teacher_training.py), [teacher_driver.py](teacher_driver.py), [expanded_fixture_runtime.py](expanded_fixture_runtime.py) |
| Feature schema, factory lịch sử | [feature_support_schema.py](feature_support_schema.py), [legacy_model_factory.py](legacy_model_factory.py) |

Byte Python giữ nguyên. Driver cũ vẫn import namespace `mh` qua facade `mh/ → .compat/mh/`;
notebook copy đủ 40 tên source cũ để fingerprint/checkpoint không đổi. Không sửa import hoặc thêm tên file vào
snapshot đã chạy. Factory lịch sử không thay quyết định learned model hiện DAGSNet-only.

[Ownership và tên cũ/mới](../docs/operations/package-ownership.md) · [Cấu trúc](../docs/operations/project-layout.md).
