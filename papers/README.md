# papers/ — bài gốc (PDF chỉ giữ local, không đẩy Git)

Nguồn local: `~/Cloud/OneDrive/NCKH/mobihet_ref/`; bài arXiv tải ngày 29/10/2026.

| file | phương pháp | bài | nguồn |
|---|---|---|---|
| `afpha_edl_cmso.pdf` | AFPHA_EDL_CMSO, DAGSNet | Khan et al., A Secure and Efficient Deep Learning-Based IDS Framework for the IoV, Sci. Rep. 2025 | mobihet_ref |
| `fd_ids.pdf` | FD-IDS | Peng et al., Sensors 25:4309, 2025 | mobihet_ref |
| `pfedes.pdf` | pFedES | Yi et al., AAAI 2025 | mobihet_ref |
| `tinyproto.pdf` | TinyProto | Lee & Choi, AAAI 2026 | mobihet_ref |
| `nilm.pdf` | NILM-FL | Li et al., IEEE TSG 2025 | mobihet_ref |
| `lwfednids.pdf` | LwFedNIDS | Bouayad et al., IEEE Access 2024 | mobihet_ref |
| `perfedskd.pdf` | PerFed-SKD | Singh et al., IEEE TCE 2024 | mobihet_ref |
| `fedavg.pdf` | FedAvg | McMahan et al., AISTATS 2017 | arXiv 1602.05629 |
| `fedprox.pdf` | FedProx | Li et al., MLSys 2020 | arXiv 1812.06127 |
| `scaffold.pdf` | SCAFFOLD | Karimireddy et al., ICML 2020 | arXiv 1910.06378 |
| `fedproto.pdf` | FedProto | Tan et al., AAAI 2022 | arXiv 2105.00243 |
| `fedtgp.pdf` | FedTGP | Zhang et al., AAAI 2024 | arXiv 2401.03230 |

**Còn thiếu — owner tải tay, đặt đúng tên file:**

| file | bài | link |
|---|---|---|
| `apfed.pdf` | Adaptive personalized FL with lightweight depthwise convolutional bottleneck network for novel IDS in IoV, Sci. Rep. 2025 | https://doi.org/10.1038/s41598-025-17699-3 |
| `canperfl.pdf` | CANPerFL: Improve in-vehicle intrusion detection performance by sharing knowledge, Appl. Sci. 13(11):6369, 2023 | https://doi.org/10.3390/app13116369 |
| `v2x_htfl.pdf` | Heterogeneous FL for V2X: Feature Prototype Aggregation and Generative Feedback Mechanism, IEEE OJVT 2025 | IEEE Xplore (không tìm thấy bản arXiv) |

## Nguồn tham khảo (không là trạng thái triển khai)

Nguồn để trích dẫn về sau, không phải bằng chứng utility trên VeReMi; cơ chế đã chạy ghi ở [method](../docs/research/method.md).
"Tóm tắt" = mới đọc abstract/tóm tắt tìm kiếm, phải đọc toàn văn trước khi trích chi tiết.

| bài | liên quan | mức đã đọc | nguồn |
|---|---|---|---|
| Duan et al., Fed-TDA: Federated Tabular Data Augmentation on Non-IID Data, 2022 | **gần với bootstrap Core**: phân bố từng cột + covariance toàn cục sau biến đổi về chuẩn, client tự sinh bảng giả; phải nêu khi tuyên bố đóng góp | đối chiếu thêm trong [nguồn primary](../docs/research/generator-primary-sources.md) | https://arxiv.org/abs/2211.13116 |
| Beitollahi et al., Parametric Feature Transfer: One-shot FL with Foundation Models, 2024 | gửi GMM theo lớp trên đặc trưng foundation model, một lượt; đối chiếu khi bàn hướng one-shot | tóm tắt | https://arxiv.org/abs/2402.01862 |
| Yi et al., Federated Model Heterogeneous Matryoshka Representation Learning (FedMRL), NeurIPS 2024 | model nhỏ đồng nhất đi kèm model local khác kiến trúc, chỉ truyền model nhỏ | tóm tắt | https://proceedings.neurips.cc/paper_files/paper/2024/hash/7a9ee756fb71d401bcb5230db02c11db-Abstract.html |
| Zhang et al., DENSE: Data-Free One-Shot Federated Learning, NeurIPS 2022 | one-shot, sinh dữ liệu từ model client; cần upload model nên không hợp ràng buộc hiện tại | tóm tắt | NeurIPS 2022 |
| Menon et al., Long-tail Learning via Logit Adjustment, ICLR 2021 | LA trong Core; đề xuất PH (hiệu chỉnh prior local sau train) | tóm tắt | https://arxiv.org/abs/2007.07314 |
| Izmailov et al., Averaging Weights Leads to Wider Optima and Better Generalization (SWA), UAI 2018 | đề xuất EM5/EM8 (trung bình trọng số theo round) | tóm tắt | https://arxiv.org/abs/1803.05407 |
| Tarvainen & Valpola, Mean Teachers are Better Role Models, NeurIPS 2017 | EMA trọng số; đề xuất EM5/EM8 | tóm tắt | https://arxiv.org/abs/1703.01780 |
| Jeong et al., Communication-Efficient On-Device ML: Federated Distillation and Augmentation, 2018 | trao đổi xác suất theo lớp (Q của Core); đề xuất QK | tóm tắt | https://arxiv.org/abs/1811.11479 |
| Hinton et al., Distilling the Knowledge in a Neural Network, 2015 | soft target/nhiệt độ; đề xuất QT2 | tóm tắt | https://arxiv.org/abs/1503.02531 |
| van der Heijden et al., VeReMi: A Dataset for Comparable Evaluation of Misbehavior Detection in VANETs, 2018 | nguồn gốc dataset | tóm tắt | https://arxiv.org/abs/1804.06701 |
