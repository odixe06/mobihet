# Proposal 2: Phương pháp MobiHet Core cho IDS V2X với low communication budget và hỗ trợ model heterogeneity

Tài liệu trình bày phương pháp MobiHet Core cho bài toán phát hiện xâm nhập trong mạng V2X, giải quyết đồng thời ba điều kiện: communication budget thấp, không chia sẻ dữ liệu cục bộ, và hỗ trợ các phương tiện sử dụng mô hình AI khác nhau.

---

## 1. Bài toán và ràng buộc hệ thống

Hệ thống được đánh giá trên tập dữ liệu VeReMi NextGen v4 gồm $K = 5.343$ xe với 16 lớp nhãn và 77 đặc trưng đầu vào. Mỗi xe $k \in \{1, \dots, K\}$ giữ một tập dữ liệu cục bộ:

$$
\mathcal{D}_k = \{ (x_{k, i}, y_{k, i}) \}_{i=1}^{n_k}, \quad x_{k, i} \in \mathbb{R}^{77}, \; y_{k, i} \in \{0, \dots, 15\}
$$

Dữ liệu giữa các xe có tính chất non-IID: 848 xe có ít hơn 256 mẫu huấn luyện và trung vị chỉ quan sát được 8 trên 16 lớp. Nhiều lớp tấn công không xuất hiện trong dữ liệu cục bộ của các xe nhỏ.

Hệ thống tuân thủ ba ràng buộc:
1. Quyền riêng tư: không chia sẻ dữ liệu thô $\mathcal{D}_k$, nhãn từng dòng, trọng số mô hình hoặc gradient. Máy chủ chỉ dùng phép cộng gộp toàn mạng, tương thích với secure aggregation.
2. Giới hạn communication budget: không truyền trọng số mô hình qua mạng vô tuyến.
3. Hỗ trợ model heterogeneity: mỗi xe được phép dùng kiến trúc mạng nơ-ron riêng $f_k(x; \theta_k)$, miễn là nhận đầu vào 77 chiều và xuất xác suất 16 lớp.

---

## 2. Mô hình toán và cơ chế của MobiHet Core

Quá trình huấn luyện gồm hai giai đoạn: thiết lập Gaussian copula một lần lúc đầu và các round huấn luyện lặp lại.

### 2.1. Giai đoạn 1: Bootstrap Gaussian copula nén

Để bù đắp các lớp vắng ở xe nhỏ mà không cần dữ liệu thật từ xe khác, hệ thống ước lượng mô hình sinh Gaussian copula trên toàn mạng.

#### Bước 1: Ước lượng phân bố biên
Mỗi xe $k$ lập biểu đồ tần số biên $H_{k, c, j}$ trên 256 bin cho từng đặc trưng $j \in \{1, \dots, 77\}$ và lớp $c \in \{0, \dots, 15\}$. Máy chủ cộng dồn:

$$
H_{c, j} = \sum_{k=1}^K H_{k, c, j}
$$

Hàm phân phối tích lũy thực nghiệm $\hat{F}_{c, j}$ được lượng tử hóa thành số nguyên không dấu uint16 để giảm dung lượng rồi gửi lại các xe.

#### Bước 2: Ước lượng tương quan normal-score
Mỗi mẫu $x_{k, i}$ thuộc lớp $c$ được chuẩn hóa qua hàm nghịch đảo phân phối chuẩn:

$$
z_{k, i, j} = \Phi^{-1}(\hat{F}_{c, j}(x_{k, i, j}))
$$

Xe tính tổng vector bậc một và ma trận bậc hai:

$$
S_{k, c}^{(1)} = \sum_{i: y_{k,i}=c} z_{k, i}, \quad S_{k, c}^{(2)} = \sum_{i: y_{k,i}=c} z_{k, i} z_{k, i}^T
$$

Máy chủ gộp toàn mạng để tính vector trung bình $\mu_c$ và ma trận hiệp phương sai $\Sigma_c$:

$$
\mu_c = \frac{\sum_{k=1}^K S_{k, c}^{(1)}}{\sum_{k=1}^K n_{k, c}}, \quad \Sigma_c = \frac{\sum_{k=1}^K S_{k, c}^{(2)}}{\sum_{k=1}^K n_{k, c}} - \mu_c \mu_c^T
$$

Máy chủ phân rã Cholesky $\Sigma_c = L_c L_c^T$, lưu ở định dạng fp16 và gửi về các xe.

#### Sinh synthetic data tại xe
Khi cần synthetic samples của lớp $c$, xe tự sinh trên GPU mà không cần truyền nhận dữ liệu:
1. Lấy mẫu vector chuẩn tắc $u \sim \mathcal{N}(0, I)$.
2. Áp đặt tương quan: $z = \mu_c + L_c u$.
3. Ánh xạ ngược về miền đặc trưng:

$$
\tilde{x}_j = \hat{F}_{c, j}^{-1}(\Phi(z_j)), \quad \forall j \in \{1, \dots, 77\}
$$

---

### 2.2. Giai đoạn 2: Ma trận phân bố xác suất lớp toàn mạng

Tại mỗi round $r$, các xe chia sẻ ma trận dự đoán mà không phụ thuộc vào kiến trúc mô hình:

1. Phía xe: xe $k$ cho dữ liệu thật qua mô hình hiện tại, tính tổng vector xác suất softmax theo từng lớp thật $c$:

$$
V_{k, c} = \sum_{i: y_{k, i} = c} \text{softmax}(f_k(x_{k, i}; \theta_k))
$$

Xe gửi ma trận $V_k \in \mathbb{R}^{16 \times 16}$ và vector số mẫu $n_k$ lên máy chủ với kích thước 544 bytes.

2. Phía máy chủ: máy chủ tính ma trận chuyển tiếp xác suất toàn mạng:

$$
Q_c = \frac{\sum_{k=1}^K V_{k, c}}{\sum_{k=1}^K n_{k, c}}
$$

Hàng $Q_c$ biểu diễn phân bố xác suất dự đoán trung bình của toàn mạng cho một mẫu thuộc lớp $c$. Ma trận $Q \in \mathbb{R}^{16 \times 16}$ đóng vai trò soft teacher độc lập kiến trúc và được gửi về các xe với dung lượng 544 bytes.

---

### 2.3. Giai đoạn 3: Huấn luyện cục bộ tại xe

Tại mỗi round, xe $k$ tối ưu hàm loss kết hợp giữa dữ liệu thật và synthetic data:

#### 1. Loss cross-entropy trên dữ liệu thật
Để giảm thiểu ảnh hưởng của mất cân bằng lớp cục bộ, hàm mất mát cross-entropy được kết hợp logit adjustment với hệ số $\tau = 1.0$:

$$
\mathcal{L}_{\text{real}}(\theta_k) = - \frac{1}{|\mathcal{B}_{\text{real}}|} \sum_{(x, y) \in \mathcal{B}_{\text{real}}} \log \frac{\exp(f_{k, y}(x) + \tau \log \pi_{k, y})}{\sum_{j=0}^{15} \exp(f_{k, j}(x) + \tau \log \pi_{k, j})}
$$

Trong đó $\pi_{k, c} = n_{k, c} / n_k$ là tỷ lệ của lớp $c$ tại xe $k$.

#### 2. Loss distillation trên synthetic samples
Ở mỗi bước, xe sinh 64 mẫu nhân tạo $\mathcal{B}_{\text{syn}}$ với nhãn giả định $y_{\text{syn}}$ phân bố đều trên 16 lớp. Mô hình được hướng dẫn theo phân bố $Q$ thông qua Kullback-Leibler divergence:

$$
\mathcal{L}_{\text{syn}}(\theta_k) = \frac{1}{|\mathcal{B}_{\text{syn}}|} \sum_{(\tilde{x}, y_{\text{syn}}) \in \mathcal{B}_{\text{syn}}} D_{\text{KL}}(Q[y_{\text{syn}}] \parallel \text{softmax}(f_k(\tilde{x}; \theta_k)))
$$

$$
D_{\text{KL}}(Q_c \parallel p) = \sum_{j=0}^{15} Q_{c, j} \log \left( \frac{Q_{c, j}}{p_j} \right)
$$

#### 3. Hàm mục tiêu tối ưu
Tổng hàm mất mát tại mỗi bước huấn luyện:

$$
\mathcal{L}_{\text{total}}(\theta_k) = \mathcal{L}_{\text{real}}(\theta_k) + \mathcal{L}_{\text{syn}}(\theta_k)
$$

#### 4. Cơ chế step floor
Với các xe ít dữ liệu, một local epoch kết thúc sau một vài bước nhỏ. Cơ chế step floor quy định:
- Dữ liệu thật được duyệt đúng một epoch.
- Nếu số bước thật $S_k < 16$, xe chạy thêm $16 - S_k$ bước chỉ trên dữ liệu nhân tạo với hàm mất mát $\mathcal{L}_{\text{syn}}$.
- Cơ chế này giúp các lớp vắng được tối ưu đủ số bước mà không phát sinh thêm chi phí truyền thông.

---

## 3. Bảng so sánh communication cost

Điều kiện thực nghiệm: 5.343 xe, 50 round liên lạc, DAGSNet W=1 với 395.408 tham số và 3.264 buffer chuẩn hóa, độ chính xác fp32.

| Phương pháp | Upload mỗi xe một round | Download mỗi xe một round | Tổng mỗi xe một round | Bootstrap một lần | Tổng communication budget mỗi xe qua 50 round | Tổng lưu lượng toàn mạng 5.343 xe | Mức giảm communication budget so với FedAvg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Centralized | - | - | - | - | 0 B | 0 B | - |
| SCAFFOLD | 3,18 MB | 3,18 MB | 6,35 MB | 0 B | 317,63 MB | 1.697,1 GB | -99,1% |
| FedAvg | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | 159,47 MB | 852,0 GB | Mốc so sánh |
| FedProx | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | 159,47 MB | 852,0 GB | 0% |
| AFPHA | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | 159,47 MB | 852,0 GB | 0% |
| FD-IDS | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | 159,47 MB | 852,0 GB | 0% |
| APFed | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | 159,47 MB | 852,0 GB | 0% |
| NILM-FL | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | 159,47 MB | 852,0 GB | 0% |
| PerFed-SKD | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | ~95,7 MB | ~511,2 GB | Giảm ~40% |
| CANPerFL | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | 159,47 MB | 852,0 GB | 0% |
| LwFedNIDS | 162 KB | 162 KB | 324 KB | 0 B | 16,20 MB | 86,6 GB | Giảm ~10 lần |
| V2X-HtFL | 16,4 KB | 16,4 KB | 32,8 KB | 0 B | 1,64 MB | 8,76 GB | Giảm ~97 lần |
| FedProto | 16,4 KB | 16,4 KB | 32,8 KB | 0 B | 1,64 MB | 8,76 GB | Giảm ~97 lần |
| FedTGP | 16,4 KB | 16,4 KB | 32,8 KB | 0 B | 1,64 MB | 8,76 GB | Giảm ~97 lần |
| TinyProto | 1,7 KB | 1,7 KB | 3,3 KB | 0 B | 0,17 MB | 0,91 GB | Giảm ~940 lần |
| pFedES | 1,4 KB | 1,4 KB | 2,8 KB | 0 B | 0,14 MB | 0,75 GB | Giảm ~1.100 lần |
| MobiHet Core | 544 B | 544 B | 1.088 B | 1,00 MB | 1,05 MB | 5,63 GB | Giảm ~151 lần mỗi round và ~37 lần tổng 50 round |

Ghi chú:
- Các phương pháp weight-sharing như FedAvg, APFed, SCAFFOLD phải gửi toàn bộ tham số mô hình qua từng round nên tổng lưu lượng toàn mạng lên đến 852 GB đến 1,7 TB.
- Các phương pháp dùng prototype giảm lưu lượng xuống mức kilobyte nhưng đi kèm suy giảm độ chính xác.
- MobiHet Core gửi 1.088 bytes mỗi round. Tính cả giai đoạn bootstrap một lần lúc đầu, tổng communication budget cho 50 round là 1,05 MB mỗi xe.

---

## 4. Bảng so sánh computational cost

Quy ước: $F = 4,45$ MFLOP là chi phí một lượt forward qua mô hình DAGSNet full trên một mẫu. $n_k$ là số mẫu huấn luyện của xe $k$ với tổng toàn mạng 37.361.273 mẫu. Phép đo thực hiện trên hai GPU NVIDIA T4.

| Phương pháp | Computational cost tại client trên mỗi mẫu mỗi round | Độ phức tạp client mỗi round | Độ phức tạp server mỗi round | Local batch size | Thời gian thực thi 50 round trên 2 GPU T4 | Khả năng đáp ứng phần cứng ECU xe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Centralized | - | - | $O(N_{\text{total}} F)$ | 4.096 | ~4,05 giờ | Cần máy chủ trung tâm lớn |
| FedAvg | $2,99 \times F \approx 13,3$ MFLOP | $O(3 n_k F)$ | $O(K \cdot P)$ | 64 | ~4,20 giờ | Nhẹ |
| FedProx | $2,99 \times F \approx 13,3$ MFLOP | $O(3 n_k F + S_k P)$ | $O(K \cdot P)$ | 64 | ~4,30 giờ | Nhẹ |
| SCAFFOLD | $2,99 \times F \approx 13,3$ MFLOP | $O(3 n_k F + S_k P)$ | $O(K \cdot P)$ | 64 | ~4,80 giờ | Vừa phải |
| AFPHA | $2,99 \times F \approx 13,3$ MFLOP | $O(3 n_k F + S_k P)$ | $O(K \cdot P + K \cdot \text{clusters})$ | 64 | ~4,35 giờ | Nhẹ |
| FD-IDS | $3,99 \times F \approx 17,8$ MFLOP | $O(4 n_k F)$ | $O(K \cdot P)$ | 64 | ~4,60 giờ | Vừa phải |
| LwFedNIDS | $0,29 \times F \approx 1,3$ MFLOP | $O(3 n_k F')$ | $O(K \cdot P')$ | 64 | ~3,80 giờ | Rất nhẹ |
| FedProto | $2,99 \times F \approx 13,3$ MFLOP | $O(3 n_k F)$ | $O(N_p \cdot d)$ | 64 | ~4,20 giờ | Nhẹ |
| FedTGP | $3,99 \times F \approx 17,8$ MFLOP | $O(4 n_k F)$ | $O(S \cdot N_p C d + S d^2)$ | 64 | ~4,90 giờ | Máy chủ nặng |
| TinyProto | $3,99 \times F \approx 17,8$ MFLOP | $O(4 n_k F)$ | $O(N_p \cdot d)$ | 64 | ~4,40 giờ | Nhẹ |
| V2X-HtFL | $4,11 \times F \approx 18,3$ MFLOP | $O(4 n_k F + n_k d^2)$ | $O(S \cdot N_p C d)$ | 64 | ~4,70 giờ | Máy chủ nặng |
| APFed | $5,97 \times F \approx 26,6$ MFLOP | $O(6 n_k F)$ | $O(K \cdot P)$ | 64 | ~7,80 giờ | Nặng tại xe |
| NILM-FL | $5,97 \times F \approx 26,6$ MFLOP | $O(6 n_k F)$ | $O(K \cdot P)$ | 64 | ~8,10 giờ | Nặng tại xe |
| pFedES | $7,98 \times F \approx 35,5$ MFLOP | $O(8 n_k F)$ | $O(K \cdot P_G)$ | 64 | ~8,50 giờ | Nặng tại xe |
| CANPerFL | $2,99 \times F$ pha 1 và 10 epoch pha 2 | $O(3 n_k F) + O(30 n_k F)$ | $O(K \cdot P)$ | 64 | ~5,10 giờ | Vừa phải |
| MobiHet Core | $4,00 \times F \approx 17,8$ MFLOP | $O(4 n_k F)$ | $O(K \cdot C^2)$ | 64 | ~9,19 giờ | Phù hợp ECU xe |

Ghi chú:
- Hệ số $c_m \times F$ đo lượng phép tính trên mỗi mẫu. FedAvg tiêu chuẩn tương đương ba lượt forward và backward.
- MobiHet Core tiêu tốn khoảng $4,0 \times F$ tại xe do thêm bước sinh và tính loss trên synthetic data, phù hợp với các dòng vi xử lý biên trên xe.
- Phía máy chủ, MobiHet Core chỉ cộng ma trận $16 \times 16$, thời gian xử lý dưới 5 mili-giây mỗi round.

---

## 5. Bảng so sánh performance trên tập holdout test tại round 50

Điều kiện thực nghiệm đồng nhất cho tất cả các phương pháp:
- Dataset: VeReMi NextGen v4 chuẩn hóa với 5.343 xe.
- Tập huấn luyện: 37.361.273 mẫu gồm fold train và val, mỗi round chạy đúng một local epoch.
- Tập kiểm tra: 9.347.679 mẫu cố định từ 20% timeline cuối của từng xe, đánh giá theo hình thức temporal holdout.
- Kiến trúc mô hình chuẩn: DAGSNet W=1 với 395.408 tham số, dropout 0.1.
- Tham số chung: local batch size 64, optimizer AdamW, cosine annealing learning rate từ $10^{-3}$ xuống $10^{-5}$ qua 50 round, seed 42. Số liệu của tất cả các phương pháp lấy chính xác tại round 50.

| Phương pháp | View đánh giá | Hỗ trợ model heterogeneity | Accuracy | Macro recall | Macro F1-score | Client P10 F1 của 10% xe yếu nhất | Client min F1 của xe thấp nhất |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Centralized | Global | Không | 0,9580 | 0,9531 | 0,9593 | - | - |
| APFed | Client own-test | Không | 0,9278 | 0,9032 | 0,9119 | 0,7533 | 0,1500 |
| CANPerFL | Client fine-tuned | Không | 0,9233 | 0,8986 | 0,9088 | 0,7684 | 0,1545 |
| FedAvg | Global | Không | 0,9228 | 0,8957 | 0,9042 | 0,7540 | 0,1500 |
| FD-IDS | Global | Không | 0,9220 | 0,8914 | 0,8992 | 0,7552 | 0,2176 |
| AFPHA | Global | Không | 0,9211 | 0,8932 | 0,8990 | 0,7523 | 0,1786 |
| FedProx | Global | Không | 0,9214 | 0,8941 | 0,8998 | 0,7524 | 0,1786 |
| SCAFFOLD | Global | Không | 0,9225 | 0,8859 | 0,8911 | 0,7738 | 0,2963 |
| LwFedNIDS | Global | Không | 0,9162 | 0,8830 | 0,8904 | 0,7291 | 0,2139 |
| NILM-FL | Client own-test | Không | 0,9156 | 0,8717 | 0,8846 | 0,5777 | 0,0000 |
| FedTGP | Client proto | Có | 0,9020 | 0,8610 | 0,8719 | 0,5712 | 0,0000 |
| PerFed-SKD | Client own-test | Không | 0,8994 | 0,8598 | 0,8710 | 0,6573 | 0,1241 |
| FedProto | Client proto | Có | 0,8978 | 0,8651 | 0,8698 | 0,5911 | 0,0000 |
| V2X-HtFL | Client own-test | Có | 0,8974 | 0,8554 | 0,8661 | 0,5519 | 0,0000 |
| TinyProto | Client proto | Có | 0,8866 | 0,8457 | 0,8534 | 0,5521 | 0,0000 |
| pFedES | Client own-test | Có | 0,8864 | 0,8405 | 0,8511 | 0,5264 | 0,0000 |
| MobiHet Core | Client own-test | Có | 0,9094 | 0,9022 | 0,8992 | 0,7562 | 0,1538 |

Phân tích kết quả:
- Trong nhóm hỗ trợ model heterogeneity gồm FedProto, FedTGP, TinyProto, pFedES và V2X-HtFL, macro F1 dao động từ 0,8511 đến 0,8719. Điểm F1 của nhóm 10% xe yếu nhất rơi xuống mức 0,52 đến 0,59 và điểm thấp nhất về 0 do thiếu biểu diễn nguyên mẫu cho các lớp vắng. MobiHet Core đạt macro F1 0,8992, macro recall 0,9022 và giữ client P10 ở mức 0,7562.
- So với nhóm weight-sharing FL yêu cầu cùng kiến trúc mô hình, MobiHet Core có macro F1 đạt 0,8992, tương đương FedProx 0,8998, AFPHA 0,8990, FD-IDS 0,8992 và sát với FedAvg 0,9042, đồng thời giảm 99,3% lưu lượng mạng.

---

## 6. Phân tích cơ chế và trade-off của các phương pháp đối chứng

1. Centralized DAGSNet:
Mô hình tập hợp toàn bộ dữ liệu về một máy chủ, huấn luyện với batch size lớn 4.096 và xáo trộn IID. Kết quả F1 0,9593 đóng vai trò cận trên tham chiếu. Cách tiếp cận này không thể triển khai thực tế do vi phạm quy định bảo mật dữ liệu và tạo tắc nghẽn mạng truyền thông.

2. FedAvg:
Máy chủ tính trung bình cộng trọng số mô hình cục bộ. Phương pháp yêu cầu toàn bộ xe dùng chung một kiến trúc mô hình và gặp hiện tượng client drift khi dữ liệu non-IID, đồng thời tiêu tốn 159 MB mỗi xe.

3. FedProx:
Bổ sung số hạng proximal vào hàm mất mát cục bộ để hạn chế độ lệch trọng số giữa xe và máy chủ. Kết quả macro F1 đạt 0,8998, tương đương FedAvg, và vẫn yêu cầu đồng nhất kiến trúc mô hình.

4. SCAFFOLD:
Sử dụng control variates để điều chỉnh hướng cập nhật gradient của từng client. Phương pháp nâng client P10 lên 0,7738 nhưng làm tăng gấp đôi dung lượng truyền thông lên 317 MB mỗi xe và nhạy cảm với bất ổn số học ở những round đầu.

5. APFed:
Áp dụng cá nhân hóa mô hình bằng cách học trọng số kết hợp giữa mô hình cục bộ và biểu diễn toàn cục. Phương pháp đạt macro F1 0,9119 nhưng chi phí tính toán tại xe tăng gấp đôi lên $5,97 \times F$ và vẫn phụ thuộc vào việc trao đổi trọng số.

6. NILM-FL:
Mỗi xe huấn luyện hai mô hình gồm private model và proxy model qua cơ chế distillation hai chiều. Chi phí tính toán tại xe tăng gấp đôi, trong khi macro F1 giảm xuống 0,8846 và client P10 giảm về 0,5777 do các xe nhỏ không đủ dữ liệu để huấn luyện proxy model.

7. FedProto và FedTGP:
Thay vì gửi trọng số, các xe gửi vector prototype trung bình cho từng lớp trong không gian đặc trưng 256 chiều. Cách làm này giảm băng thông nhưng biểu diễn prototype bị méo mó khi dữ liệu non-IID và thiếu nhãn. Các xe nhỏ không nhận được prototype cho các lớp vắng, khiến macro F1 chỉ đạt 0,8698 đến 0,8719 và client P10 giảm dưới 0,59.

8. TinyProto:
Áp dụng nén thưa hóa các vector prototype để giảm communication budget xuống 1,7 KB một round. Việc nén thưa làm suy giảm thông tin ranh giới giữa các lớp, dẫn đến macro F1 đạt 0,8534.

9. pFedES và V2X-HtFL:
Sử dụng mạng tạo sinh hoặc bộ trích xuất đặc trưng proxy để kết nối các mô hình không đồng nhất. Chi phí tính toán tại xe hoặc máy chủ tăng cao, trong khi macro F1 trên bài toán 16 lớp đạt 0,8511 đến 0,8661.

10. MobiHet Core:
Thay vì truyền prototype trong không gian ẩn phụ thuộc vào bộ trích xuất đặc trưng, MobiHet Core tách rời biểu diễn bằng cách ước lượng phân bố copula ở không gian đầu vào và chỉ trao đổi ma trận xác suất $Q$ kích thước $16 \times 16$ ở không gian đầu ra. Kết hợp cùng cơ chế step floor, phương pháp giải quyết đồng thời bài toán non-IID, lớp vắng và model heterogeneity với communication budget 1,05 MB cho 50 round.
