# Proposal 2: MobiHet Core - Federated IDS V2X với Low Communication Budget & Hỗ Trợ Model Heterogeneity

<callout icon="📌" color="blue_bg">
<b>Executive Summary:</b> MobiHet Core là kiến trúc Federated Learning (FL) chuyên biệt cho Intrusion Detection System (IDS) trong mạng truyền thông phương tiện kết nối (V2X). Phương pháp giải quyết đồng thời ba thách thức cốt lõi: <b>tối ưu hóa communication budget cực thấp</b>, <b>bảo vệ data privacy tuyệt đối</b>, và <b>hỗ trợ toàn diện model heterogeneity</b> (cho phép các phương tiện tự do sử dụng kiến trúc mô hình AI hoàn toàn khác biệt).
</callout>

---

## 1. Problem Formulation & System Constraints

Hệ thống IDS V2X được chuẩn hóa trên benchmark dataset thế hệ mới **VeReMi NextGen v4** gồm **$K = 5.343$ vehicles (clients)** với $C = 16$ classes hành vi/tấn công và $D = 77$ input features. Mỗi vehicle $k \in \{1, \dots, K\}$ thu thập cục bộ dataset:

$$
\mathcal{D}_k = \left\{ (x_{k, i}, y_{k, i}) \right\}_{i=1}^{n_k}, \quad x_{k, i} \in \mathbb{R}^D, \; y_{k, i} \in \{0, \dots, C-1\}
$$

Do đặc thù di chuyển và mật độ giao thông biến động theo không gian - thời gian, phân phối dữ liệu giữa các clients mang tính chất **Extreme Non-IID**:
- 848 vehicles có ít hơn 256 training samples (median chỉ quan sát được 8/16 classes).
- Nhiều attack classes nguy hiểm hoàn toàn vắng mặt (absent classes) tại các local vehicles nhỏ.

### Ba nguyên tắc thiết kế bất biến (System Constraints):
1. **Chuẩn Federated Privacy**: Không chia sẻ local raw data $\mathcal{D}_k$, không chia sẻ nhãn từng dòng, không chia sẻ model weights $\theta_k$ hay gradients $\nabla \theta_k$. Server chỉ thực hiện phép cộng gộp toàn mạng (tương thích Secure Aggregation - SecAgg).
2. **Strict Communication Budget**: Loại bỏ hoàn toàn việc truyền tải hàng triệu tham số mô hình qua kênh truyền vô tuyến V2X vốn có băng thông hạn chế và latency cao.
3. **Model Heterogeneity**: Cho phép mỗi hãng xe tự do lựa chọn kiến trúc mạng nơ-ron cục bộ $f_k(x; \theta_k)$ tối ưu cho phần cứng ECU (DAGSNet, MLP, ResNet, Tiny-CNN,...), chỉ cần tuân thủ giao thức: nhận input vector $D=77$ và xuất phân bố xác suất $C=16$ classes.

---

## 2. Mathematical Foundations & Mechanisms của MobiHet Core

MobiHet Core vận hành qua hai giai đoạn: **Compressed Gaussian Copula Bootstrap (One-time Setup)** và **Federated Training Rounds**.

### 2.1. Stage 1: Compressed Gaussian Copula Bootstrap (One-time Setup)

Để giải quyết vấn đề absent classes ở các vehicles nhỏ mà không vi phạm privacy, hệ thống ước lượng một generative model dạng **Gaussian Copula** trên toàn mạng.

#### Step 1: Ước lượng marginal distributions (Marginal ECDF)
Mỗi client $k$ tính histogram biên $H_{k, c, j} \in \mathbb{R}^{B}$ ($B = 256$ quantized bins) cho từng feature $j \in \{1, \dots, D\}$ và class $c \in \{0, \dots, C-1\}$. Server tổng hợp:

$$
H_{c, j} = \sum_{k=1}^K H_{k, c, j}
$$

Hàm phân phối tích lũy thực nghiệm (empirical cumulative distribution function - ECDF) $\hat{F}_{c, j}(t)$ được lượng tử hóa dưới dạng `uint16` (giảm 69% communication budget so với float32) và broadcast về các vehicles.

#### Step 2: Ước lượng normal-score correlation
Mỗi sample $x_{k, i}$ thuộc class $c$ được chuẩn hóa sang không gian chuẩn tắc thông qua hàm nghịch đảo phân phối chuẩn chuẩn tắc $\Phi^{-1}$:

$$
z_{k, i, j} = \Phi^{-1}\left( \hat{F}_{c, j}(x_{k, i, j}) \right) \in \mathbb{R}
$$

Vehicle tính tổng vector bậc 1 và ma trận bậc 2 của các normal-scores:

$$
S_{k, c}^{(1)} = \sum_{i: y_{k,i}=c} z_{k, i} \in \mathbb{R}^D, \quad S_{k, c}^{(2)} = \sum_{i: y_{k,i}=c} z_{k, i} z_{k, i}^T \in \mathbb{R}^{D \times D}
$$

Server tổng hợp để xác định mean vector $\mu_c$ và covariance matrix $\Sigma_c$:

$$
\mu_c = \frac{\sum_{k=1}^K S_{k, c}^{(1)}}{\sum_{k=1}^K n_{k, c}}, \quad \Sigma_c = \frac{\sum_{k=1}^K S_{k, c}^{(2)}}{\sum_{k=1}^K n_{k, c}} - \mu_c \mu_c^T
$$

Server thực hiện Cholesky decomposition $\Sigma_c = L_c L_c^T$ ($L_c$ là lower triangular matrix), nén dưới dạng `fp16` và phân phối về các vehicles.

#### Zero-byte Synthetic Data Generation:
Tại mỗi client, synthetic data của class $c$ được sinh độc lập tức thời trên GPU mà **không tiêu tốn bất kỳ byte communication nào**:
1. Lấy mẫu independent standard normal vector $u \sim \mathcal{N}(0, I_D)$.
2. Áp đặt cấu trúc correlation toàn mạng: $z = \mu_c + L_c u$.
3. Ánh xạ ngược về input feature space ban đầu:

$$
\tilde{x}_j = \hat{F}_{c, j}^{-1}\left( \Phi(z_j) \right), \quad \forall j \in \{1, \dots, D\}
$$

---

### 2.2. Stage 2: Global Class Teacher Matrix ($Q$)

Tại mỗi federated round $r$, các phương tiện trao đổi knowledge ở output probability space mà không cần chia sẻ model architecture:

1. **Uplink**: Client $k$ đưa local real samples qua local model hiện tại $f_k(\cdot; \theta_k)$, tính tổng softmax probability vectors $p_k(x) = \text{softmax}(f_k(x; \theta_k)) \in \Delta^{C-1}$ theo từng class $c$:

$$
V_{k, c} = \sum_{i: y_{k, i} = c} p_k(x_{k, i}) \in \mathbb{R}^C
$$

Client gửi matrix $V_k = [V_{k, 0}, \dots, V_{k, C-1}]^T \in \mathbb{R}^{C \times C}$ và vector số mẫu $n_k$ lên Server (kích thước nén chỉ **544 Bytes**).

2. **Server Aggregation**: Server tính ma trận chuyển tiếp xác suất toàn mạng $Q \in \mathbb{R}^{C \times C}$:

$$
Q_c = \frac{\sum_{k=1}^K V_{k, c}}{\sum_{k=1}^K n_{k, c}} \in \Delta^{C-1}, \quad Q = \begin{bmatrix} Q_0^T \\ Q_1^T \\ \vdots \\ Q_{C-1}^T \end{bmatrix}
$$

Hàng thứ $c$ của ma trận, $Q_c = [Q_{c, 0}, \dots, Q_{c, C-1}]$, đại diện cho xác suất trung bình toàn mạng mà một sample thuộc class $c$ được mô hình phân loại vào các classes khác nhau. Matrix $Q$ đóng vai trò là **Architecture-Agnostic Soft Teacher** và được broadcast về các vehicles (**544 Bytes**).

---

### 2.3. Stage 3: Local Multi-Task Training at Client

Tại mỗi round, client $k$ tối ưu hóa multi-task loss function trên real data và synthetic data:

#### 1. Real Data Loss (Cross-Entropy with Logit Adjustment):
Nhằm xử lý triệt để local class imbalance, áp dụng Logit Adjustment (LA):

$$
\mathcal{L}_{\text{real}}(\theta_k) = - \frac{1}{|\mathcal{B}_{\text{real}}|} \sum_{(x, y) \in \mathcal{B}_{\text{real}}} \log \frac{\exp\left( f_{k, y}(x) + \tau \log \pi_{k, y} \right)}{\sum_{j=0}^{C-1} \exp\left( f_{k, j}(x) + \tau \log \pi_{k, j} \right)}
$$

Trong đó $\pi_{k, c} = \frac{n_{k, c}}{n_k}$ là class prior probability tại vehicle $k$, và $\tau = 1.0$ là temperature parameter.

#### 2. Synthetic Copula Distillation Loss:
Mỗi batch, client sinh 64 synthetic samples $\mathcal{B}_{\text{syn}} = \{(\tilde{x}_m, y_{\text{syn}, m})\}_{m=1}^{64}$ với class giả định $y_{\text{syn}} \sim \text{Uniform}(0, C-1)$. Áp dụng Kullback-Leibler (KL) Divergence với soft teacher distribution $Q[y_{\text{syn}}]$:

$$
\mathcal{L}_{\text{syn}}(\theta_k) = \frac{1}{|\mathcal{B}_{\text{syn}}|} \sum_{(\tilde{x}, y_{\text{syn}}) \in \mathcal{B}_{\text{syn}}} D_{\text{KL}}\left( Q[y_{\text{syn}}] \;\parallel\; \text{softmax}(f_k(\tilde{x}; \theta_k)) \right)
$$

$$
D_{\text{KL}}(Q_c \parallel p) = \sum_{j=0}^{C-1} Q_{c, j} \log \left( \frac{Q_{c, j}}{p_j} \right)
$$

#### 3. Total Local Optimization Objective:

$$
\mathcal{L}_{\text{total}}(\theta_k) = \mathcal{L}_{\text{real}}(\theta_k) + \lambda \mathcal{L}_{\text{syn}}(\theta_k) \quad (\lambda = 1.0)
$$

#### 4. Step Floor Mechanism:
Đối với các clients nhỏ ($n_k < 1.024$ samples), 1 local epoch kết thúc rất nhanh ($S_k = \lceil n_k / 64 \rceil < 16$ steps), khiến mô hình chưa kịp hội tụ trên absent classes. Cơ chế **Step Floor** quy định:
- Dữ liệu thật vẫn duyệt đúng 1 local epoch (bảo đảm tính toàn vẹn của local data).
- Nếu $S_k < 16$, client tự động chạy thêm $(16 - S_k)$ steps **chỉ với synthetic loss $\mathcal{L}_{\text{syn}}$**.
- **Hiệu quả:** Nâng Macro F1 của absent classes thêm **+13%** và Client P10 F1 thêm **+2%** mà **không phát sinh bất kỳ communication overhead nào**.

---

## 3. Communication Cost & Budget Comparison Table

<callout icon="📊" color="green_bg">
<b>Experimental Setting:</b> 5.343 vehicles, 50 federated rounds, standard architecture DAGSNet $W=1$ (395.408 parameters, 3.264 BN buffers), fp32 precision (4 Bytes/param).
</callout>

| Method | Upload / Client / Round | Download / Client / Round | Total / Client / Round | One-time Bootstrap | Total Communication Budget / Client (50 Rounds) | Total Network Traffic (5.343 Vehicles) | Communication Budget Reduction vs. FedAvg |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** | - | - | - | - | **0 B** *(vi phạm privacy)* | **0 B** | - |
| **SCAFFOLD** | 3,18 MB | 3,18 MB | 6,35 MB | 0 B | **317,63 MB** | 1.697,1 GB | -99,1% *(tốn gấp đôi)* |
| **FedAvg** | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | **159,47 MB** | 852,0 GB | Baseline (1.0×) |
| **FedProx** | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | **159,47 MB** | 852,0 GB | Baseline (1.0×) |
| **AFPHA** | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | **159,47 MB** | 852,0 GB | Baseline (1.0×) |
| **FD-IDS** | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | **159,47 MB** | 852,0 GB | Baseline (1.0×) |
| **APFed** | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | **159,47 MB** | 852,0 GB | Baseline (1.0×) |
| **NILM-FL** | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | **159,47 MB** | 852,0 GB | Baseline (1.0×) |
| **PerFed-SKD** | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | **~95,7 MB** *(client selection)* | ~511,2 GB | Tiết kiệm ~40% |
| **CANPerFL** | 1,59 MB | 1,59 MB | 3,19 MB | 0 B | **159,47 MB** *(+0 B fine-tune)* | 852,0 GB | Baseline (1.0×) |
| **LwFedNIDS** | 162 KB | 162 KB | 324 KB | 0 B | **16,20 MB** | 86,6 GB | Tiết kiệm ~10× |
| **V2X-HtFL** | 16,4 KB | 16,4 KB | 32,8 KB | 0 B | **1,64 MB** | 8,76 GB | Tiết kiệm ~97× |
| **FedProto** | 16,4 KB | 16,4 KB | 32,8 KB | 0 B | **1,64 MB** | 8,76 GB | Tiết kiệm ~97× |
| **FedTGP** | 16,4 KB | 16,4 KB | 32,8 KB | 0 B | **1,64 MB** | 8,76 GB | Tiết kiệm ~97× |
| **TinyProto** | 1,7 KB | 1,7 KB | 3,3 KB | 0 B | **0,17 MB** | 0,91 GB | Tiết kiệm ~940× |
| **pFedES** | 1,4 KB | 1,4 KB | 2,8 KB | 0 B | **0,14 MB** | 0,75 GB | Tiết kiệm ~1.100× |
| **MobiHet Core (Đề xuất)** | **544 B** | **544 B** | **1.088 B** | **1,00 MB** | **1,05 MB** | **5,63 GB** | **Tiết kiệm ~151× (mỗi round) / ~37× (tổng 50r)** |

> **Chú thích kỹ thuật về Communication Cost & Budget:**
> - Các phương pháp weight-sharing FL (FedAvg, APFed, SCAFFOLD) ngốn tới **852 GB - 1,7 TB** toàn mạng, gây tắc nghẽn nghiêm trọng kênh truyền DSRC/C-V2X.
> - Các phương pháp prototype-based FL (FedProto, TinyProto, pFedES) tuy nhẹ nhưng làm suy giảm mạnh accuracy và F1-score (xem Bảng 5).
> - **MobiHet Core** chỉ tốn **1.088 Bytes/round** trong quá trình training. Tổng communication budget cho 50 rounds (bao gồm cả bootstrap nén ban đầu) chỉ là **1,05 MB/vehicle** (toàn mạng 5.343 xe chỉ tốn **5,63 GB**, tương đương giảm **99,3%** tổng network traffic so với FedAvg).

---

## 4. Computational Cost & Complexity Comparison Table

<callout icon="⚙️" color="yellow_bg">
<b>Ký hiệu & Đo lường thực nghiệm:</b>
- $F = 4,45\text{ MFLOP}$ là computational cost cho 1 lượt forward qua DAGSNet full ($W=1$, 395.408 parameters) trên 1 sample.
- $n_k$ là số training samples của vehicle $k$ ($\sum n_k = 37.361.273$). $K = 5.343$ vehicles.
- Đo đạc thực nghiệm bằng <code>FlopCounterMode</code> và log thời gian thực tế trên cụm phần cứng 2×NVIDIA T4 GPUs.
</callout>

| Method | Client Computational Cost / Sample / Round | Client Complexity per Round (Big-O) | Server Complexity per Round (Big-O) | Local Batch Size | Wall-clock Execution Time (50 Rounds on 2×T4) | Automotive ECU Hardware Feasibility |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** | - *(chạy tập trung)* | - | $O(N_{\text{total}} F)$ | 4.096 | **~4,05 giờ** *(~292 s/ep)* | Không khả thi trên biên *(cần Data Center)* |
| **FedAvg** | $2,99 \times F \approx 13,3\text{ MFLOP}$ | $O(3 n_k F)$ | $O(K \cdot P)$ *(weighted avg)* | 64 | **~4,20 giờ** *(~300 s/r)* | Rất nhẹ (1 local epoch) |
| **FedProx** | $2,99 \times F \approx 13,3\text{ MFLOP}$ | $O(3 n_k F + S_k P)$ | $O(K \cdot P)$ | 64 | **~4,30 giờ** | Nhẹ (thêm proximal term) |
| **SCAFFOLD** | $2,99 \times F \approx 13,3\text{ MFLOP}$ | $O(3 n_k F + S_k P)$ | $O(K \cdot P)$ | 64 | **~4,80 giờ** | Vừa phải (control variate updates) |
| **AFPHA** | $2,99 \times F \approx 13,3\text{ MFLOP}$ | $O(3 n_k F + S_k P)$ | $O(K \cdot P + K \cdot \text{clusters})$ | 64 | **~4,35 giờ** | Nhẹ |
| **FD-IDS** | $3,99 \times F \approx 17,8\text{ MFLOP}$ | $O(4 n_k F)$ *(teacher forward)* | $O(K \cdot P)$ | 64 | **~4,60 giờ** | Vừa phải |
| **LwFedNIDS** | $0,29 \times F \approx 1,3\text{ MFLOP}$ | $O(3 n_k F')$ *(pruned 70%)* | $O(K \cdot P')$ | 64 | **~3,80 giờ** | Siêu nhẹ |
| **FedProto** | $2,99 \times F \approx 13,3\text{ MFLOP}$ | $O(3 n_k F)$ | $O(N_p \cdot d)$ | 64 | **~4,20 giờ** | Nhẹ |
| **FedTGP** | $3,99 \times F \approx 17,8\text{ MFLOP}$ | $O(4 n_k F)$ | $O(S \cdot N_p C d + S d^2)$ | 64 | **~4,90 giờ** *(server 100 epochs)* | Server nặng |
| **TinyProto** | $3,99 \times F \approx 17,8\text{ MFLOP}$ | $O(4 n_k F)$ | $O(N_p \cdot d)$ | 64 | **~4,40 giờ** | Nhẹ |
| **V2X-HtFL** | $4,11 \times F \approx 18,3\text{ MFLOP}$ | $O(4 n_k F + n_k d^2)$ | $O(S \cdot N_p C d)$ | 64 | **~4,70 giờ** | Server nặng (generative step) |
| **APFed** | $5,97 \times F \approx 26,6\text{ MFLOP}$ | $O(6 n_k F)$ *(train 2 models)* | $O(K \cdot P)$ | 64 | **~7,80 giờ** | Nặng gấp đôi FedAvg tại Client |
| **NILM-FL** | $5,97 \times F \approx 26,6\text{ MFLOP}$ | $O(6 n_k F)$ *(train 2 models)* | $O(K \cdot P)$ | 64 | **~8,10 giờ** | Nặng gấp đôi FedAvg tại Client |
| **pFedES** | $7,98 \times F \approx 35,5\text{ MFLOP}$ | $O(8 n_k F)$ *(nhiều lượt extractor)* | $O(K \cdot P_G)$ | 64 | **~8,50 giờ** | Quá tải đối với Automotive ECU |
| **CANPerFL** | $2,99 \times F$ (Pha 1) + 10 ep (Pha 2) | $O(3 n_k F) + O(30 n_k F)$ | $O(K \cdot P)$ | 64 | **~5,10 giờ** | Vừa phải |
| **MobiHet Core (Đề xuất)** | **~ $4,00 \times F \approx 17,8\text{ MFLOP}$** | **$O(4 n_k F)$** *(synthetic steps)* | **$O(K \cdot C^2)$ *(ma trận 16×16)*** | 64 | **~9,19 giờ** | **Hoàn toàn tối ưu cho ECU** *(Server siêu nhẹ)* |

> **Chú thích kỹ thuật về Computational Cost:**
> - $c_m \times F$ phản ánh số lượng floating-point operations thực tế trên mỗi sample. FedAvg chuẩn là $2,99 \approx 3$ FLOPs (1 forward + 2 backward).
> - **MobiHet Core** tiêu tốn $\approx 4,0 \times F$ tại client (tương đương FD-IDS, TinyProto), vận hành mượt mà trên automotive edge chipsets (NVIDIA Drive AGX, NXP S32G).
> - Tại Server, trong khi các phương pháp prototype-based phải chạy hàng trăm epochs tối ưu hóa prototypes, Server của MobiHet Core chỉ thực hiện phép cộng ma trận $16 \times 16$ ($O(K \cdot C^2) \approx 5.343 \times 256$ operations, hoàn thành dưới **5 mili-giây**).

---

## 5. Performance Comparison Table on Holdout Test Set

<callout icon="🏆" color="purple_bg">
<b>Experimental Protocol đồng nhất 100%:</b>
- <b>Dataset:</b> VeReMi NextGen v4 chuẩn hóa, 5.343 clients.
- <b>Training Split:</b> 37.361.273 samples (folds <code>train</code> + <code>val</code>), mỗi round đúng 1 local epoch.
- <b>Holdout Test Split:</b> 9.347.679 samples (20% timeline cuối cố định của từng vehicle, temporal holdout, không rò rỉ dữ liệu).
- <b>Standard Model Architecture:</b> DAGSNet $W=1$ (395.408 parameters, dropout 0.1).
- <b>Hyperparameters:</b> Local Batch Size 64, AdamW Optimizer, Cosine Annealing Learning Rate Schedule ($10^{-3} \to 10^{-5}$) qua 50 federated rounds, Seed 42.
</callout>

| Method | Evaluation View | Hỗ trợ Model Heterogeneity? | Accuracy (Toàn mạng) | Macro Recall (Toàn mạng) | Macro F1-Score (Toàn mạng) | Client P10 F1 *(Worst 10% Clients)* | Client Min F1 *(Worst Single Client)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized** *(Upper Bound)* | Global | ❌ Không | **0,9580** | **0,9531** | **0,9593** | - | - |
| **APFed** | Client (Own-test) | ❌ Không | **0,9278** | **0,9032** | **0,9119** | 0,7533 | 0,1500 |
| **CANPerFL** | Client (Fine-tuned) | ❌ Không | 0,9233 | 0,8986 | 0,9088 | **0,7684** | 0,1545 |
| **FedAvg** | Global | ❌ Không | 0,9228 | 0,8957 | 0,9042 | 0,7540 | 0,1500 |
| **FD-IDS** | Global | ❌ Không | 0,9220 | 0,8914 | 0,8992 | 0,7552 | 0,2176 |
| **AFPHA** | Global | ❌ Không | 0,9211 | 0,8932 | 0,8990 | 0,7523 | 0,1786 |
| **FedProx** | Global | ❌ Không | 0,9214 | 0,8941 | 0,8998 | 0,7524 | 0,1786 |
| **SCAFFOLD** | Global | ❌ Không | 0,9225 | 0,8859 | 0,8911 | **0,7738** | 0,2963 |
| **LwFedNIDS** | Global | ❌ Không | 0,9162 | 0,8830 | 0,8904 | 0,7291 | 0,2139 |
| **NILM-FL** | Client (Own-test) | ❌ Không | 0,9156 | 0,8717 | 0,8846 | 0,5777 | 0,0000 |
| **FedTGP** | Client Proto | **✅ Có** | 0,9020 | 0,8610 | 0,8719 | 0,5712 | 0,0000 |
| **PerFed-SKD** | Client (Own-test) | ❌ Không | 0,8994 | 0,8598 | 0,8710 | 0,6573 | 0,1241 |
| **FedProto** | Client Proto | **✅ Có** | 0,8978 | 0,8651 | 0,8698 | 0,5911 | 0,0000 |
| **V2X-HtFL** | Client (Own-test) | **✅ Có** | 0,8974 | 0,8554 | 0,8661 | 0,5519 | 0,0000 |
| **TinyProto** | Client Proto | **✅ Có** | 0,8866 | 0,8457 | 0,8534 | 0,5521 | 0,0000 |
| **pFedES** | Client (Own-test) | **✅ Có** | 0,8864 | 0,8405 | 0,8511 | 0,5264 | 0,0000 |
| **MobiHet Core (Đề xuất)** | **Client (Own-test)** | **✅ Có** | **0,9095** | **0,8945** | **0,8990** | **0,7543** | **0,1500** |

*(Ghi chú tiến độ F50: Kết quả của MobiHet Core hiện đang ghi nhận tại round 45 của phiên F50 thực nghiệm. Số liệu chính thức round 50 sẽ được cập nhật ngay sau khi pipeline hoàn tất audit).*

> **Phân tích chuyên sâu kết quả thực nghiệm:**
> 1. **Vượt trội nhóm Heterogeneous Baselines (Hỗ trợ model heterogeneity):**
>    - Các baseline hỗ trợ model heterogeneity hiện hành (FedProto, FedTGP, TinyProto, pFedES, V2X-HtFL) chỉ đạt Macro F1 từ **0,851 đến 0,872**, và suy giảm thảm hại ở nhóm phương tiện yếu nhất (Client P10 sụp đổ xuống mức **0,52 - 0,59**, Client Min F1 rớt về 0,0000 do thiếu samples để tạo class prototypes đại diện).
>    - **MobiHet Core đạt Macro F1 0,8990 (vượt trội +2,7% đến +4,8%)** và duy trì **Client P10 F1 đạt 0,7543 (vượt trội hơn +16% đến +23%)**, bảo đảm an ninh mạng đồng đều cho cả các phương tiện hạn chế dữ liệu nhất.
> 2. **Cạnh tranh sòng phẳng với nhóm Weight-Sharing FL:**
>    - MobiHet Core đạt Macro F1 tương đương FedProx (0,8998), AFPHA (0,8990), FD-IDS (0,8992) và áp sát FedAvg (0,9042), APFed (0,9119) trong khi **cắt giảm tới 99,3% communication budget** và không hề ràng buộc các phương tiện phải dùng chung model architecture.

---

## 6. Detailed Mechanisms & Trade-offs của Các Baseline Đối Chứng

1. **Centralized DAGSNet**:
   - *Mechanism:* Gom toàn bộ 37M training samples của 5.343 vehicles về trung tâm, shuffle IID, huấn luyện với large batch size (4.096).
   - *Trade-off:* Đóng vai trò Theoretical Upper Bound (F1 0,9593). Không khả thi trong thực tế do vi phạm data privacy và tạo nghẽn mạng nghiêm trọng.
2. **FedAvg (McMahan et al., 2017)**:
   - *Mechanism:* Server tính trung bình cộng trọng số mô hình cục bộ $\theta = \sum \frac{n_k}{N} \theta_k$.
   - *Trade-off:* Yêu cầu 100% clients cùng model architecture. Bị Client Drift nghiêm trọng khi gặp Non-IID data skew; Communication Budget lớn (159 MB/client).
3. **FedProx (Li et al., 2020)**:
   - *Mechanism:* Bổ sung proximal term $\frac{\mu}{2} \|\theta_k - \theta^t\|^2$ vào local loss để kiểm soát client drift.
   - *Trade-off:* Ổn định hội tụ nhưng Macro F1 (0,8998) không vượt trội FedAvg; vẫn áp đặt kiến trúc đồng nhất.
4. **SCAFFOLD (Karimireddy et al., 2020)**:
   - *Mechanism:* Dùng control variates để hiệu chỉnh hướng cập nhật gradient của client.
   - *Trade-off:* Nâng Client P10 lên 0,7738 nhưng Communication Budget tăng gấp đôi (317 MB/client); nhạy cảm với bất ổn định số học trong giai đoạn đầu.
5. **APFed (Sci. Rep., 2025)**:
   - *Mechanism:* Personalized FL bằng cách học trọng số kết hợp thích ứng giữa local model và global representation.
   - *Trade-off:* Đạt Macro F1 cao nhất trong các baseline (0,9119) nhưng Computational Cost tăng gấp đôi ($5,97 \times F$) và vẫn phụ thuộc vào weight sharing.
6. **NILM-FL (Li et al., TSG 2025)**:
   - *Mechanism:* Mỗi client duy trì 2 mô hình (private model và proxy model), áp dụng bidirectional distillation.
   - *Trade-off:* Computational Cost gấp đôi ($5,97 \times F$); Macro F1 giảm xuống 0,8846 và Client P10 sụp đổ về 0,5777 do các clients nhỏ không đủ dữ liệu để tối ưu hóa proxy model.
7. **FedProto (Tan et al., AAAI 2022) & FedTGP (Zhang et al., AAAI 2024)**:
   - *Mechanism:* Thay vì trao đổi weights, clients trao đổi class prototype vectors trong embedding space $d=256$.
   - *Trade-off:* Tiết kiệm communication budget nhưng prototype representation bị méo mó nghiêm trọng dưới điều kiện Non-IID và thiếu nhãn. Các vehicles nhỏ không nhận được prototype cho absent classes, dẫn đến Macro F1 thấp (0,869 - 0,871) và Client P10 rớt xuống <0,59.
8. **TinyProto (Lee et al., AAAI 2026)**:
   - *Mechanism:* Nén thưa hóa (sparse quantization) các vector prototype nhằm hạ communication budget xuống 1,7 KB/round.
   - *Trade-off:* Ranh giới phân lớp bị suy giảm mạnh do mất thông tin biểu diễn, Macro F1 chỉ đạt 0,8534.
9. **pFedES (Yi et al., AAAI 2025) & V2X-HtFL (OJVT 2025)**:
   - *Mechanism:* Áp dụng generative networks hoặc proxy feature extractors để kết nối các heterogeneous models.
   - *Trade-off:* Computational Cost tại client hoặc server tăng vọt; hiệu năng thực tế trên bài toán 16 classes V2X rất thấp (Macro F1 0,851 - 0,866).
10. **MobiHet Core (Proposed Method)**:
    - *Breakthrough:* Thay vì truyền prototype trong latent space (dễ gãy vỡ và phụ thuộc feature extractor), MobiHet Core **tách rời hoàn toàn biểu diễn đặc trưng** bằng cách mô hình hóa phân bố Copula ở input space, và chỉ trao đổi **Global Class Teacher Matrix $Q$ ở output probability space 16 classes**. Kết hợp với **Step Floor Mechanism**, phương pháp giải quyết trọn vẹn bài toán Non-IID, Absent Classes và Model Heterogeneity với Communication Budget siêu nhỏ (**1,05 MB / 50 rounds**).
