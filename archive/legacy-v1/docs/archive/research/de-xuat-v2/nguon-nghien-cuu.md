# Nguồn nghiên cứu cho đề xuất FL V2X đợt tiếp theo

Ngày đọc: 2026-10-11. **Chỉ nghiên cứu và thiết kế; chưa triển khai, chưa kiểm nghiệm các adaptation bên dưới.**
Đọc cùng [LESSONS](../../../../LESSONS.md), [P1](../probes/pilot-p1.md), [P2](../probes/pilot-p2.md),
[S-a](../probes/pilot-new-fl.md). Nguồn hỗ trợ thiết kế, không phải tuyên bố mới trong học thuật.
[Index bộ V2](README.md). N1/N2/N3 trong ghi chú nguồn tương ứng V2-A/V2-B/V2-C.

## 1. Ranh giới đã chốt

- Chủ dự án xác nhận trong phiên: **mọi mô hình học được đều DAGSNet**, gồm private, proxy, teacher và model phụ.
  Tree, learned kernel surrogate, GMM học nhiều thành phần không thuộc đề xuất chính.
- R2 vẫn ở thiết kế: giao diện tri thức cho private khác kiến trúc trong tương lai; hiện chưa chạy đa kiến trúc.
- Không record thật, latent từng record, gradient từng record rời xe. Tính gradient bằng record tại xe vẫn là FL.
- Thống kê/cập nhật chỉ được mở sau gộp cohort ít nhất 5 client, cùng schema và version. Ngoại lệ riêng hiện có:
  codebook synthetic khởi tạo từ nhiễu; không coi ngoại lệ này là quyền đọc riêng gradient/statistics.
- Paper ngoài không chứng minh hiệu quả IDS V2X, DAGSNet, 66 feature hay B0/B2.
- KME/DM chỉ trượt các adaptation tĩnh đã thử; không phủ định mọi density learning hoặc synthetic FL.

## 2. Tám nguồn gốc đã mở và đọc

### S1 — FetchSGD: sketch tuyến tính, residual tại server

**Rothchild và cộng sự, ICML 2020.** Đọc §3, thuật toán 1, §4 và giới hạn implementation §5.

Count Sketch tuyến tính: `S(Σᵢgᵢ)=ΣᵢS(gᵢ)`. Server giữ momentum và sai số nén trong sketch,
giải nén thành update sparse. Giảm phụ thuộc vào việc từng client quay lại mang residual riêng.
Cần cùng tọa độ model; bảo đảm có giả định heavy hitters/contraction, không áp cho mọi gradient.
Thuật toán gốc cho client tải chênh lệch tích lũy `wₜ−w₀`: sparse update mỗi bước không xóa chi phí bắt kịp lịch sử.
[Paper hội nghị](https://proceedings.mlr.press/v119/rothchild20a/rothchild20a.pdf).

### S2 — Error feedback: bảo toàn phần tín hiệu chưa nén được

**Karimireddy và cộng sự, Error Feedback Fixes SignSGD and other Gradient Compression Schemes, ICML 2019.** Đọc thuật toán 1, giả định compressor.

Với bước chưa nén `uₜ`, residual `eₜ`, compressor `C`:

\[
p_t=u_t+e_t,\quad q_t=C(p_t),\quad e_{t+1}=p_t-q_t.
\]

Tín hiệu bị nén mất đi được mang sang bước sau. Cơ chế sửa sai số nén, không tự sửa client drift,
nhãn sai hay bias chọn client. Không mang nguyên bảo đảm sang AdamW, local epoch, quantization và dropout.
[Paper hội nghị](https://proceedings.mlr.press/v97/karimireddy19a/karimireddy19a.pdf).

### S3 — FedKSeed: seed cố định và accumulator scalar

**Qin và cộng sự, Federated Full-Parameter Tuning of Billion-Sized Language Models with Communication Cost under 18 Kilobytes, ICML 2024.**
Đọc §4, thuật toán 1 và phụ lục D/G trong bản tác giả; xác nhận venue bằng trang PMLR.

Server giữ `K` seed cố định và tổng scalar theo seed. Hai lần tính loss với nhiễu trái dấu ước lượng directional derivative.
Model được dựng từ `w₀−ηΣⱼaⱼzⱼ`; bắt kịp model không cần replay mọi local step nếu accumulator còn dùng chung cơ sở.
Thí nghiệm fine-tune **LLM đã pretrained**; không chứng minh DAGSNet từ khởi tạo ngẫu nhiên học tốt trong subspace nhỏ.
Lịch sử scalar local trong bản gốc không tự có secure aggregation.
[Paper tác giả](https://arxiv.org/html/2312.06353v3), [venue ICML 2024](https://proceedings.mlr.press/v235/qin24a.html).

### S4 — FedKRSO: first-order random subspace, nhưng khác fixed scalar basis

**Yang, Wu, Guo, Sun & Gong, FedKRSO: Communication and Memory Efficient Federated Fine-Tuning of Large Language Models, 2026.**
Đọc bản tác giả v1 §IV, thuật toán 1–2; dùng nội dung preprint, không dựa vào trạng thái xuất bản để suy độ chắc chắn.

Với matrix weight `W`, chiếu gradient `G_B=∇F(W)Pᵀ`; cập nhật `W←W−ηG_BP`.
Chain rule tại `B=0` của `F(W+BP)` tính gradient không cần biểu diễn mọi full-gradient tensor.
Accumulator là **matrix** `d_m×r`, không một scalar mỗi seed. Seed pool đổi mỗi round;
model dựng nối tiếp `Wᵗ=Wᵗ⁻¹+ΣₖBₖᵗPₖᵗ⁻¹`.
Fixed scalar basis suốt run là adaptation khác; không kế thừa trực tiếp luận cứ unbiased khi projection độc lập gradient.
[Paper tác giả](https://arxiv.org/html/2602.03019v1).

### S5 — Noise-contrastive estimation: học phân biệt thật và noise

**Gutmann & Hyvärinen, Noise-contrastive estimation: A new estimation principle for unnormalized statistical models, AISTATS 2010.**
Đọc §2 và phần giả định mô hình/noise.

NCE dùng logistic discrimination giữa dữ liệu và noise để ước lượng model mật độ chưa chuẩn hóa.
Chênh log-density quyết định odds phân biệt. Tính nhất quán trong paper có điều kiện về họ mô hình,
noise và giới hạn mẫu; không phải bảo đảm cho DAGSNet finite-sample, FedAvg hay ratio bị clip.
Nguồn hỗ trợ nguyên lý data-versus-noise; proposal ratio-weighted synthetic là adaptation riêng, không nguyên thuật toán paper.
[Paper hội nghị](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf).

### S6 — Secure aggregation: giao thức có chi phí và dropout

**Bonawitz và cộng sự, Practical Secure Aggregation for Federated Learning on User-Held Data, NIPS workshop 2016.**
Đã đọc abstract chính thức; không coi đây là audit đầy đủ giao thức hay dùng overhead trong nguồn dự báo hệ thống này.

Mục tiêu tính tổng, không lộ giá trị riêng ngoài thông tin suy từ tổng; xử lý client rớt giữa giao thức.
Số mở rộng giao tiếp trong nguồn áp cho cohort/vector lớn cụ thể, không chuyển thẳng sang cohort 5 và payload vài kB.
Đây là bản workshop 2016, không gán nhầm bản CCS 2017.
[Trang chính thức Google Research](https://research.google/pubs/practical-secure-aggregation-for-federated-learning-on-user-held-data/).

### S7 — Domain generalization phải có quy tắc chọn model

**Gulrajani & Lopez-Paz, In Search of Lost Domain Generalization, ICLR 2021.** Đọc phần model selection và thảo luận DomainBed.

So sánh cần đồng nhất huấn luyện, kiến trúc và cách chọn model; thuật toán thiếu quy tắc chọn HP còn chưa hoàn chỉnh.
ERM triển khai cẩn thận là đối chứng mạnh. Không suy từ DomainBed rằng domain invariance chắc chắn vô dụng cho V2X.
[Paper tác giả](https://arxiv.org/pdf/2007.01434).

### S8 — JEM: nguồn đối chiếu, chưa chọn triển khai

**Grathwohl và cộng sự, Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One, ICLR 2020.**
Đọc §4–6 và phụ lục H.

Logits classifier định nghĩa joint energy; cần huấn luyện thêm objective mật độ, không chỉ dùng classifier CE cũ để sinh mẫu.
Paper dùng SGLD/contrastive divergence; tác giả nêu bất ổn và chi phí sampling. Experiments chính bỏ BatchNorm;
footnote nói đã huấn luyện được với BN sau đó, không có nghĩa cấu hình DAGSNet hiện hành đã được xác nhận.
Giữ làm hướng dự phòng học thuật. N3 chính dùng data/noise logistic, không gọi nó JEM và không dùng SGLD.
[Paper tác giả](https://arxiv.org/html/1912.03263v3).

## 3. Suy luận thiết kế mới — chưa có kết quả

Phần dưới là suy luận/đề xuất riêng cho dự án. Kích thước chỉ là phép tính payload, **không phải đo wire hay chứng nhận R1**.

### N1. DAGSNet proxy với Count Sketch và residual RSU

Giữ proxy DAGSNet chung; private DAGSNet riêng. Sketch gradient/cập nhật proxy của nhiều xe,
gộp cohort rồi cập nhật residual RSU. Private nhận tri thức qua KD/DML tại xe.
Đây là mở rộng truyền thông H1/H6, không gọi thành họ FL hoàn toàn khác.

Chọn rõ gradient tại cùng snapshot hay local-epoch delta. FetchSGD làm cơ sở nhánh gradient;
local-epoch delta là adaptation khác với bias và staleness riêng.
Cùng hash/seed/schema trên mọi xe giúp secure sum. Local top-k index riêng không tự cộng được
bằng cùng dense secure-aggregation schema; cần protocol khác/padding và tính đủ byte.

Ví dụ `3×256` counter fp32: **3.072 B upload**. Sparse downlink `k=256`, uint32 index + fp16 value:
**1.536 B**, chưa scale, checksum, buffers, setup, frame/ACK. DAGSNet 395.024 tham số nên index uint16 không đủ toàn model.
Không suy từ tổng nhỏ rằng sketch collision thấp hoặc hai chiều cùng vừa một contact B0.

Residual trừ **giá trị giải mã đã commit**, không float trước codec. Mất ACK cần idempotency;
frame dở trả phí, không commit. Mỗi update gắn base version; xe nhận đủ delta chain hoặc snapshot mới.
BN buffers cũng là trạng thái; seed khởi tạo không thay BN đã học. Residual RSU không khôi phục gradient chưa upload hay cohort đã hủy.

### N2. DAGSNet trong subspace tham số sinh bằng seed

Đặt `θ(a)=θ₀+Za`, `Z∈R^{395024×K}` sinh từ seed chung, `a∈R^K`.
Mọi weight được tái tạo đúng hình dạng DAGSNet; basis chỉ là tham số hóa tối ưu, không model học thứ hai.
Private học toàn model; chỉ proxy bị ràng buộc subspace. R2 tương lai vẫn qua output của proxy.

**First-order:** chain rule cho gradient hệ số `∇ₐL=Zᵀ∇θL`.
**Zeroth-order:** chênh loss hai phía theo `zⱼ` ước lượng đạo hàm tương ứng, với cùng minibatch và randomness.
So sánh FO/ZO cùng basis giúp phân biệt thiếu sức chứa với nhiễu ước lượng gradient.
Đây là đề xuất fixed-basis riêng, lấy ý tưởng accumulator từ FedKSeed và projected gradient từ FedKRSO.
Không nói fixed `ZZᵀg` là unbiased full gradient trong mọi bước: `θ` phụ thuộc chính basis đã cố định.

`K=512`, coefficient fp32: **2.048 B** mỗi chiều; `K=1024`: **4.096 B** mỗi chiều.
Full vector accumulator cố định chiều có thể secure-aggregate; phải cộng đủ trọng số/mẫu số và kiểm overflow của wire representation.
Seed/schema/count/setup/BN không miễn phí. Đổi basis giữa run phá tính gọn lifetime accumulator;
muốn đổi phải trả chi phí old-basis state/snapshot và ghi version.

Rủi ro chính: fine-tuning pretrained LLM khác học IDS từ đầu; basis ít chiều chặn ranh giới tốt,
FO vẫn cần backward, ZO cần nhiều loss evaluations, dense `Za` tăng compute. Chính sách BN/dropout phải làm model reconstructible:
không giả định vector coefficient chứa running mean/variance hoặc RNG state. Khóa cách này trước pilot.

### N3. DAGSNet học density ratio để sửa trọng số synthetic liên tục

Giữ proposal Gaussian theo lớp `q_y` từ H2 aggregate đã có; không học thêm generator khác kiến trúc.
DAGSNet 16 output giờ mang ý nghĩa log-ratio từng lớp. Với cân bằng real/noise cho lớp `y`, objective lý tưởng:

\[
\mathcal L_y(r)=-\mathbb E_{x\sim p_y}\log\sigma(r_y(x))
-\mathbb E_{x\sim q_y}\log[1-\sigma(r_y(x))].
\]

Khi các mật độ tồn tại trên cùng measure và support phù hợp, nghiệm population không bị hạn chế:
`r_y^*(x)=log[p_y(x)/q_y(x)]`. Nếu tỷ lệ noise:real là `ν`, logit classifier tối ưu bằng
`log[p_y/(νq_y)]`; bỏ qua offset `logν` làm sai ratio.

Private sinh Gaussian mới mỗi bước, lấy weight `exp(r_y)` đã clip/self-normalize, rồi CE trên nhãn lớp Gaussian.
Không dùng soft label teacher yếu để thay nhãn như H5. Toàn private backbone nhận gradient; bank không cố định như S-a.
Mô hình ratio được học FL trên real tại xe và noise tại xe; chỉ aggregate updates rời client.

**Điều kiện support là rủi ro khoa học lớn.** Các feature cờ nhị phân/điền 0 và quan hệ xác định có thể làm phân phối real
nằm trên tập có xác suất bằng 0 dưới Gaussian liên tục. Khi `p_y` không tuyệt đối liên tục theo `q_y`,
đẳng thức importance weighting không áp dụng. Classifier có thể chỉ học dấu hiệu “trên manifold hay không”,
không học ranh giới hữu ích; clipping không khắc phục support bị thiếu.
Không tự thêm jitter/dequantization/đổi feature để né lỗi: việc này cần thiết kế riêng và giữ protocol 66 feature.
Ngay cả khi xem như approximation finite-resolution, phải nói rõ không phải phục hồi chính xác `p_y`.
Thiết kế chi tiết [V2-C](c-density-ratio.md) đã đề nghị mục tiêu `p_y^τ` có perturbation chỉ tại local và giữ raw input private;
đây là thiết kế augmentation cần khóa trước triển khai, chưa được chạy và không biến `p_y^τ` thành raw `p_y`.

**Trọng số FL theo lớp:** `p_y=Σᵢ(n_iy/N_y)p_iy` cho pooled empirical distribution.
Cân bằng 16 lớp tại từng xe rồi FedAvg bằng `n_i` thường không thực hiện cùng objective này.
Cần khóa sampling/loss weights và nêu rõ mục tiêu là pooled-class hay client-balanced;
count lớp vẫn chỉ mở tổng cohort, không gửi histogram riêng. Local nhiều bước còn gây lệch khỏi exact pooled gradient.

Đánh giá không chỉ data/noise accuracy. Bắt buộc private F1, ESS theo lớp
`(Σw)^2/Σw²`, tỷ lệ clip, weight quantiles, loss real/noise tách riêng, receiver holdout.
Data/noise accuracy gần hoàn hảo có thể là dấu hiệu support tách rời, không phải teacher tốt.
Đối chứng tối thiểu: Gaussian unweighted; ratio không học; FL ratio; ratio chỉ từ chính xe.
Nếu FL ratio không thắng chính Gaussian, không tăng clipping range hoặc kéo dài để tìm thắng.
N3 phải tải model ratio và thống kê proposal; không có claim giải R1 chỉ vì private sinh samples tại chỗ.

## 4. Yêu cầu chung trước triển khai tương lai

1. **RSU chỉ thấy tổng cohort.** Không upload model riêng cho ensemble, energy của từng client hay scalar gradient riêng
   rồi gọi đó là secure aggregation. Ngưỡng 5 không đồng nghĩa DP; truy vấn gần giống khác một client có thể lộ qua lấy hiệu.
2. **Contact hữu hạn, protocol nhiều bước.** Tính key/setup, masked upload, recovery, cohort commit, hai chiều, retransmission,
   prior, scaler/statistics đổi version, catch-up. Mô phỏng oracle tổng phải được ghi rõ chưa chứng minh wire security/deadline.
3. **Receiver shift giữ ưu tiên.** Group/temporal holdout; không chọn HP bằng test chuẩn. Báo benign, positionMirroring,
   đủ 10 metric, proxy/ratio diagnostics, private `head` và `gp`. Không kết luận từ tổng training loss.
4. **Chỉ Markdown trong phiên này.** Không code, notebook hay run mới; không Git/DVC. Khi triển khai sau này:
   W&B online trước train theo quy định đã chốt; local chỉ kiểm tra nhỏ, pilot nặng Kaggle.

## 5. Bản đồ bằng chứng

| Hướng | Nguồn hỗ trợ cơ chế | Điều chưa được nguồn chứng minh |
|---|---|---|
| N1 sketch proxy | S1 + S2 | DAGSNet, B0, local epoch, cohort dropout, chi phí BN/catch-up |
| N2 seed subspace | S3 + S4 | Học từ đầu với K nhỏ; fixed-basis FO giữ chất lượng; model reconstructible qua BN |
| N3 ratio Gaussian | S5 | Raw V2X support; FedAvg học đúng pooled ratio; clipped weights cải thiện private IDS |
| Mọi hướng | S6 + S7 | Secure protocol thật vừa deadline; tổng quát receiver mới |
| JEM dự phòng | S8 | Không phải phương án chính; chưa chốt sampling/BN/compute |

Không phương án nào có quyền ghi “đạt R1/R2” bằng số đo. Không có thao tác thủ công bắt buộc để hoàn thành tài liệu nghiên cứu này.
