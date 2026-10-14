# V2-C — DAGSNet học tỷ số mật độ qua FL, điều chỉnh trọng số Gaussian liên tục

Ngày 2026-10-11. **Đề xuất tri thức mới, chưa code/đo.**
[Index](README.md) · [hợp đồng chung](thiet-ke-chung.md) · [nguồn NCE S5](nguon-nghien-cuu.md).

## 1. Vấn đề muốn sửa

H2 Gaussian liên tục là mốc mạnh, nhưng quan hệ phi tuyến của quần thể có thể nằm ngoài moment bậc hai.
KME/DM tĩnh chưa chứng minh lợi tối ưu synthesis; teacher soft-label yếu của H5 lại làm kém nhãn lớp Gaussian.
Đề xuất: giữ Gaussian sinh mẫu mới liên tục và nhãn lớp của nó, nhưng **học điểm nào trong Gaussian đáng
đóng góp nhiều/ít vào loss private**, bằng DAGSNet được huấn luyện liên client trên dữ liệu thật tại xe.

Không dùng prototype, không thêm generator MLP, không tối ưu bank 128 điểm hoặc encoder ngẫu nhiên.
Model phụ vẫn đúng DAGSNet 66→16. Tri thức truyền là tham số của một hàm tỷ số mật độ,
không phải mẫu local hoặc soft label từ ensemble local yếu.

[NCE của Gutmann–Hyvärinen](https://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf) cung cấp nguyên lý
học mật độ bằng phân biệt dữ liệu với noise. Cách kết hợp FL, H2 và weighted auxiliary ở đây là thiết kế riêng;
không có kết quả NCE gốc nào chứng minh nó tăng F1 IDS hoặc đáp ứng B0.

## 2. Rủi ro support phải giải thích trước công thức

66 feature gồm cờ nhị phân, one-hot, cụm điền 0, đại lượng phụ thuộc nhau.
Phân phối record thật có thể nằm trên các mặt có xác suất 0 dưới Gaussian liên tục.
Vì vậy công thức “p thật/q Gaussian” không luôn tồn tại; classifier data/noise hoàn hảo có thể chỉ nhận ra
cờ không nguyên hoặc quan hệ hình học sai. **Clip weight không sửa được support bị thiếu.**

Thiết kế đề nghị xác định **mục tiêu có làm trơn**, thay vì giả vờ học chính xác mật độ raw:

\[
q_y=\mathcal N(\mu_y,\operatorname{diag}(\sigma_y^2)),\quad
x^+=x+\tau\operatorname{diag}(\sigma_y)\epsilon,\quad
\epsilon\sim\mathcal N(0,I).
\]

`μ_y, σ_y` từ H2 cohort aggregate, variance floor10^-4; `τ=0,05` là giá trị đề nghị cần khóa trước pilot.
Phân phối dương là `p_y^τ`, kernel-smoothed empirical distribution, không phải raw `p_y`.
Do noise có variance dương trên cả66 chiều, p_y^τ và q_y có density chung; điều kiện support lý thuyết được định nghĩa.

**x và x⁺ đều chỉ tồn tại ở xe sở hữu record. Không upload x⁺, kể cả dưới tên synthetic.**
Đây là perturbation local để học tỷ số, không thuộc ngoại lệ codebook khởi tạo từ noise được gửi RSU.
Private vẫn đọc raw feature đã chuẩn hóa như protocol, không thêm/cắt/đổi66 cột và không bật M4b.
Feature cờ trong x⁺ có thể không còn rời rạc; đó là surrogate huấn luyện model tỷ số, không giả làm bản tin V2X hợp lệ.

Làm trơn gây bias, có thể xóa ranh giới hẹp; τ không được tune bằng official test.
Nếu chủ dự án triển khai hướng này, phải chốt rõ augmentation local và mục tiêu p^τ trong manifest.
Đây chưa phải quyết định thay preprocessing dataset.

## 3. Objective FL đúng theo lớp

Gọi DAGSNet tỷ số là `r_φ(x)∈R^16`. Dùng một negative Gaussian cùng lớp cho mỗi positive local, tỷ lệ1:1:

\[
\mathcal L_y(\phi)=
E_{x^+\sim p_y^\tau}\operatorname{softplus}(-r_{\phi,y}(x^+))+
E_{z\sim q_y}\operatorname{softplus}(r_{\phi,y}(z)).
\]

Ở nghiệm population không hạn chế, `r*_y=log(p_y^τ/q_y)`.
Không dùng `softmax(r)` để lấy weight: đây là16 logistic tasks theo lớp, không classifier16 lớp thông thường.
Nếu sau này đổi noise:real thành ν thì phải xử lý offset logν và trọng số noise; không chỉ tăng số negative.

Mục tiêu chọn lớp đều nhưng pooled samples trong từng lớp:

\[
\mathcal L=\frac1{16}\sum_y\mathcal L_y,\qquad
p_y=\sum_i\frac{n_{iy}}{N_y}p_{iy}.
\]

Không cân bằng lớp riêng từng client rồi FedAvg n_i mà gọi đó là cùng mục tiêu.
Một cách thực hiện được đề nghị ở U full participation:

- Xe lấy minibatch theo phân phối local tự nhiên, positive kèm một Gaussian negative cùng nhãn.
- Loss mỗi cặp nhân `w_y=1/(16π_y)`, với `π_y=N_y/N` nhận từ aggregate train đã trả phí.
- Aggregate gradient bằng n_i/N cho ra đúng objective trên; dùng local epoch rồi average delta chỉ là xấp xỉ FedAvg,
  không còn đồng nhất exact pooled gradient. Cần ghi sự khác biệt này khi diễn giải kết quả.

Ở C, cohort hữu hạn/lệch nguồn làm target khác toàn quần thể; π của proposal ban đầu không tự khử selection bias.
Không mở histogram n_iy riêng ở RSU. Lưu roster/version và báo độ phủ quần thể, không dùng count toàn bộ chưa truyền
để sửa phương pháp từ evaluator.

## 4. Quá trình học và dùng tri thức

1. **Bootstrap H2.** Client gửi count/moment gộp; nhận q và π. U dùng20client như fixture;
   C chỉ dùng aggregate đã hoàn tất, ≥5 client. Khóa q thành một proposal version cho pilot, không đổi giữa chừng.
2. **Học tỷ số local.** DAGSNet ratio clone từ φ_v, một epoch trên positive local có làm trơn và negative q mới.
   AdamW wd0, LR cosine1e-3→1e-5/reset, loss fp32/clip1 là điểm khởi đầu đề nghị.
3. **Gộp FL.** Chỉ aggregate delta φ từ cohort cùng φ_v/q_version; private weights không upload.
4. **Nhận tri thức.** Client chỉ dùng φ_v hoàn chỉnh. Chưa nhận: weight=1, hành vi về H2 diag đã kích hoạt;
   chưa nhận H2: về local. Không lấy model ratio hiện tại của RSU qua con trỏ bộ nhớ simulator.
5. **Học private.** Mỗi batch private sinh Gaussian mới tại xe, chạy ratio snapshot eval/no-grad để có weight,
   train toàn DAGSNet private bằng CE real + weighted CE synthetic. Ratio không lấy gradient từ private.

Để hàm ratio dùng chung không khác vì BN, V2-C đề nghị chính sách BN khác A/B:
**canonical calibration bằng Gaussian q public**, 4.096 điểm cân bằng lớp từ seed/version cố định,
thứ tự batch cố định; không record thật. Calibrate sau khi dựng φ snapshot, rồi freeze BN/dropout trong epoch ratio
và khi chấm weight. All ratio arms dùng cùng recipe. Affine BN vẫn train.
Các buffers được tính từ φ, q và recipe đã phân phối, không truyền miễn phí thống kê local.
Compute calibration phải tính vào runtime; recipe/seed/version có byte bootstrap.
Nếu tái dựng buffers giữa thiết bị không đạt tolerance preflight, phải truyền state canonical và cập nhật ledger,
không tiếp tục dùng công thức byte thiếu buffer.

Đây là policy mới cần kiểm; việc Gaussian calibration không đại diện real có thể làm ratio learner yếu.
Không lấy chất lượng proxy BN-local của A/B hoặc H1 để bảo đảm cho ratio model.

## 5. Weight ổn định và giới hạn ý nghĩa

Đề nghị auxiliary256 điểm/batch,16 điểm/lớp, lớp lấy đều. Với mỗi lớp:

\[
u_b=\min\{\exp(\operatorname{clip}(r_y(z_b),-10,10)),10\},\qquad
\bar u_b={u_b\over\frac1{m_y}\sum_{j:y_j=y}u_j}.
\]
\[
L_{private}=CE(f_i(x),y)+\lambda\frac1{B_s}\sum_b\bar u_b\,CE(f_i(z_b),y_b),\quad\lambda=1.
\]

Giữ tổng đóng góp mỗi lớp để contrast không chỉ do tăng/giảm liều loss.
Clip và self-normalize tạo bias; đây là weighted surrogate, không còn estimator không chệch của risk thật.
Chọn nhãn y từ q_y như H2, không thay bằng nhãn model ratio.
Nhãn này là nhãn của proposal, không đảm bảo Gaussian rơi đúng vùng class thật.

Nếu q hầu như không sinh vùng quan trọng, reweight không tạo được vùng đó từ không có mẫu.
Mẫu gần manifold trong66 chiều có thể hiếm đến mức ESS sụp; p^τ có density hợp lệ không làm Monte Carlo tự hiệu quả.
Không tăng auxiliary batch vô hạn để che chi phí, không dùng record thật từ client khác làm proposal.

## 6. R1: phương án chất lượng cần carrier

Pilot U dùng dense aggregate để cô lập câu hỏi “ratio có hơn Gaussian không”. Chỉ trainable φ có395.024 phần tử:

- Upload fp32 delta +count uint64 +header32: **1.580.136 B/client/version**.
- Download fp16 weights +header32: **790.080 B/client/version**.
- Còn bootstrap q/prior, recipe/seed, frames/ACK, secure protocol và codec metadata.

Tính fp16 thực rồi dùng model decode, không chỉ ghi byte fp16 trong report. BN tái dựng như §4; nếu chuyển sang truyền BN,
cộng state thật. Dense C **không phải giải pháp B0**; ở B2 vẫn cần nhiều contact/chiều trước cohort update.
RSU cũng lấy weights đã decode làm canonical φ cho version mới; delta tiếp theo phải trừ đúng base đó,
không để RSU dùng bản fp32 khác bản client nhận rồi vẫn gắn cùng version.

Chỉ nếu V2-C qua U và V2-A hoặc B qua carrier gate, mới đề xuất ma trận2×2:
`dense / compressed carrier × unweighted / ratio-weighted auxiliary`.
Không ghép cả hai ngay rồi gán gain cho ratio hoặc compressor. Compression có thể thay calibration/ratio tails
dù F1 classifier của carrier cũ ổn, nên phải kiểm ESS và ratio riêng sau nén.
Nếu không carrier nào giữ được ratio hữu ích dưới deadline thì V2-C dừng ở kết quả chất lượng, chưa đáp ứng R1.

Tương lai R2: private bất kỳ kiến trúc nhận66 feature và weighted CE; không cần hidden dimension chung.
Hiện mọi private và ratio network đều DAGSNet; không có kết quả heterogeneous architecture.

## 7. Đối chứng, diagnostic và cổng đề nghị

| arm | mục đích |
|---|---|
| local | mốc IDS không tri thức |
| H2 diag continuous | baseline chính, cùng q/auxiliary batch/lịch/prior |
| ratio-own | ratio chỉ học positive của chính xe, q vẫn là H2 gộp chung |
| ratio-FL | ratio học aggregate, weight gắn đúng synthetic sample |
| ratio-FL-shuffle | hoán vị weight trong từng class batch, giữ cùng trọng số/liều nhưng phá quan hệ weight–x |

H2 control có thể chạy ratio learner như compute placebo nhưng đặt weight=1; báo cả chi phí với/không placebo.
Ratio-own dùng cùng augmentation, codec/cap/lịch phiên bản giả lập như ratio-FL để tách phần liên client.
Không gọi toàn bộ lợi ratio-own là local-only: q của nó đã chứa tri thức H2; contrast FL−own mới đo thêm từ ratio gộp.

Trên diagnostic draw độc lập từ q,4.096 điểm/lớp, đo ESS:

`ESS_y=(Σu)^2/Σu²`, báo `ESS_y/4096`, tỷ lệ clip, quantiles của logit/weight, negative loss và positive loss riêng.
Đề nghị yêu cầu median ESS fraction≥0,25, từng lớp≥0,10 cả hai seed.
Logistic accuracy gần1 **không** là cổng thành công; có thể là tách support/shortcut quá dễ.
Số diagnostic không thay cổng IDS và không được dùng sửa cap sau khi thấy F1.

Cổng riêng ở U, cả hai seed:

- `ratio-FL group gp − H2 diag group gp ≥0,015`.
- `ratio-FL − ratio-own ≥0,01` và `ratio-FL − ratio-FL-shuffle ≥0,01`.
- Cổng chung local/old/benign, ESS guard; `head` phải báo để phát hiện prior mismatch.
- positionMirroring là diagnostic định trước, không dùng nó một mình cứu macro-F1 trượt.

Nếu thắng H2 nhưng không hơn shuffle: có thể chỉ là thay noise/liều weighting, chưa chứng minh học được vị trí
quan trọng trong distribution. Nếu thắng own nhưng thua H2: tri thức gộp có tác dụng trong họ ratio,
nhưng vẫn không đủ lý do triển khai. Nếu F1 qua nhưng ESS fail: chưa chấp nhận độ ổn định theo thiết kế này.

## 8. Phần mới cần xác nhận trước triển khai

τ làm trơn local, cap10, normalization theo class, BN trên q và cohort target phải được review/khóa trước code pilot.
Các giá trị trên là đề nghị có thể bác bỏ, không phải sweep tối ưu sẵn. Không có thao tác tay cần làm hiện tại.

Đóng góp tiềm năng cần chứng minh: học FL phần điều chỉnh phi tuyến của Gaussian với toàn model DAGSNet,
truyền được cho private và giữ được dưới deadline. Không gọi đây là “phục hồi dữ liệu thật”, không có bảo đảm DP,
không bảo đảm generalization khỏi shortcut receiver chỉ vì có density ratio.
