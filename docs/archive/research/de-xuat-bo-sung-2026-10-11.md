# Đề xuất bổ sung — condensation dữ liệu, sketch phi tuyến, contact hữu hạn

Ngày 2026-10-11. **S2/S3 đã pilot hai seed và trượt cổng**: [kết quả S-a](probes/pilot-new-fl.md).
S1 vẫn là thiết kế chưa pilot. Phần dưới giữ lập luận đề xuất trước chạy; không phải đóng góp đã chứng minh.
Nền: [P1](probes/pilot-p1.md), [H1–H6](de-xuat-huong-moi.md), [kế hoạch](pilot-plan.md),
[bài học](../../../LESSONS.md), [archive](../legacy/kien-thuc-ke-thua.md).
Private và proxy phân loại vẫn DAGSNet theo quyết định hiện hành; R2 mới kiểm ở thiết kế.
Local chỉ smoke; pilot nặng chạy Kaggle. Không truyền record local, kể cả đối chứng.
**Chốt bổ sung trong phiên:** được thoát khỏi H1–H6; RSU được đọc codebook **tổng hợp** do client học, không record thật.
Đây là ngoại lệ đã được chủ dự án chấp nhận cho giả định RSU chỉ đọc tổng ở Q4; không cần giả vờ codebook riêng tương thích secure aggregation.

## 1. Chọn hướng theo nguyên nhân

| bằng chứng đã có | cơ chế cần thêm | ứng viên |
|---|---|---|
| H2 hạng 4 giữ phần lớn lợi full, nhưng đường lên vẫn 145.792 B/client | nén ngay tại client bằng sketch cộng được; dùng từng lớp khi đủ | **S1: Nyström + thống kê từng lớp** |
| Gaussian không học được positionMirroring trong khi FedAvg tăng rõ | học trực tiếp ít điểm tổng hợp có ích qua phân phối phi tuyến | **S2: federated dataset condensation** |
| Tâm lớp không đủ; latent trôi; teacher yếu làm giảm lợi H2 | gửi mô tả phân phối phi tuyến cố định rồi tổng hợp input cho toàn model | **S3: Fourier distribution sketch** |

S2 và S3 là hai hướng mới so với H1–H6: một hướng trao đổi điểm tổng hợp **được tối ưu**, một hướng trao đổi sketch phân phối rồi sinh ở nơi nhận.
S1 chỉ là **mở rộng giao thức H2/H6**, giữ làm dự phòng về byte, không gọi là họ FL mới.
Cả ba là thiết kế thích nghi cho MobiHet, chưa phải đóng góp mới đã được chứng minh trong literature.
Ưu tiên pilot lần này: **S2 và S3**. Không mở lại head-only, prototype trung bình hoặc DS-FL với ensemble local yếu.

## 2. S1 — Sketch covariance cộng được + phát từng lớp

**Vấn đề.** `Synth`/`ChanSynth` hiện tính full covariance ở đường lên cho cả `lr4`; giảm byte chỉ ở đường xuống.
P1 cho `lr4` +0,091/+0,092 nhưng full upload khiến nó khó hoạt động sớm ở B0.
Nguồn cài đặt: [pilot.py](../../../core/pilot_runtime.py), [p2.py](../../../mh/p2.py).

**Cơ chế đề xuất.** Cố định ma trận ngẫu nhiên trực chuẩn `Ω ∈ R^(66×r)` bằng seed/schema chung.
Mỗi lớp, mỗi client chỉ gửi các tổng cộng được:

```text
n = số mẫu; s = Σx; q = Σ(x⊙x); Z = Σ[x(xᵀΩ)]
μ = s/n
Y = Z/n − μ(μᵀΩ)                  # covariance toàn cục nhân Ω
B = sym(ΩᵀY)
U = Y B^(†/2)                    # xử lý trị riêng âm/nhỏ do lượng tử hóa
sd² = max(q/n − μ² − diag(UUᵀ), ε)
x̃ = μ + Uz_r + sd⊙z_66
```

RSU cộng `n,s,q,Z` trước khi khôi phục; **không cộng eigenvector/factor của từng client**.
Đây là ứng dụng Nyström từ sketch tuyến tính `AΩ`; công thức và tính cộng dồn có trong
[Tropp et al., Fixed-Rank Approximation of a PSD Matrix from Streaming Data, §2–3](https://arxiv.org/pdf/1706.05736).
`r=4` không có oversampling là lựa chọn tiết kiệm để kiểm nghiệm, không hưởng mặc nhiên bảo đảm xấp xỉ của cấu hình sketch rộng hơn trong paper.

Tách object theo `(class, schema, version, cohort)`; tiếp tục fragment qua contact.
Chỉ phát lớp khi đủ cohort, ví dụ ≥5 client có đóng góp, và chỉ dùng sau khi toàn object lớp đến.
Kích hoạt từng lớp coherent được; không ghép `μ`, `U`, `sd` của các version khác nhau.
Thứ tự vòng tròn cố định để lớp cuối/client cuối không luôn chịu thiệt. Histogram prior đi qua object riêng có version đầy đủ.

**Byte đề xuất, `C=16,D=66`, count uint32, tổng fp32, tham số xuống fp16:**

```text
upload   = C [4 + 4D(r+2)]
download = C [4 + 2D(r+2)]        # gồm count từng lớp; chưa histogram prior riêng
r=4: 25.408 B lên + 12.736 B xuống; mỗi lớp 1.588 B lên / 796 B xuống
r=2: 16.960 B lên +  8.512 B xuống
```

Với `r=4`, upload giảm 82,6% so với 145.792 B hiện tại. Số trên là payload, **chưa** schema/seed, header, ACK, truyền lại, prior riêng hay giao thức secure aggregation.
Seed/schema đã thống nhất từ cấu hình là giả định phải ghi; nếu phân phối qua RSU thì trả byte thực.
H6 có cursor vẫn cần vì B0 p10 dưới kích thước một lớp. Tất cả tham số phải serialize/decode thật, không chỉ giảm bộ đếm byte.

**Cổng có thể bác bỏ.** Hai seed, cùng lịch P1:

- Không kênh: `nys4 − lr4_exact ≥ −0,01` ở group gp; benign F1 giảm không quá 0,03.
- B0: `nys4_class − diag_class ≥ +0,01` cả hai seed, và giữ ≥70% lợi không kênh của chính `nys4`.
- Tách đóng góp chunking bằng 2×2 `{diag,nys4} × {whole_object,class_object}`. Không gán lợi truyền sớm cho covariance.
- Báo round đầu có knowledge, số lớp/cohort đã kích hoạt, bytes thực, macro-F1 theo round và F1 từng lớp.

**Rủi ro.** Sketch quá hẹp bỏ hướng hiệp phương sai hữu ích; fp32 sum rồi fp16 download có thể gây covariance lỗi số.
Tích lũy fp64 tại client, giải bài toán nhỏ fp64 tại RSU, kiểm tính hữu hạn/PSD sau decode; báo tỉ lệ residual phải clip.
Secure aggregation ở đây là mô phỏng tổng; ngưỡng cohort không phải bảo đảm DP hay bảo vệ khỏi phép trừ hai bản tổng.

## 3. S2 — Federated dataset condensation: học codebook rồi chuyển giao

**Vấn đề.** Gaussian giữ moment nhưng chưa đủ quan hệ phi tuyến; H2 thắng do mẫu tổng hợp tác động toàn backbone.
Thay vì chọn thêm dạng Gaussian, học trực tiếp một codebook nhỏ ở input nhằm giữ nhiều phép đo phi tuyến của dữ liệu.

**Tiền lệ.** [FedDM, CVPR 2023, §3.2](https://arxiv.org/pdf/2207.09653) học synthetic set bằng khớp feature/logit rồi gửi set lên server;
[code tác giả](https://github.com/anonymifish/fed-distribution-matching).
Paper đồng bộ model toàn cục theo round, và thí nghiệm khởi tạo synthetic bằng ảnh thật (§4.1).
**Không sao chép hai lựa chọn đó:** ở đây khởi tạo bằng noise/Gaussian của chính client; model private không gửi đi.

**Cơ chế MobiHet đề xuất:**

1. Mỗi client tạo `m=4` điểm/lớp từ Gaussian diag local, RNG riêng; không lấy/chỉnh trực tiếp record thật.
2. Dùng DAGSNet khởi tạo ngẫu nhiên, đóng băng trọng số, chế độ eval; pilot dùng một encoder/seed run.
   Tối ưu tọa độ codebook theo sai số mean latent256chiều trên local cùng lớp. Gradient chỉ cập nhật codebook.
3. Upload codebook fp16 + số điểm synthetic/lớp. Histogram nhãn local đi qua tổng nhiều client riêng cho prior;
   không gửi histogram local riêng kèm codebook. RSU giữ codebook tổng hợp theo nguồn/lớp, chuyển tiếp theo contact.
4. Xe nhận CE trên mẫu tổng hợp cộng CE local; gradient cập nhật **toàn private DAGSNet**.
   Không averaging model/latent private; input codebook dùng được cho kiến trúc private khác.

Đây là **static condensation lấy cảm hứng FedDM**, không reproduction FedDM; không nhận trước khả năng giữ loss landscape quanh model đã train.
Vì embedding ngẫu nhiên cố định, thống kê không bị cũ theo private backbone. Rủi ro ngược lại: embedding chưa học có thể yếu.
Làm rõ từ audit v1: `books` chỉ chứa tọa độ;64B count/metadata trong dự toán chưa có wire codec.
Ngoại lệ đọc synthetic không mở quyền đọc histogram local riêng; histogram real-count vẫn gộp cohort.

**Byte.** `up = C m D ×2 = 8.448 B/client`, cộng 64 B count và metadata.
Một object `(source,class,version)` có 4×66×2 = **528 B** tọa độ; nhận đủ object mới sử dụng.
Download toàn bộ codebook của 19 client khác là **160.512 B/xe**, chưa header/count; không được báo mỗi xe chỉ tải 8,4 kB.
B0 cần chuyển dần; ưu tiên vòng tròn source/class, hạn chế lặp nguồn, tránh object lớn của toàn bank.
Không kênh có thể đọc đủ tổng hợp; arm kênh chỉ đọc bank thực nhận.

**Đối chứng quyết định:** codebook Gaussian local **chưa tối ưu**, cùng seed, 4 điểm/lớp, nguồn, byte, lịch truyền và liều phụ.
Chỉ khác bước condensation. Nó kiểm trực tiếp liệu DM thêm tri thức hơn làm mượt + diversity nguồn hay không.
Local-only và H2 diag là mốc chung; không dùng tập record thật liên client làm upper control.

**Cổng:** DM − codebook chưa tối ưu ≥+0,01 group gp ở cả hai seed; DM − local ≥+0,03; benign F1 ≥local−0,03.
Đồng thời báo positionMirroring; chỉ nhận đã sửa lớp này nếu F1 hơn đối chứng ≥+0,03 cả hai seed.
Sau cổng không kênh, bật B0/B2, yêu cầu giữ ≥70% lợi riêng so với codebook đối chứng cùng kênh.

**Rủi ro.** Bank nhỏ gây overfit; DM học fingerprint receiver; optimization tạo giá trị feature không hợp lệ;
điểm tối ưu có thể ghi nhớ dù không khởi tạo bằng record thật. Kiểm provenance, raw-init cấm, đo khoảng cách láng giềng tại chính client.
Không chia sẻ raw không đồng nghĩa DP; chưa có nhiễu/accountant thì không gắn nhãn DP. Báo chi phí condensation ngoài contact.

## 4. S3 — Fourier distribution sketch → input tổng hợp

**Vấn đề.** S1 vẫn Gaussian, S2 chỉ gửi ít điểm/lớp. Cần thống kê nhận biết phân phối phi tuyến mà không phụ thuộc backbone/private architecture.

**Tiền lệ chính.** [DP-MERF, AISTATS 2021](https://proceedings.mlr.press/v130/harder21a.html) dùng random-feature kernel mean embedding để khớp dữ liệu tổng hợp,
có thí nghiệm dữ liệu bảng; [code tác giả](https://github.com/frhrdr/dp-merf) có nhánh `code_tab`.
Paper không chứng minh sẵn giao thức RSU bên dưới. Phiên bản không thêm nhiễu ở đây **không mang bảo đảm DP của DP-MERF**.

**Thích nghi FL đề xuất.** Client tính một lần `n_c` và `t_c=Σφ(x)` theo lớp, với Fourier map cố định bằng seed/schema chung.
RSU cộng các tổng; phát `m_c=t_c/n_c`. Từ Gaussian base/seed độc lập, mỗi client tối ưu một bank điểm 66 chiều theo
`Σ_c ||mean φ(x̃_c)−m_c||²`; giới hạn số bước cố định, ngoài contact. Sau đó CE trên nhãn sinh lớp, gradient đi qua toàn DAGSNet.
Không dùng logits teacher, không cần model phụ khác DAGSNet. Tối ưu trực tiếp bank thay generator là lựa chọn pilot của MobiHet, khác thuật toán gốc.

Đây vẫn là mean trong không gian phi tuyến, nhưng không phải tâm input/latent đã thất bại: nhiều hàm Fourier đo nhiều thành phần phân phối.
Finite sketch **không** xác định duy nhất phân phối; không được suy từ tính characteristic của kernel vô hạn rằng `M=128` đủ.

**Byte thêm**, `M=128`: `up=C(4+4M)=8.256 B`, `down=C(4+2M)=4.160 B`, chưa Gaussian base/prior/schema/overhead.
Tổng gồm base diag ít nhất **25.216 B/client**. Mỗi class sketch ~520 B lên /260 B xuống.
Khởi tạo bằng Gaussian đã nhận; không lấy bank mẫu local làm điểm khởi tạo để phát đi. Sinh tại client tránh payload hàng trăm kB cho bank điểm.

**Cổng.** So với H2 cùng base, liều phụ, số bước DAGSNet: tăng group gp ≥+0,015 cả hai seed; benign F1 giảm ≤0,03.
Đối chứng Fourier mục tiêu tính từ chính Gaussian base, cùng số bước sinh, nhằm tách lợi của optimizer khỏi tri thức client.
Báo sai số trên bộ Fourier kiểm tra độc lập, cố định trước; sai số khớp chính sketch dùng tối ưu không đủ chứng minh fidelity.
Nếu gửi sketch kiểm tra thì trả byte; nếu tính tại client thì chỉ trả metric gộp. Không đọc raw data tại RSU để chấm.

**Rủi ro.** Khớp hữu hạn có thể sinh điểm không hợp lệ, mất mode hoặc overfit bank; lặp bank nhỏ từng làm DS-FL kém H2.
Giới hạn miền lấy từ schema/giá trị train gộp hợp lệ, không holdout; tái sinh theo seed round nếu chi phí cho phép và báo FLOP/thời gian.
Giữ chính 66 feature đầu vào; Fourier map chỉ phục vụ thống kê, không thay input classifier bằng feature mới.

## 5. Pilot nhỏ nhất nên triển khai trước

**S2 và S3 trước.** Local smoke kiểm provenance/codebook gradient, sketch cộng đúng, wire round-trip, không đọc record client khác,
frame đứt không activate, prior chưa nhận không dùng. Tách chẩn đoán số học khỏi kết luận F1. Không train sweep local.

Kaggle: hai seed 42/43; cùng fixture/20 round/P1; private DAGSNet, không proxy mới.
Sàng lọc không kênh gồm 6 arm; cấu hình/cổng chính xác ở [pilot-new-fl.md](pilot-new-fl.md):

| arm | mục đích |
|---|---|
| `local` | mốc dưới, cùng prior/đánh giá |
| `h2_diag` | mốc Gaussian quần thể |
| `kme` | S3, finite Fourier sketch → synthetic input |
| `kme_gauss` | target Fourier giải tích của Gaussian, cùng optimizer/bank/byte |
| `codebook_gauss` | codebook Gaussian local hữu hạn, chuyển giao nguyên dạng tổng hợp |
| `codebook_dm` | S2, cùng codebook khởi tạo nhưng có condensation |

Đây là **sàng lọc 12 run**; đối chứng optimizer KME được chạy ngay cùng đợt.
Sinh bank tại client từ moment local cũng được, nhưng phải cố định lựa chọn giữa arm; không chuyển sang pooled moment miễn phí.
Chỉ hướng qua cổng mới lên B0/B2; S1 chạy riêng nếu cần giải bài toán byte, không ghép để cứu F1.
Chỉ tái sử dụng kết quả cũ khi source, preprocessing, codec, lịch và đánh giá khớp; sửa prior/quantization làm mốc P2 cũ không còn đối chứng trực tiếp.

Mọi arm báo đủ 10 metric, group gp chính, head phụ, benign F1, từng lớp, byte lên/xuống, contact tới activation.
Prior toàn cục phải dựa trên snapshot thực nhận; tách metric oracle-prior nếu cần chẩn đoán và không dùng nó chốt cổng.
Đặt tên/hash code mới; thu output rồi verify source đã chạy. Quota ưu tiên account còn ít giờ nhưng đủ ETA +15%, kiểm quota/slot lúc push.
Chưa cần thao tác tay cho phần đề xuất; xác thực/điều khoản tài khoản chỉ yêu cầu khi preflight thực sự báo thiếu.

Không mở rộng run để tìm seed thắng. DM trượt: phân biệt lỗi tối ưu, bank ít, embedding yếu và tổng quát hóa receiver.
KME khớp training sketch nhưng F1 không tăng: kiểm sketch độc lập/synthetic validity trước khi tăng số round hoặc số feature.
