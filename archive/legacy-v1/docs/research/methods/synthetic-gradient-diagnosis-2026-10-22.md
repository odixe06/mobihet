# Synthetic có tạo đúng cập nhật private? — chẩn đoán tiếp sau compact

Owner22/10 yêu cầu tiếp tục tìm nguyên nhân, giữ cổng/R1 và phương pháp chốt. Đọc
[bằng chứng P23–28](synthetic-learning-recovery-2026-10-22.md),
[build cũ/veto](synthetic-build-history-audit-2026-10-22.md),
[bàn giao trước compact](../../archive/operations/continuation-synthetic-2026-10-22.md).
Ghi protocol **trước khi chạy P29**; không dùng official test, không launch online FL.

## Khoảng chưa được cô lập

P24 teacher agreement cao hơn nhưng private gain nhỏ; P27 cùng teacher với real support tốt hơn rõ.
Hai phép này chưa tách target probabilities khỏi đặc trưng private do input sinh ra.
Với frozen Linear head A và `h̃_k=[h_k;1]`, gradient KL là
`G_k(Q)=E_Q[(softmax(A h̃_k)−q_T(x)) h̃_kᵀ]`.
Khớp mean/covariance của `h_T(x)` không trực tiếp khớp tích trên `h_k(x)`.
Đây là đồng nhất thức gradient; còn utility cần can thiệp, không suy từ cosine đơn lẻ.

## P29: phép đo khóa trước chạy

- Cùng audited xa s42 ep15,20 client; fixture2,5%, old/group65.536, backbone/BN frozen.
- Tái lập P24:16.384 own rows/client, teacher Project48, bank4.096, seed24, inversion80 bước.
  Thêm real pooled support oracle theo đúng class schedule của P27. Oracle chỉ chẩn đoán R3-ineligible.
- Cùng original/private head: đo own CE có logit adjustment, own KL, synthetic KL;
  full-bank scalar `w=1/(CE_own+CE_proxy_old)` frozen. Đây là surrogate toàn bank,
  không giả vờ là kỳ vọng chính xác của adaptive minibatch weight trong full harness.
- Mỗi teacher cũ/mới × ba support, đo gradient với target teacher theo từng input.
  Đối chứng **cùng class-conditional mean target của real oracle** trên cả ba support;
  gradient còn khác thì target mean riêng từng nguồn không thể giải thích hết.
  Đây là oracle diagnostic, không được truyền target từ real bank thành kênh FL mới.
- Reference gradient trên old/group nhãn thật: record-mean CE và class-balanced CE;
  ghi benign riêng. Holdout chỉ đo gradient/ảnh hưởng, không tối ưu input/step theo holdout.
- Bước can thiệp SGD theo từng hướng, chuẩn hóa logit RMS trên **own train** thành0,01,
  rồi chấm old/group. Common scale cố định trước chạy; không line search theo metric.
  Chỉ đo tính có lợi cục bộ, không thay gate P24 hoặc horizon.
- Đo class-conditioned teacher-feature moment error, private-feature mean/variance mismatch so real bank;
  target mass, gradient cosine/norm và phần mean/covariance trong residual×feature decomposition.
- Kiểm gradient công thức khớp autograd; mean+centered decomposition khớp gradient;
  finite/dtypes/provenance/source hash. Lưu JSON đủ10metric/CM cho bước can thiệp.

## Giả thuyết và dấu hiệu bác bỏ

| Giả thuyết | Dấu hiệu phân biệt | Bác bỏ / giới hạn |
|---|---|---|
| H1: thiếu thông tin xác định gradient private dù teacher feature khớp | Teacher moment error giảm, private gradient/moments không cải thiện tương ứng; cùng target oracle vẫn lệch giữa support | Nếu same-target gradients gần real và finite-step giữ benign/worst, H1 chưa giải thích phần thất bại đó |
| H2: target mass/nhãn gây phần lớn gradient sai | Same-target control làm Gaussian/inversion gần real; classwise residual/bias lệch lớn ở original target | Target đổi mà gradient/step vẫn lệch thì target mass không đủ giải thích |
| H3: xung đột với own CE/own KL gây mất lớp đã tốt | Synthetic direction có lợi riêng nhưng tổng loss đổi sang hướng hại real/benign | Nếu syn riêng đã lệch, không được chỉ sửa loss để gọi đã chữa support |

Không đòi một kết luận duy nhất; phân biệt hướng tại snapshot với quỹ đạo Adam1000 bước và online BN/backbone.
Không thử lại inversion/gradient-only adaptation đã FAIL; đây là phép đo để xác định thông tin bộ sinh còn thiếu.

## Kết quả

P29 đã chạy đủ20 client,80,7s local; [JSON](../../methods/full-drift/results/probe29-private-gradient-s42.json),
[audit](../../methods/full-drift/results/probe29-private-gradient-s42.audit.json),
[source](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe29.py).
1.120 lượt predictions→CM→đủ10metric PASS. CE/KL gradient khớp autograd max|error|2,78e−17/1,39e−17;
mean+centered decomposition PASS. Công thức KL dùng `p*sum(q)−q` giữ đúng row mass target fp32,
không nới threshold kiểm. Không có method utility PASS từ phép đo này.

| Support | Teacher48 mean scaled MSE | Mean private-feature mean scaled MSE | Cosine gradient syn với balanced-CE group | Cosine gradient tổng với balanced-CE group |
|---|---:|---:|---:|---:|
| Gaussian | 1,069456 | 1,1316 | 0,1230 | 0,0038 |
| Inversion | 0,094089 | 0,4432 | 0,2044 | 0,0413 |
| Real oracle | 0,005939 | 0 | 0,5488 | 0,2085 |

Hai cột cosine dùng teacher **Project48**; cùng original private head. Balanced CE là surrogate theo lớp,
**không phải gradient macro-F1**. Cosine tổng âm ở10/20 client cho Gaussian/inversion,6/20 cho real oracle.
Own CE hướng với record-CE group cosine0,4622 nhưng với balanced-CE−0,0561; own KL teacher mới là0,4739/−0,0332.
Đây là alignment tại snapshot/full-bank surrogate, không chứng minh mọi bước Adam/epoch thực đều có cùng hướng.

**Giữ cùng class-mean target của real oracle:** cosine syn gradient so real trên cùng target:
Gaussian/inversion0,1678/0,3078 (teacher cũ),0,1928/0,5419 (teacher mới), real=1.
Target mean được giữ đúng theo component class ở control này nhưng private gradient vẫn khác.
Vì vậy soft-target mass riêng của mỗi nguồn **không đủ** giải thích toàn bộ khác biệt support.
Control x-independent đã bỏ intra-class target/input coupling; không xem nó là generator đã được nhận.

**Target mass thực:** ở teacher mới, Gaussian dành22,87% cho randomSpeedOffset và4,52% cho benign,
so real oracle5,54% và12,15%; inversion đưa về6,91% và12,68%. Inversion sửa đáng kể mass,
nhưng balanced-CE gradient vẫn xa real. Cả support và target allocation đều có phần trong vấn đề.

**Giới hạn:** teacher feature fit của inversion vẫn chưa hoàn hảo. P29 không giữ teacher distribution giống hệt
giữa các support, nên chưa cô lập residual optimization error khỏi inverse ambiguity trong model thật.
Phản ví dụ trong [ghi chú nhận diện](synthetic-transfer-identifiability-2026-10-22.md) chứng minh **không có bảo đảm**,
không thay một intervention final50. Step SGD RMS0,01 chỉ chẩn đoán cục bộ; mọi cổng utility/horizon vẫn giữ.

## P30 — kiểm carrier liên kết, khóa trước chạy

Động cơ: P29 cho thấy inversion giảm teacher mean error mạnh hơn private mean error; cùng target theo lớp,
private gradients vẫn khác giữa support. [Phân tích nhận diện + nguồn](synthetic-transfer-identifiability-2026-10-22.md)
chứng minh marginal teacher/input moments không đủ xác định update tùy ý. Không có bảo đảm carrier mới đủ.

**Thông tin thêm:** cohort sums `B[c,a,j]=Σ w 1[y=c] p_T,a(x)[1;x_active]_j`,53 active features,
cùng frozen teacher gốc. Chuẩn hóa sau khi cộng; client `w=N_full,kc/m_sample,kc` từ audited full train manifest,
không dùng fixture population làm full. Serialize/decode thật fp32+CRC; B giữ joint label×teacher×input.
Input mean/covariance tĩnh vẫn là loại thống kê Gaussian đã có; trong probe ước lượng từ cùng weighted sample,
production muốn tái dùng bootstrap full phải đối chiếu estimator. Không gọi ngân sách của riêng B là R1 toàn phương pháp.

**Sinh:** noise Gaussian class-weighted input → tối ưu từng lớp80 bước Adam lr0,03, m4b luôn0.
Khớp standardized input mean/covariance và conditional mean teacher probability; trust penalty0,01 như P24.
Arm `joint` thêm normalized crosscov `Cov(p_T,x_active|true_class)`; control `mean_only` bỏ đúng term này.
Control `zero_cross` dùng cùng term/trọng số nhưng target cross bằng0, tách thông tin covariance khỏi thêm regularization.
Mọi loss dimension-mean trọng số1; q scale `sqrt(qbar*(1−qbar)).clamp_min(0,05)`, input scale std clamp0,1.
Không fit vào real record, không chọn step/weight theo holdout. Đây là probability-weighted input moment matching,
cùng họ gradient/distribution matching, không claim novelty và không lặp unconditional head-gradient.

**Private:** teacher Project48 cố định, frozen backbone/BN, đúng `P24.adapt`1000 bước,
512 own+256 synthetic, cùng seeds. Năm support: Gaussian gốc (matched reproducibility control),
Gaussian với weighted input moments, weighted `mean_only`, weighted `zero_cross`, weighted `joint`. Cùng class schedule/bank4096.
B được tạo từ teacher gốc FP32; logits KD/eval theo cùng AMP path P24, precision khác được ghi rõ.
Vì teacher tạo B khác head KD Project48, hiệu ứng joint có thể gồm thay target probabilities của KD;
không được gọi contrast này là đã cô lập hoàn toàn geometry. Covariance là weighted empirical estimate,
không tự thành unbiased full covariance chỉ bởi denominator N_full−1.

**Cổng local trước Kaggle:** joint−mean_only ≥+0,005 và joint−Gaussian gốc≥+0,005 mean-private group;
benign≥−0,005, worst≥−0,010 cho cả hai contrast. Đo tất cả20, đủ10metric/predictions/CM.
Muốn nhận ảnh hưởng joint phải hơn control weighted Gaussian và `zero_cross` cùng gain/guards;
codec/fit loss không thay utility. Negative control và kiểm sai số centered cross sau codec được thêm
trước khi có bất kỳ kết quả utility P30; lỗi tuple/logits lúc khởi tạo đã sửa, chưa chạy bước học nào ở lượt lỗi.
Nếu FAIL: đóng adaptation này, không tăng steps/đổi weights để tìm win. Nếu PASS: hai-seed T4 mechanism screen,
online ep15/horizon/R1/R2 vẫn riêng và giữ cổng cũ. Chưa launch online.

**R1 trước utility:** B standalone raw~111k hai chiều; stack Project48 mọi round vượt cap1,5× ở15/50.
Sparse refresh là khả năng cần matched teacher-age controls, chưa chốt lịch; bootstrap/contact q3 vẫn chưa giải quyết.

### P30 đã audit — FAIL, không lên Kaggle

[JSON](../../methods/full-drift/results/probe30-joint-input-s42.json),
[audit/cổng](../../methods/full-drift/results/probe30-joint-input-s42.audit.json),
[source](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe30.py).
Đủ20 client,297,5s local,200 lượt predictions→CM→10metric và finite head checks PASS.

| Support | Mean private group F1 | Benign F1 | Worst client |
|---|---:|---:|---:|
| Gaussian gốc — rerun | 0,828539 | 0,781529 | 0,782718 |
| Gaussian weighted | 0,829048 | 0,781421 | 0,779867 |
| Mean-only | 0,832315 | 0,782384 | 0,792109 |
| Zero-cross | 0,832284 | 0,781531 | 0,791870 |
| Joint | 0,832083 | 0,781852 | 0,792433 |

Joint−mean-only **−0,000232**,−zero-cross−0,000201;−Gaussian gốc+0,003544 <+0,005.
Mọi gain gate FAIL; không tăng bước/thay trọng số hoặc launch Kaggle để tìm win.
Cross loss giảm0,010594→0,002485 so mean-only, không tạo utility từ joint carrier đã thêm.
Không kết luận mọi nonlinear joint carrier đều vô ích; thống kê cụ thể này chưa chứng minh mang hướng cập nhật cần thiết.

B serialize thật **55.400 B lên +55.400 B xuống =110.800 B/client/query**;
sai số relative Frobenius của centered cross sau encode/decode2,33e−7. Chỉ B, chưa toàn communication.
Input-scaled cross energy9,81e−5 không phải correlation hoặc phần trăm information; không đổi tên thành thông tin có ích.
Gaussian gốc tái lập P24 trong0,000483 F1/client, không bit-exact; mọi contrast utility trên cùng rerun P30.
Số raw110.848 B ở ghi chú primary là thiết kế uint64 counts khác; không trộn với codec fp32 counts đã đo ở đây.

## P31 — đo hướng input teacher không nhìn thấy nhưng private có đáp ứng

Khóa trước chạy:16 train rows (mỗi true class1, chọn seed31, source client=c), cùng xa ep15,20 private.
Không train/optimize generator, không official test/holdout labels. Trên53 active standardized inputs:
Jacobian teacher feature48 `J_T`, null projector N qua SVD (rank threshold max(1e−8,1e−5σ_max));
Jacobian centered private logits `J_k`. Đo `||J_k N||_F²/||J_k||_F²` và finite perturb ±εv,
v là hướng private nhạy nhất trong N, ε cố định0,01 và0,001. Kiểm `J_T v` nhỏ.
Teacher latent drift và private logit drift chuẩn hóa theo own Jacobian norm, không so hai đơn vị thô.
Nếu private còn đáp ứng mạnh trong N, objective latent48 bỏ sót hướng quyết định **ở những điểm đo**;
không chứng minh đây là nguyên nhân duy nhất/mức đóng góp vào F1 final50.
Nếu ratio gần0, không tiếp tục đổ mismatch cho fiber ambiguity mà phải tìm lỗi moment/optimization/loss khác.

### P31 đã đo — có hướng mù, chưa phải nguyên nhân chính

[JSON](../../methods/full-drift/results/probe31-private-observability-s42.json),
[source](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe31.py).
16 điểm×20 private=320 trường hợp,47,9s local; teacher Jacobian rank48/null5 ở cả16 điểm.
Source SHA, số điểm/client, rank, finite values và linear null response đã kiểm lại từ JSON;
không phải audit predictions/utility vì phép này không train hoặc chấm classifier.

Phần năng lượng Jacobian private trong teacher-nullspace: **mean0,4618%, median0,1618%, max4,8707%**.
Linear teacher response trên hướng null có relative error tối đa cỡ1e−16.
Ở ε=0,001, finite response đã chuẩn hóa theo Jacobian riêng: teacher mean0,0002645,
private mean0,04249; ε=0,01 tương ứng0,002091/0,04201.
Hai response có normalization riêng, không dùng tỷ số của chúng làm phần trăm thông tin mất.

Lượt đo cuối tắt cả matmul/cuDNN TF32; lượt đầu còn cuDNN TF32 gây nhiễu finite difference nhỏ
đã được thay bằng phép đo strict FP32. Chỉ sửa diagnostic P31, không đổi precision production.
Có hướng private đáp ứng mà teacher48 không nhìn thấy, nhưng năng lượng trung bình nhỏ trên16 train points.
**Không quy thất bại chủ yếu cho nén48 chiều**, không suy từ phép này ra toàn phân phối hoặc F1 final50.

## Kết luận giữ qua compact

1. **Đã có bằng chứng support và target allocation cùng góp phần.** P29 giữ class-mean target giống nhau
   vẫn còn lệch private gradient; inversion sửa target mass và teacher fit nhưng chưa tạo update như real support.
   Sai lệch own CE/KD với balanced-CE còn làm tổng hướng yếu đi; balanced-CE chỉ là surrogate tại snapshot.
2. **Thêm một joint moment chưa chữa được.** P30 khớp cross loss tốt hơn nhưng không hơn mean-only/zero-cross,
   gain so Gaussian dưới cổng. Đóng adaptation này; không tune bước/trọng số hoặc push Kaggle.
3. **Chưa có một căn nguyên duy nhất cho plateau/gap final50.** P31 không ủng hộ đổ phần lớn lỗi cho mất chiều
   tại những điểm đã đo. P29–31 là frozen xa s42 ep15/fixture; report final dùng protocol khác và chỉ mô tả quỹ đạo.
   Không gộp gradient cosine, finite perturbation hoặc likelihood thành bằng chứng utility online.
4. **Bước sau còn là proposal:** kiểm capacity của conditional input density learner, học cấu trúc input từ dữ liệu thật
   ở client, không phụ thuộc inverse teacher. [Thiết kế/control/falsifier và nominal byte §7](synthetic-transfer-identifiability-2026-10-22.md).
   Chưa code/train; atom/dequantization, numerical scale và training/contact budget chưa chốt.
   Không tự jitter; pooled capacity-oracle chỉ local chẩn đoán R3-ineligible, không thành payload FL.

Phương pháp chốt và mọi cổng R1/R2/horizon giữ nguyên. Trạng thái toàn phiên/report và việc còn lại ở
[bàn giao22/10](../../archive/operations/continuation-2026-10-22.md); không có probe/collector mới chờ chạy.
