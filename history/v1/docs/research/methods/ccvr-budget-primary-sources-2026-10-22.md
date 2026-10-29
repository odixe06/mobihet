# CCVR tiết kiệm byte — nguồn gốc, suy luận, giới hạn

Ngày 22/10/2026. Nghiên cứu nguồn sơ cấp + suy luận đại số; chưa có utility mới. `cvl` FAIL giữ nguyên, không mở lại covariance chung.
Owner chốt phiên hiện tại: tổng byte/client ≤ 1,5× `xa` ở cả 15 và 50 round; từng contact ≤ **231.000 B**, kể cả bootstrap.
Các số dưới là kích thước vector thống kê giả định 4 B/phần tử, **chưa phải packet/SecAgg/contact PASS**.

## 1. Paper thực sự chứng minh gì

- **CCVR gốc:** lấy cùng extractor global đã train; ước lượng mean và covariance **riêng từng lớp**, sinh Gaussian đặc trưng,
  fit classifier sau huấn luyện. Algorithm 1 gửi thống kê, không gửi activation từng record. MobiHet sửa teacher mỗi epoch là adaptation
  ngoài bằng chứng post-calibration của paper. [Luo et al., NeurIPS 2021, §4, Algorithm 1](https://papers.neurips.cc/paper/2021/file/2f2b265625d76a6704b08093c652fd79-Paper.pdf).
- **Nyström/linear sketch:** giữ `Y = AΩ` với Ω chung; cộng cập nhật vào Y trước khi dựng xấp xỉ PSD từ
  `Y(ΩᵀY)†Yᵀ`. Chặn sai số ma trận phụ thuộc phổ/rank và sketch size; không phải chặn sai số classifier.
  [Tropp et al., NeurIPS 2017, Eq. 2.3–2.7](https://proceedings.neurips.cc/paper_files/paper/2017/file/4558dbb6f6f8bb2e16d03b85bde76e2c-Paper.pdf).
- **FedGen:** server học generator latent từ predictor của các client rồi phát generator xuống; §3.3 cho phép chỉ chia sẻ predictor.
  Không tự giải quyết tọa độ latent khác nhau giữa private model. [Zhu et al., ICML 2021, Eq. 4–5, §3.3](https://proceedings.mlr.press/v139/zhu21b/zhu21b.pdf).
- **DeepInversion:** tối ưu input từ noise, teacher cố định; khớp mean/variance BN; Adaptive DI thêm bất đồng teacher–student.
  Code gốc dùng prior ảnh, không thể bê nguyên sang 66 feature V2X.
  [Paper CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.pdf),
  [code NVIDIA, `DeepInversionFeatureHook`/`get_image_prior_losses`](https://github.com/NVlabs/DeepInversion/blob/master/deepinversion.py).

**Suy luận cho dự án:** giữ full covariance theo lớp là đối chứng mạnh; nén covariance trước khi bỏ cấu trúc theo lớp.
Lợi head teacher offline chưa chứng minh lợi private online. Bằng chứng hiện có: full CCVR 0,893, `cvl` 0,848,
teacher gốc 0,842; xem [hồ sơ cv](teacher-virtual-feature-calibration.md), [results vòng 25](../../methods/full-drift/results.md).

## 2. Giao thức thống kê cộng được

Các công thức còn lại là **suy luận cho MobiHet**, không phải biến thể đã được paper chứng minh utility.
Với snapshot extractor `f_v` cố định, `h=f_v(x)∈Rᵈ`, client k/lớp c tính:

```text
n_kc = số dòng
s_kc = Σ h
Q_kc = Σ h hᵀ
n_c = Σ_k n_kc; s_c = Σ_k s_kc; Q_c = Σ_k Q_kc
μ_c = s_c/n_c
Σ_c = Q_c/n_c − μ_c μ_cᵀ                 # population convention
```

Nếu cần sample covariance: dùng `(Q_c − s_c s_cᵀ/n_c)/(n_c−1)`, đồng nhất mọi arm.
Đây là cách viết cộng tính của pooling trong CCVR Eq. 3–4. Phải gộp raw second moment trước khi center;
trung bình covariance local bỏ mất biến thiên mean giữa client.

| Biểu diễn gửi | Cộng bằng SecureSum tuyến tính? | Mất gì / cần gì |
|---|---|---|
| `n,s,Q` | Có | O(Cd²); snapshot chung |
| `n,s,QP` với P công khai | Có | O(Cdm); dựng Nyström sau cộng |
| `n,Pᵀs,PᵀQP` với P công khai | Có | O(Cm²); chỉ biết covariance trong P |
| Các block cố định của Q | Có | Mất covariance giữa block |
| `diag(Q)` | Có | Mất mọi tương quan |
| SVD/PCA/factor covariance riêng từng client | **Không**, nếu chỉ cộng factor | Factor hóa phi tuyến, basis/sign khác nhau; không lấy được pooled covariance bằng cộng factor |
| Cholesky/factor của covariance **sau** SecureSum | Có ở bước thống kê | Phân rã tại server; chưa tự giảm upload |

**Không tự gọi SecAgg thật:** emulator hiện tại chưa mã hóa. Fixed-point phải dùng scale/range chung và tính overflow tổng K client;
scale riêng từng client không thể cộng code trực tiếp. Xem [kiểm toán luồng thông tin](../federated-information-flow.md).

## 3. Hai mức nén giữ covariance riêng

### A. Sketch một phía + diagonal dư: fidelity cao hơn, byte lớn hơn

Gửi `n,s,diag(Q),QΩ` với Ω chung `d×m`. Sau cộng:

```text
Y_c = Q_c Ω/n_c − μ_c(μ_cᵀ Ω) = Σ_c Ω
N_c = Y_c (ΩᵀY_c)† Y_cᵀ
D_c = diag(diag(Σ_c − N_c))
Σhat_c = N_c + D_c
```

Trong số học chính xác, Nyström cho `0 ≼ N_c ≼ Σ_c`, nên D không âm, Σhat PSD và giữ diagonal của covariance thật.
Đây vẫn là covariance **riêng từng lớp**, khác hẳn `cvl`. Triển khai cần regularization/rank tolerance và đo sai số sau lượng tử;
clamp eigen/diagonal là thay estimator, phải báo. Sketch m lớn không bảo đảm giữ hướng phân biệt lớp:
đo sai số margin + teacher F1/per-class thay vì chỉ covariance Frobenius error.

### B. Covariance trong basis chung: tiết kiệm nhất, giới hạn sức chứa rõ

P trực chuẩn chung `d×m`, `u=Pᵀh`; gửi `n,Σu,Σuuᵀ`, sinh Gaussian `u|c` với covariance riêng rồi fit head `A u+b`.
Từ dữ liệu đã nén, **không** dựng được covariance đầy đủ 256 chiều. Nếu lift về `h`, phần ngoài P là giả định bổ sung.

P có thể lấy từ head global hiện tại và hướng random cố định, không lấy PCA riêng từng client.
Basis phụ thuộc moment round hiện tại cần thêm lượt hỏi/đáp; basis từ snapshot hiện tại hoặc tổng round trước tránh phụ thuộc đó.
P PCA từ server cũng phải tính phí phát basis, trừ khi mọi client tái dựng bit-exact từ snapshot đã nhận.

**Bẫy diagonal dư:** với projection `P(PᵀΣP)Pᵀ`, phần `Σ − P(PᵀΣP)Pᵀ` không chắc PSD.
Lấy diagonal trừ rồi clamp không tự giữ covariance trong P. Nếu muốn giữ chính xác covariance trong P, có thể dùng
`P B_c Pᵀ + (I−PPᵀ)D_c(I−PPᵀ)` với `D_c≽0`, nhưng diagonal gốc và covariance chéo chưa được xác định.
Không đánh đồng công thức này với Nyström A.

## 4. Nén theo logit/margin: ứng viên rất nhỏ

Teacher `z=W h+b`, W có 16 hàng. Softmax bất biến khi cộng cùng scalar vào mọi logit. Đặt
`H=I−11ᵀ/16`; rowspace của `H W` có rank ≤ 15. Chọn P trực chuẩn spanning rowspace này, với quy tắc tái dựng cố định.
Khi đó có L sao cho `H W = L Pᵀ`, nên **toàn bộ centered logits cũ** được tái tạo chính xác từ `u=Pᵀh`.

Với Gaussian trong h, covariance `PᵀΣ_cP` giữ chính xác phân phối centered logits của head **cũ** theo từng lớp.
Điều này **không** chứng minh head tối ưu mới nằm trong rowspace cũ: có thể thiếu hướng mà full CCVR dùng để sửa timeDelay/posMirror.
Một head affine mới trên u = phép hiệu chỉnh phối hợp logit, mạnh hơn bias/temperature đơn; vẫn có thể không đạt gain.

Nếu m<15, mất một số hướng margin cũ. Giữ `W_old h + A Pᵀh` nhưng chỉ sinh u rồi gán phần baseline bị bỏ bằng mean
không còn mô phỏng đúng loss: cần joint moment của baseline logits và u, hoặc giả định bổ sung được kiểm chứng.
Phương án rẻ hơn giữ cả 15 hướng nhưng covariance block 8×8 + 7 diagonal; vẫn bỏ covariance chéo.

## 5. Byte logical: không nhầm với contact thật

C=16, d=256, mỗi phần tử giả định 4 B; count cũng 4 B. Head down = `16×(m+1)×4` nếu dùng head trực tiếp trên u.
Chưa cộng header/version/checksum, basis, mask/protocol SecAgg, ACK/retry, bootstrap hoặc kênh proxy hiện có.

| Phương án | Upload/client/lần | Head down | Bản chất |
|---|---:|---:|---|
| Full covariance256 + mean | 2.121.792 B | 16.448 B | Full CCVR, vượt 231.000 B riêng upload |
| Nyström m8 + mean256 + diag256 | 163.904 B | 16.448 B | Dễ vượt khi ghép proxy cùng contact |
| Nyström m12 + mean256 + diag256 | 229.440 B | 16.448 B | Upload riêng gần hết B2 |
| Block feature8 (32 block) + mean256 | 90.176 B | 16.448 B | Mất tương quan giữa block |
| Project m32 + mean **chỉ m32** | 35.904 B | 2.112 B | Head bị giới hạn 32 hướng |
| Project m16 | 9.792 B | 1.088 B | Có thể gồm đủ rowspace centered head |
| Full centered-logit span m15 | 8.704 B | 1.024 B | Giữ covariance riêng của mọi margin cũ |
| Project m12 | 5.824 B | 832 B | Mất hướng nếu head centered rank>12 |
| Project m8 | 2.880 B | 576 B | Cần probe sức chứa trước |
| Span15, covariance block8 + diag7 | 3.776 B | 1.024 B | Giữ mean/variance15; bỏ phần tương quan |

Upload full: `4C[1+d+d(d+1)/2]`; projected: `4C[1+m+m(m+1)/2]`;
Nyström+diag: `4C[1+2d+dm]`. Nếu giữ mean256 thay mean m, upload projected m16 là **25.152 B**, không phải 9.792 B.
Lượng tử 16 bit không được mặc định giảm đôi: tổng moment thô có thể overflow; cần common normalization/clip/scale,
serialize/decode thật và kiểm sai số covariance/teacher. FP16 tensor riêng không tự là fixed-point SecAgg16.

Theo hồ sơ hiện tại `xa` trung bình khoảng 223 kB/round; **trung bình không cho headroom đỉnh**.
Một lần full covariance cuối 50 round có thể rẻ về tổng nhưng vẫn quá B2/contact.
Tiling full covariance256 cần tối thiểu 10 contact nếu mỗi contact dành trọn 231.000 B cho thống kê;
ghép proxy/overhead làm số contact tăng. Mọi tile phải gắn **cùng frozen extractor/version**, atomic commit đủ tile.
Không gom tile từ encoder thay đổi rồi gọi là covariance của một teacher.

Bootstrap `xa` khoảng 1,81 MB cũng quá một contact B2: phải chia contact thật/đếm đủ trước khi tuyên bố R1.
Không dùng tỷ lệ tổng ≤1,5× để thay cổng từng contact. Chuẩn so: [protocol §2](../protocol.md).

## 6. Final-only calibration: đúng bằng chứng gốc, cần đúng headline

Giữ extractor sau train cố định rồi tính moment/sửa head một lần tránh thống kê lỗi thời, bớt upload lặp.
Nhưng nếu chỉ sửa teacher **sau** epoch15/r50, private không tự thay đổi. Private headline chỉ được lợi khi:

1. Teacher được sửa trước epoch cuối rồi private học trong lịch đã khóa; hoặc
2. Có bước distill/head-fit private cuối, báo thêm compute và matched control cùng số bước.

Đây là hai protocol khác nhau; cần owner chốt trước run. Nếu dùng contact bổ sung cho final calibration, đếm trong tổng byte/contact/latency.
Thêm 10 contact nhỏ có thể hợp hạn byte nhưng không chứng minh deadline/contact availability.

## 7. Sinh input có chủ đích: khi nào thêm giá trị

**Vấn đề hiện tại:** label Gaussian lấy từ teacher; phần lỗi private mà teacher cũng sai không thể giải quyết chỉ bằng ưu tiên bất đồng.
Sau khi teacher mạnh hơn, virtual h vẫn không trực tiếp vào private khác kiến trúc; private cần input66 hoặc không gian chung có adapter đã chứng minh.
Đây là suy luận theo [probe 21/22 và vòng 25](../../methods/full-drift/results.md), không phải kết quả paper FedGen trên VeReMi.

Ứng viên có cơ chế mới: **đảo ngược snapshot theo moment lớp**, noise → input66; khớp projected/full class moments đã có,
khớp prior input/marginal hiện hành và teacher label; giữ snapshot cố định khi tối ưu. Teacher logits dùng cùng input cho mọi private architecture,
không cần align private latent. Vay nguyên lý DeepInversion; class-conditional covariance và prior V2X là adaptation mới.

Đối chứng tối thiểu: cùng teacher + Gaussian hiện tại, BN-only inversion, class-mean-only inversion, covariance lớp đúng vs shuffled,
cùng synthetic count/optimizer step. Sinh mẫu bắt đầu từ noise nếu rời client; không khởi tạo bằng record thật rồi coi là synthetic exception.

**Falsifier:** loss khớp moment thấp nhưng refit head trên synthetic không hơn Gaussian/DI cũ; teacher không đúng hơn trên lỗi private;
benign/lớp tốc độ bị mất như `cvl`; gain chỉ tăng tốc ep2–4 rồi mất ep15. Khi đó dừng, không tăng số bước synthesis để cứu cổng.
Raw moment đúng chỉ giữ bậc 1–2, chưa giữ atom/đa mode/quan hệ vật lý; không coi fidelity là utility.

## 8. Thứ tự probe đề nghị

1. **Frozen snapshot teacher:** full CCVR đối chứng offline → span15 full → span15 block → span32 gồm rowspace head + hướng bổ sung.
   Đo trần refit bằng dữ liệu thực chỉ dưới nhãn oracle; không biến oracle thành protocol FL.
2. **Giữ được gain + benign:** chọn phương án có packet/contact/tổng horizon thật phù hợp; nếu small-span FAIL nhưng full PASS,
   chuyển Nyström/final tiling, không dùng lại covariance chung.
3. **Teacher tăng nhưng private chưa tăng:** kiểm lớp/lỗi dạy được trên synthetic support; mới thử inversion theo moment lớp.
4. Cổng utility cuối vẫn private ep15 và seed43, benign, worst client; official test chỉ mô tả. Kết quả early epoch không thay headline.

Không phương án nào ở đây đã được xác nhận PASS. Thống kê/projector chỉ giải quyết khả năng mang tri thức; utility và R1/R2 phải kiểm riêng.

## 9. Client đang học gì; thay bộ sinh ở đâu mới có cơ chế mới

Đọc trực tiếp [fd.py](../../../methods/full-drift/builds/fd-probe/source/proj/fd.py), SHA256
`a19342eaab65f7d36a32839ec9cbfa30208ccbd82def5d6a685de31c7cfe8431`, ngày 22/10;
đối tượng = `gpool_q3_frz_gkd_la_kdpx_zl` (`xa`), không gộp cơ chế của arm khác.
Các nhận định "code làm gì" dưới đây là kiểm source; giải thích nguyên nhân phải giữ mức giả thuyết nếu chưa có intervention.

**Loss thực tế sau khi có snapshot gộp (`method.version>0`).** Với p private, q teacher frozen cả epoch, v proxy local,
π_k prior local, π_g prior gộp; mỗi KL tính batch mean riêng:

```text
CE_p = CE(z_p(x_real) + log π_k − log π_g, y_real)
CE_v = CE(z_v(x_real), y_real)
w = stop_gradient[1 / max(CE_p + CE_v, 1e−6)]
L_p = CE_p + w KL(q(x_real) || p(x_real))
            + w KL(q(x_syn)  || p(x_syn))
L_v = CE_v + min(w,1) KL(stop_gradient[p(x_real)] || v(x_real))
```

Source: prior dòng 335–342/623–631, loss dòng 946–991; `KD_CAP=1` dòng109.
KL proxy ở công thức là private→proxy, không phải KL mathematically đảo teacher→private trên cùng hai distribution;
và p được detach ở nhánh proxy. w detach nên gradient không đi xuyên denominator.
Trước snapshot đầu tiên, frozen-teacher KL tắt; do `syn=None`, `aux_ce` Gaussian **hard CE còn giữ trong r1**.
Không mô tả toàn bộ training là teacher KL từ bước đầu.

| Thành phần | Tri thức thực sự đi vào private | Giới hạn / phép đo cần |
|---|---|---|
| Real CE + logit adjustment | Nhãn own; dịch softmax theo π_k/π_g | Dưới giả định label shift, raw z hướng posterior prior gộp; khác receiver làm giả định thiếu. Không tự tạo feature của client khác |
| KL trên real | Hàm teacher trên support own | Nếu lớp/receiver khác hiếm trong own, gradient không chạm đủ vùng cần sửa |
| Gaussian256 + KL | Hàm teacher trên input từ moment lớp gộp | **Nhãn lớp dùng để rút Gaussian không còn là target CE** sau r1; teacher có thể cho nhãn khác |
| Proxy CE + capped KL | Teacher vòng sau học nhãn own và private predictions trước aggregation | Cải thiện private có thể phản hồi lên teacher; freeze teacher riêng giữ đối chứng nhân quả rõ hơn |

Logit adjustment dựa tần suất nhãn là họ kỹ thuật có nguồn gốc [Menon et al., ICLR 2021](https://openreview.net/pdf?id=37nvvqkCo5);
π_k/π_g ở đây là adaptation của dự án. Giải thích posterior ở bảng là suy luận label-shift, không phải bằng chứng VeReMi có label shift thuần.
KL soft-target là cơ chế distillation trong [Hinton et al., 2015](https://arxiv.org/abs/1503.02531);
công thức gradient dưới đây được suy trực tiếp từ CE/KL, không viện paper để khẳng định utility VeReMi.

**256 dòng không đồng nghĩa prior target uniform.** `aux_labels=arange(256)%16` ⇒ mỗi component Gaussian được rút16 dòng/batch
(source628–630). Nhưng target gkd là q(x_syn), nên mass target thực tế `mean_x q(c|x)` chưa chắc uniform.
Hai KL đều batch mean rồi cùng nhân w: 256 synthetic so512 real không tự làm tổng synthetic loss bằng nửa real loss.
Tăng số dòng chủ yếu đổi sampling/compute/BN; không có cơ sở xem đó là thêm nguồn tri thức.

**Private BN nhìn hỗn hợp real+synthetic.** `private.train()` rồi forward một lần trên concatenation `xp` (dòng858,913–925).
Với batch real512 + synthetic256, synthetic chiếm1/3 số dòng BN; real256 + synthetic256 chiếm1/2, trừ batch cuối.
Proxy train chỉ x_real; teacher eval. [DAGSNet](../../../core/dagsnet.py) có BN trong các nhánh convolution;
do đó thay support đồng thời thay batch statistics và running statistics của private. Đây là thay đổi chắc chắn của phép tính,
**chưa chứng minh BN là nguyên nhân FAIL**. Không tách BN như một "fix" trước khi đo intervention; bài học cũ về BN-real-only vẫn giữ.

**Ba nút thắt khác nhau, không chữa bằng cùng một thủ thuật.**

1. **Nguồn teacher:** trên cùng dòng holdout thật, teacher cũng sai ở phần lớn lỗi private (77% trong probe21).
   Gradient KL theo logit private là `p−q`; nếu q sai thì tăng coverage có thể tăng ép private theo lỗi đó.
   Câu này mô tả hướng gradient, không phải định lý student không bao giờ vượt teacher khi còn CE real/regularization.
2. **Support:** teacher tốt trên holdout thật chưa chắc tốt trên Gaussian lệch manifold. Tách teacher F1 thật,
   class-conditional label agreement trên synthetic, tỷ lệ lỗi private mà teacher đúng, rồi đo transfer cùng teacher trên support khác.
3. **Gradient hữu ích:** ngay cả nhãn đúng và support hợp lý, bước cập nhật có thể xung đột gradient CE own hoặc làm mất lớp đã tốt.
   Đo per-class teacher−private target mass, norm/cosine gradient CE_real vs KL_real vs KL_syn và sau một bước trên holdout riêng;
   cosine chỉ là chẩn đoán, không thay gain/worst-client gate.

**Vì sao virtual latent CCVR chưa thể dạy thẳng private khác kiến trúc:** h của snapshot global và h_k private không chung hàm,
thứ tự chiều hoặc kích thước; cùng256 chiều cũng chưa align. Gắn head private vào virtual h_teacher thay đổi ý nghĩa tọa độ.
FedGen cho thấy synthetic latent có ích trong giao diện predictor chung của họ, không cung cấp phép align miễn phí.
Đường phù hợp hiện tại: sửa teacher trong latent **chung**, rồi tạo/giữ input66 và distill logits sang private.
Nếu dùng adapter private-local, adapter phải học trên dữ liệu own và chứng minh coverage ngoài own; không tự giải quyết thông tin thiếu.

**Sớm/muộn và thống kê trôi.** `cvl` chỉ hơn head gốc ep2–4; từ ep6 mất lợi là quan sát đã audit,
không đủ để kết luận mọi CCVR muộn đều hỏng hoặc covariance chung là nguyên nhân duy nhất.
Full covariance offline ep15 còn lợi ⇒ kiểm compression trên snapshot muộn trước online.
Moment của f_v chỉ dùng để synth/calibrate head trên f_v; không gắn Σ_v vào f_{v+1} rồi coi là distribution hiện tại.
Snapshot cố định trong một bước final calibration giải quyết mismatch **nội bộ bước đó**;
snapshot teacher giữ nguyên nhiều epoch đổi bài toán sang teacher cũ, cần đánh giá riêng thay vì gọi là hết drift toàn training.

**Xếp thay đổi synthesis theo khả năng tác động, có điều kiện:**

| Ưu tiên | Cơ chế mới | Falsifier cụ thể | Đối chứng tối thiểu |
|---|---|---|---|
| 1 | Inversion input66 từ noise, khớp covariance **riêng lớp** của teacher đã hiệu chỉnh + mean/prior input. Thay support mang hàm teacher, không tăng KD strength | Moment loss giảm nhưng refit trên synthetic/transfer private không hơn Gaussian cùng teacher; gain không còn ở ep15; benign<−0,005 hoặc worst<−0,010 | Teacher cũ/mới × Gaussian/inversion =4 ô; cùng batch/prior/step và shared-BN policy; inversion mean-only/shuffled-cov chỉ thêm khi4 ô cho thấy tín hiệu |
| 2 | Hard CE theo label lớp **có nguồn aggregate** trên synthetic mới khớp class covariance; phá phụ thuộc hoàn toàn vào q sai | Chỉ thắng vì đổi prior/mass hoặc synthetic được teacher ép nhãn; oracle trên holdout không cho thấy label đáng tin; benign/lớp tốc độ giảm | Support cũ/mới × target teacherKL/hardCE =4 ô, cùng teacher và target mass được báo; ô Gaussian+hardCE là control của recipe cũ, không gọi hướng mới |
| 3 | Chọn basis đo covariance theo hướng phân biệt lớp: gồm centered-head rowspace, bổ sung class-mean directions từ **thống kê round trước** thay random/PCA-only | Covariance reconstruction tốt nhưng class weak không tăng; loss do nén không nhỏ hơn random basis cùng m; lợi chỉ xuất hiện với oracle basis có truy cập nhãn holdout | Full covariance oracle thông tin; task-basis vs random cùng m/byte/prior/solver, evaluate tại cùng snapshot |

Ưu tiên2 có điều kiện mạnh: aggregate class label không bảo đảm label từng điểm Gaussian/inversion đúng khi phân bố overlap.
Confidence của chính teacher hoặc CE teacher thấp không phải kiểm độc lập về label. Gate dựa own calibration riêng,
báo coverage/tỷ lệ bỏ mẫu theo lớp và control cùng khối lượng; không thêm nhãn record rời client.
Gradient hardCE là `p−e_c`, khác KL `p−q`; nếu e_c sai, lực sai có thể mạnh hơn soft target.
Vì vậy không quay lại hardCE Gaussian nguyên trạng đã thử, không đơn thuần đổi loss rồi gọi bộ sinh mới.

Với ưu tiên3, dùng pooled covariance chung để **chọn basis** vẫn khác dùng chung covariance để **sinh mọi lớp**:
covariance synth cuối phải giữ B_c riêng. Hướng class mean chưa chắc giữ biến thiên phân biệt benign↔timeDelay;
full-covariance reference kiểm xem gain bị mất ở estimator hay đã mất vì subspace.

**Phép thử đầu đủ phân biệt teacher/support:** freeze backbone weights **và BN statistics**/snapshot, fit private head và đo cùng holdout đã khóa;
2×2 teacher{gốc, calibrated đủ gain} × input{Gaussian, covariance-matched inversion}. Nếu teacher mới chỉ giúp trên dòng thật oracle,
không giúp cả hai synthetic support, chưa đủ bằng chứng triển khai online. Nếu inversion chỉ tăng hiệu ứng teacher cũ mà teacher mới không giúp,
kiểm label/BN/gradient trước khi tuyên bố teacher bottleneck đã giải quyết. Không dùng raw record oracle làm kênh FL.
Chỉ nhánh PASS mới chạy private toàn model, horizon/seed43; không mở lại `xo`, `cvl`, hay quét hệ số KD như lời giải.

**Cập nhật giới hạn byte thực đo từ nhánh chính phiên này:** xa screening peak15 =228.658 B/contact;
chuỗi final za peak50 =246.556 B/contact, đã **vượt231.000** ngay cả trước kênh CCVR. Các số là phạm vi hai run cụ thể,
không suy mọi account/client/horizon đều giống nhau. Main đang kiểm packet để chọn codec/channel; bảng §5 vẫn chỉ logical raw32.
Seeded initialization1.042 B thay bootstrap model1.593.440 B là ứng viên lossless chỉ khi tái dựng
weights+buffers bit-exact trên backend được dùng; chưa có chứng minh trong note này.
R1 cần sửa baseline vượt cổng rồi tính cả kênh mới, không trừ bootstrap bằng kỳ vọng hoặc lấy average để che peak.

## 10. Final50: private chững, teacher còn tăng — đo mô tả, chưa kết luận nguyên nhân

Đọc trực tiếp [metrics.csv](../../methods/full-drift/results/final/metrics.csv) và
[costs.csv](../../methods/full-drift/results/final/costs.csv) hiện có; arm `q3_zl la_kdpx`, seed42, test20% toàn cục,
private **head** và proxy **head**, mean-client macro F1. Không dùng `gp` thay headline ở đây.
Đây là phân tích mô tả các run đã chốt; **không chọn HP, round, lớp cần tune hoặc lịch LR trên official test**.
Các giả thuyết bên dưới phải kiểm lại bằng old/group nội bộ và đối chứng trước run mới.

### Mốc round và tốc độ tăng

| Client | Round | Private | Proxy | Proxy−private |
|---:|---:|---:|---:|---:|
| 20 | 10 | 0,811993 | 0,820306 | 0,008313 |
| 20 | 20 | 0,824455 | 0,832952 | 0,008497 |
| 20 | 30 | 0,828235 | 0,841917 | 0,013683 |
| 20 | 40 | 0,832971 | 0,850330 | 0,017359 |
| 20 | 50 | 0,832052 | 0,860333 | **0,028281** |
| 50 | 10 | 0,801064 | 0,813348 | 0,012285 |
| 50 | 20 | 0,812881 | 0,825015 | 0,012134 |
| 50 | 30 | 0,816003 | 0,829336 | 0,013333 |
| 50 | 40 | 0,816965 | 0,834522 | 0,017557 |
| 50 | 50 | 0,817759 | 0,842441 | **0,024682** |
| 100 | 10 | 0,782273 | 0,797805 | 0,015531 |
| 100 | 20 | 0,801069 | 0,814006 | 0,012937 |
| 100 | 30 | 0,807523 | 0,820795 | 0,013273 |
| 100 | 40 | 0,809870 | 0,825541 | 0,015670 |
| 100 | 50 | 0,811120 | 0,834641 | **0,023521** |

| Client | Δ private20→50 | Δ proxy20→50 | Δ private30→50 | Δ proxy30→50 |
|---:|---:|---:|---:|---:|
| 20 | +0,007597 | +0,027381 | +0,003817 | +0,018416 |
| 50 | +0,004878 | +0,017426 | +0,001756 | +0,013105 |
| 100 | +0,010051 | +0,020635 | +0,003597 | +0,013846 |

Không có plateau hoàn toàn từ round20: private vẫn tăng, nhất là100c. Nhưng từ30→50 tăng chỉ0,0018–0,0038,
trong khi proxy tăng0,0131–0,0184. **Bằng chứng hiện tại là hiệu quả hấp thụ thêm của private thấp hơn mức teacher đang cải thiện**;
chưa chứng minh giới hạn capacity của kiến trúc private hoặc teacher đã hội tụ.

Mean mỗi block5 round; mỗi ô là **private / proxy / gap**. Block là mô tả được chọn trước khi xem cực trị, không chọn checkpoint:

| Block | 20c | 50c | 100c |
|---|---|---|---|
| 1–5 | 0,738140 / 0,698969 / −0,039171 | 0,706660 / 0,685212 / −0,021447 | 0,684639 / 0,665851 / −0,018788 |
| 6–10 | 0,809108 / 0,814945 / 0,005836 | 0,794117 / 0,806837 / 0,012720 | 0,773464 / 0,789897 / 0,016433 |
| 11–15 | 0,817935 / 0,825674 / 0,007739 | 0,807374 / 0,818327 / 0,010953 | 0,791584 / 0,805345 / 0,013761 |
| 16–20 | 0,824660 / 0,833612 / 0,008953 | 0,812260 / 0,823564 / 0,011304 | 0,799532 / 0,811864 / 0,012332 |
| 21–25 | 0,824909 / 0,835873 / 0,010964 | 0,815303 / 0,827229 / 0,011925 | 0,803901 / 0,816792 / 0,012891 |
| 26–30 | 0,827726 / 0,841193 / 0,013467 | 0,815431 / 0,828913 / 0,013482 | 0,806549 / 0,820169 / 0,013620 |
| 31–35 | 0,829460 / 0,844106 / 0,014646 | 0,816313 / 0,831660 / 0,015347 | 0,807423 / 0,821607 / 0,014184 |
| 36–40 | 0,831385 / 0,849574 / 0,018189 | 0,817374 / 0,833886 / 0,016512 | 0,809093 / 0,824845 / 0,015751 |
| 41–45 | 0,832425 / 0,853586 / 0,021160 | 0,817102 / 0,836855 / 0,019753 | 0,809751 / 0,827543 / 0,017792 |
| 46–50 | 0,832016 / 0,858390 / 0,026374 | 0,817639 / 0,842173 / 0,024534 | 0,810369 / 0,831718 / 0,021349 |

Mean private block46–50 so41–45: −0,000409 / +0,000537 / +0,000618; không phải plateau do một round ngẫu nhiên duy nhất.
Đồng thời block proxy cuối vẫn lên. Chưa có paired multi-seed final50 để gắn ý nghĩa thống kê cho các delta nhỏ.

### LR, bước train và lag teacher

Inheritance đã kiểm: `fd.Run(FullRun)` → `full_run.Run(Base)` → `v2_run.Run(P.Run)`;
[pilot.Run.lr](../../../core/pilot_runtime.py) dòng119–121, khớp source trong artifact eval ze04:

```text
lr(r) = 1e−5 + 0.5 × (1e−3 − 1e−5) × [1 + cos(π(r−1)/49)]
```

Config artifact train của cả3 kịch bản xác nhận `rounds=50, lr=.001, lr_min=.00001`, batch512/512/256.
LR hằng trong mỗi epoch và optimizer AdamW được tạo lại theo client/round (§9).

| Round | LR |
|---:|---:|
| 10 | 9,19853611921e−4 |
| 20 | 6,75955701939e−4 |
| 30 | 3,64158844618e−4 |
| 40 | 1,08300257175e−4 |
| 49 | 1,10170305886e−5 |
| 50 | 1,00000000000e−5 |

Từ30 LR giảm khoảng36,4× tới50: **có thể** hạn chế private hấp thụ teacher mới ở cuối; chưa là nguyên nhân được cô lập,
vì proxy cùng lịch vẫn tăng và hai model có loss/support khác nhau. Không suy ra cần đổi LR hoặc chạy thêm epoch.
Muốn kiểm, trước tiên đo tại frozen snapshot nội bộ: một bước/một epoch cùng LR có cải thiện private khi đổi teacher/support hay không;
không chọn LR dựa curve test này.

`costs.csv` ghi mọi round vẫn đủ84.083 /84.098 /168.200 local steps; r31–50 skip477/419/877,
tương ứng0,0284% /0,0249% /0,0261%. Không thấy đa số update bị skip có thể giải thích plateau;
đây không phải kiểm toán toàn bộ health/gradient. Peak payload/client của3 chuỗi là246.556 /255.895 /253.731 B, đều vượt B2 hiện hành.

Private round50 học frozen S49; proxy được báo là S50 sau aggregation. Đo mô tả lag:

| Client | Proxy49 | Proxy50−proxy49 | Proxy49−private50 |
|---:|---:|---:|---:|
| 20 | 0,859074 | +0,001260 | **0,027022** |
| 50 | 0,844653 | −0,002212 | **0,026894** |
| 100 | 0,831589 | +0,003052 | **0,020469** |

So với đúng snapshot S49 vẫn còn gap0,0205–0,0270. Sai khác S49→S50 **không đủ làm phép giải thích số học toàn bộ gap**;
nhưng hiệu ứng nhân quả của teacher lag chưa được đo: hai teacher phải dạy cùng private/init/support/step để phân biệt.

### Gap đến từ lớp nào: confusion r50 đã audit

Đọc `round_050.json` dưới các notebook có `AUDITED.json complete=true`:
[za20c](../../../methods/full-drift/builds/fd-probe/artifacts/kaggle/fd-e45-50-s42za02-20261015-r25-v1-f2/AUDITED.json),
[ze50c](../../../methods/full-drift/builds/fd-probe/artifacts/kaggle/fd-e49-50-s42ze04-20261015-r25-v1-f1/AUDITED.json),
[zi100c](../../../methods/full-drift/builds/fd-probe/artifacts/kaggle/fd-e47-50-s42zi03-20261015-r25-v1-f1/AUDITED.json).
Tính lại F1 từng lớp từ từng confusion rồi lấy mean-client, khớp `f1_per_class` tới2e−15;
không lấy pooled confusion thay mean. Bộ chấm nhanh là tolerance đã owner duyệt, cùng phạm vi final report.

Mỗi ô = **private / proxy (gap)**:

| Lớp | 20c | 50c | 100c |
|---|---|---|---|
| positionMirroring | 0,393991 /0,528722 (+0,134731) | 0,449134 /0,598378 (+0,149244) | 0,424734 /0,564305 (+0,139570) |
| timeDelayAttack | 0,201645 /0,299650 (+0,098005) | 0,057595 /0,123107 (+0,065512) | 0,021388 /0,027176 (+0,005788) |
| dataReplay | 0,772333 /0,809886 (+0,037553) | 0,736045 /0,783708 (+0,047663) | 0,735829 /0,793752 (+0,057923) |
| suddenStop | 0,811705 /0,852226 (+0,040521) | 0,750651 /0,758361 (+0,007710) | 0,717502 /0,772361 (+0,054858) |
| suddenConstantSpeed | 0,753710 /0,795500 (+0,041790) | 0,735321 /0,757507 (+0,022186) | 0,741476 /0,764974 (+0,023498) |
| constantPositionOffset | 0,875344 /0,916224 (+0,040880) | 0,861948 /0,904485 (+0,042537) | 0,858001 /0,879668 (+0,021666) |
| benign | 0,787441 /0,809278 (+0,021838) | 0,790532 /0,816676 (+0,026144) | 0,790128 /0,816324 (+0,026197) |

Ba gap lớp lớn nhất đóng60,7% /66,4% /67,1% tổng gap macro:20c gồm positionMirroring,timeDelay,suddenConstantSpeed;
50c gồm positionMirroring,timeDelay,dataReplay;100c gồm positionMirroring,dataReplay,suddenStop.
Riêng positionMirroring đóng29,8% /37,8% /37,1% tổng gap, do macro chia đều16 lớp.
Confusion còn cho thấy private nhầm positionMirroring→benign nhiều hơn proxy trung bình26.242 /55.552 /53.872 dòng/client;
dataReplay→benign nhiều hơn10.875 /20.022 /24.982 dòng/client. Đây là chênh đếm lỗi trên cùng test10.761.343 dòng/client,
không phải số record khác nhau được truyền hoặc số lỗi teacher có thể sửa độc lập.

**Hệ quả để phân biệt giả thuyết, chưa chọn phương pháp từ test:** các lớp có proxy tốt hơn private như positionMirroring
phù hợp phép thử teacher×support; timeDelay100c cả hai gần hỏng thì tăng distill cùng teacher thiếu nguồn đúng.
Không bê con số77% lỗi chung của pilot ep15 sang final50: muốn biết tỷ lệ cuối cần prediction joint teacher/private
trên holdout nội bộ tương ứng. Marginal confusion hiện tại không cho biết giao/lệch của lỗi từng dòng.
CCVR mới có thể sửa nguồn teacher; inversion mới có thể sửa support truyền; hai tác động phải tách bằng factorial §9.
Hướng cải tiến được nâng đỡ bởi **cả code và probe nội bộ trước đây**, còn bảng test này chỉ bổ sung mô tả và câu hỏi cần kiểm.
