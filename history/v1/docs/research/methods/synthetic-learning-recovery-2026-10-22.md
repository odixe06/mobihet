# Sinh synthetic giúp private học — chẩn đoán, probe và cổng mới

22/10/2026. Owner cho phép thử local/Kaggle sau bằng chứng; yêu cầu notebook nhanh, private độc lập, R1 hạn chế.
`cvl` và `xo` giữ FAIL; run cuối giữ phương pháp đã chốt. [Nguồn + phân tích loss/r50](ccvr-budget-primary-sources-2026-10-22.md).
**P23–28 là snapshot trước compact03:00Z.** Owner đã tiếp tục; [P29–31](synthetic-gradient-diagnosis-2026-10-22.md)
và [bàn giao hiện hành](../../archive/operations/continuation-2026-10-22.md) giữ kết quả mới. Chưa launch online FL mới.
[Bàn giao trước compact](../../archive/operations/continuation-synthetic-2026-10-22.md) ·
[Rà build cũ / hướng không lặp](synthetic-build-history-audit-2026-10-22.md).

## 1. Quyết định phiên này

- Cap tổng byte/client ≤1,5× `xa` tại **cả 15 và 50 round**; từng contact ≤**231.000 B**, tính bootstrap.
- Được tìm nhiều phương án trao đổi; không tự coi tăng contact miễn phí hoặc đã chứng minh deadline.
- Ưu tiên khả năng sửa học của private. Teacher F1, độ nhận mẫu, covariance fit và codec chỉ là cổng cơ chế.
- Không dùng official test chọn cấu hình. R50 chỉ phân tích mô tả; chọn phương án bằng fixture old/group đã khóa.

## 2. Nguyên nhân: điều đã biết và điều còn phải kiểm

**Private chững trong khi teacher vẫn lên.** R30→50 ở20/50/100c: private +0,0038/+0,0018/+0,0036;
proxy +0,0184/+0,0131/+0,0138. Gap r50 là0,0283/0,0247/0,0235. PositionMirroring đóng30–38% gap;
timeDelay100c cả hai model đều rất yếu. Lag một snapshot và AMP skip thấp không giải thích hết chênh lệch.
Không suy từ đây rằng cần tăng LR, kéo dài50 round, hoặc mọi client thiếu capacity.

**Có hai loại lỗi cần hai loại thay đổi.** Probe21/22 seed42 ep15: phần lớn lỗi private cũng là lỗi teacher.
KD trên nhiều mẫu hơn chỉ chở lại cùng hàm teacher; không thêm nhãn mới. Với teacher đúng mà private sai,
input Gaussian hiện tại lại đặt gradient bất đồng chủ yếu ở lớp tốc độ, khác khối lỗi thật. Hai nút thắt cùng tồn tại.
Tỷ lệ77% là của pilot ep15, chưa đo joint correctness ở final r50; không chuyển con số này sang final.

**Cách sinh phải thay thông tin và vị trí học.** Covariance riêng lớp giữ hình dạng decision support;
covariance chung `cvl` đánh đổi lớp tốc độ để cứu lớp yếu. Inversion theo covariance lớp có thể tạo input giúp
private học hàm teacher tốt hơn, nhưng việc teacher nhận nhiều mẫu hơn chưa chứng minh gradient hữu ích.
Private còn CE own, KL real, KL synthetic và BN trộn; phải tách tác động trước khi ghép vào training đầy đủ.

## 3. Probe23 local — giữ covariance riêng, giảm chiều chung

[Script](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe23.py),
[artifact](../../methods/full-drift/results/probe23-xa-s42-e15.json). Audited xa seed42 ep15;20 client ×16.384 dòng;
old/group65.536 dòng mỗi tập; backbone đóng băng. Không đọc official test.

P trực chuẩn gồm16 hướng của head hiện tại + hướng random cố định. Client gửi `(n,ΣPᵀh,Σ(Pᵀh)(Pᵀh)ᵀ)`
**riêng từng lớp**, server cộng trước khi center. Head mới bị giới hạn trong P; baseline head biểu diễn được chính xác
trong P. Đây là thay estimator/subspace có kiểm, không phải full CCVR lossless. Basis cần tái dựng thống nhất theo snapshot.

Mỗi client serialize fp32 sums bằng codec CRC/base/version, đọc lại decoded values rồi mới gộp. Bank200.000 virtual
features cố định; Adam3.000 bước ×batch4.096. Khác probe22 sinh noise mới mỗi bước; full/diag/shared controls chạy lại.

| Teacher | Group F1 | Benign | Old F1 | Uplink statistics + downlink head |
|---|---:|---:|---:|---:|
| Head gốc | 0,8420 | 0,7948 | xem artifact | — |
| Full CCVR, diagnostic | 0,8912 | 0,8074 | 0,8700 | 2.138.312 B |
| Shared covariance, diagnostic | 0,8475 | 0,7769 | 0,8371 | dùng full packet để cô lập estimator |
| Diagonal, diagnostic | 0,8190 | 0,6648 | 0,8056 | dùng full packet để cô lập estimator |
| Project16 | 0,8508 | 0,8085 | 0,8390 | 10.952 B |
| Project32 | 0,8621 | 0,8100 | 0,8491 | 38.088 B |
| **Project48** | **0,8689** | **0,8107** | **0,8573** | **81.608 B** |
| Nyström2 + diagonal dư | 0,8137 | 0,5766 | 0,7949 | 82.120 B |
| Nyström4 + diagonal dư | 0,8400 | 0,6667 | 0,8245 | 114.888 B |

Project48 giữ covariance riêng, teacher hơn gốc+0,0270, giữ benign; packet calibration nhỏ hơn full26,2×.
Nyström2/4 không giữ utility dù công thức cộng được ⇒ loại cấu hình này, không chạy online để tìm round thắng.
Project16 quá ít sức chứa; Project32 lợi vừa ngưỡng teacher, Project48 ưu tiên kiểm chuyển giao.
Số above là offline single-snapshot; không chứng minh lợi từ đầu tới ep15.

**Khe còn lại:** refit head trên pooled real là oracle không deploy: Project48 group0,8895, full2560,9051.
Trong cả hai, Gaussian feature fit thấp hơn real refit. Nén có giá về sức chứa; cả full covariance vẫn chỉ giữ moment bậc1–2.

## 4. Probe24 — teacher × nguồn mẫu, đo private thực sự học được gì

[Script](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe24.py). Factorial:

| | Gaussian input từ pooled moments | Input inversion theo class moments |
|---|---|---|
| Teacher gốc | matched continuation control | chỉ thay support |
| Teacher Project48 | chỉ thay teacher | thay cả nguồn tri thức + support |

Inversion: noise Gaussian input66 → tối ưu input, khớp target sampled từ covariance lớp của `Pᵀh_teacher`,
giữ prior khoảng cách chuẩn hóa tới noise ban đầu;13 feature m4b giữ0. Cùng input bank4.096 dòng cho mọi client,
uniform class components. Không khởi tạo bằng record thật; không đưa teacher latent trực tiếp vào private.
Thêm một diagnostic hard-CE trên cùng bank inversion để kiểm nhãn thống kê có ích hơn soft teacher không;
không mặc định hard label đúng cho mọi mẫu invert.

Private backbone **và BN statistics đóng băng**; chỉ Linear cuối train1.000 bước, batch512 own+256 synthetic.
CE own có logit adjustment; real/synthetic KD như cùng nguồn teacher. Denominator KD dùng CE của proxy frozen
trên own, khác full harness có proxy train liên tục. Đây là **head adaptation diagnostic**, không phải online ep15.
Mọi arm cùng own rows, stream, batch, step; cache feature một lần/private/source để tránh chạy backbone mỗi bước.

Seed42 local5 client đầu (đã audit predictions→CM→10 metric50 lượt):

| Arm | Mean private group F1 | Benign | Min client F1 |
|---|---:|---:|---:|
| Weights ban đầu | 0,8269 | 0,7798 | 0,8150 |
| Teacher gốc + Gaussian | 0,8254 | 0,7819 | 0,8095 |
| Teacher gốc + inversion | 0,8261 | 0,7827 | 0,8097 |
| Project48 + Gaussian | 0,8334 | 0,7938 | 0,8183 |
| Project48 + inversion | 0,8367 | 0,7959 | 0,8234 |
| Project48 + inversion hard label | 0,8352 | 0,7890 | 0,8234 |

Inversion nâng class agreement của teacher cũ0,440→0,797, nhưng private chỉ+0,0007 so matched Gaussian.
Sau hiệu chỉnh teacher, inversion thêm+0,0033. **Mẫu được nhận nhiều hơn chưa đồng nghĩa giúp học tương ứng.**
Hiện tín hiệu mạnh hơn nằm ở teacher (+0,0080 với cùng Gaussian). Chưa nhận inversion/hard-label làm cải tiến.
Đây là screening5 client, đã được thay bằng kết luận đủ20 client local và hai seed Kaggle dưới đây.

## 5. Kaggle probe — cổng trước push, 22/10

[Generator](../../../methods/full-drift/builds/fd-probe/scripts/gen_ccvr_probe.py),
[audit](../../../methods/full-drift/builds/fd-probe/scripts/check_ccvr_probes.py).
Notebook private `thuyanh316/mobihet-fd-ccvr-probes-20261022-v1`: hai T4, seed42/43 song song,
mỗi seed chạy Probe23 rồi Probe24 đủ20 client. Image pin từ runtime; sha mọi input; fixture đã cache;
không compile lại backbone chỉ để chạy probe ngắn. Quota live4,20h, budget staged1,20h gồm margin.
Đây là xác nhận cơ chế trên T4 + seed43, chưa phải FL từ đầu.

**Cổng cơ chế khóa trước push, seed42 local đã dùng chọn Project48:**

1. Teacher Project48: group ≥ gốc+0,020 và benign≥gốc−0,005 **cả hai seed**.
2. Chuyển giao teacher: Project48/Gaussian − original/Gaussian mean-private≥+0,005, benign≥−0,005,
   min-client≥−0,010 **cả hai seed**. PASS mới đủ căn cứ thử online Project48.
3. Nhận inversion riêng chỉ nếu calibrated/inversion − calibrated/Gaussian≥+0,005 với cùng các cổng bảo toàn,
   cả hai seed. Nếu fail, không ghép inversion chỉ vì class agreement/loss tốt.
4. Hard label chỉ diagnostic; chưa launch online nếu không hơn soft target với cùng support và cổng bảo toàn.
5. Muốn nhận phương pháp: online từ đầu s42 ep15 so xa, Δ≥+0,010, benign≥−0,005, worst≥−0,010,
   đủ byte/cap; rồi s43, horizon và mixed-architecture riêng. Không lấy PASS frozen head thay cổng này.

**Kết quả terminal đã pull/audit:** version1,59 file,0 thiếu; source stage và executed notebook khớp hash,
weights/fixture đúng input,400 lượt predictions→CM→10 metric. [Collector](../../../methods/full-drift/builds/fd-probe/scripts/collect_ccvr_probes.py)
ghi [AUDITED.json](../../../methods/full-drift/builds/fd-probe/artifacts/kaggle/fd-ccvr-probes-20261022-v1/AUDITED.json).
Toàn notebook360,09s; transfer269,99/265,74s; fit Project48~3,70s/seed. Không dùng trạng thái COMPLETE thay audit.

| Seed | Teacher gốc | Project48 | Cổng teacher |
|---|---:|---:|---|
| 42 | 0,841941 | 0,869723 | PASS gain + benign |
| 43 | 0,842980 | 0,882092 | PASS gain + benign |

Mean macro-F1 group của20 private; benign là mean F1 benign, worst là min macro-F1 client:

| Seed | Arm | Mean | Benign | Worst |
|---|---|---:|---:|---:|
| 42 | Weights ban đầu | 0,823466 | 0,774651 | 0,801766 |
| 42 | Gốc/Gaussian | 0,823250 | 0,777371 | 0,799867 |
| 42 | Gốc/inversion | 0,824325 | 0,777394 | 0,807610 |
| 42 | Project48/Gaussian | 0,828956 | 0,783164 | 0,785100 |
| 42 | Project48/inversion | 0,831551 | 0,783876 | 0,794854 |
| 42 | Project48/inversion hard | 0,830843 | 0,770960 | 0,796630 |
| 43 | Weights ban đầu | 0,815583 | 0,769499 | 0,793179 |
| 43 | Gốc/Gaussian | 0,821053 | 0,765048 | 0,794062 |
| 43 | Gốc/inversion | 0,820600 | 0,762091 | 0,796757 |
| 43 | Project48/Gaussian | 0,822977 | 0,762766 | 0,770246 |
| 43 | Project48/inversion | 0,824979 | 0,760570 | 0,779076 |
| 43 | Project48/inversion hard | 0,823737 | 0,748747 | 0,786674 |

Teacher transfer **FAIL cả hai seed**: s42 mean+0,005707 nhưng worst−0,014767; s43 mean+0,001924,
worst−0,023815. Inversion riêng chỉ+0,002595/+0,002002 so cùng teacher/Gaussian, **FAIL gain**.
Hard label không cứu mean, benign giảm rõ. Không nới cổng, không launch online các adaptation này.
Kaggle khác local vài phần nghìn; kết luận cổng nhất quán, không tuyên bố bit-exact cross-device.

## 6. R1: phần đã đo và phần phải giải quyết trước run online

Project48 upload78.440 B, head down3.168 B, tổng81.608 B gồm codec header; không có record activation rời client.
Basis chung là điều kiện, SecAgg hiện mô phỏng. Chưa có mask/crypto transport hoặc contact/deadline proof.
14/49 lần calibration dự toán thêm1,143/3,999 MB/client; so xa5,153/12,95 MB tương ứng khoảng×1,222/×1,309,
**50 round là dự toán theo baseline screening, chưa phải byte đo của run mới**.

Không ghép thêm81,6k vào contact q3 hiện tại rồi gọi R1 đạt: xa peak15=228.658 B; final20/50/100c peak50=
246.556/255.895/253.731 B. Bootstrap1.812.520 B cũng đã vượt cap mới. R1 mới còn đòi sửa transport baseline.

Phương án cần kiểm trước online:

- Bootstrap proxy từ seed1042 + source/runtime/model-state hash; tái dựng chính xác thay truyền1.593.440 B.
  Chỉ được bớt byte sau encode/decode và bit-exact test, giữ RNG training. Gaussian/prior vẫn trả đủ phí.
- Calibration là transaction riêng≤B2; q3 upload/downlink chỉ gộp nếu tổng≤cap, nếu không cần contact riêng.
  Một aggregation trọng số/epoch vẫn giữ; **số contact tăng**, phải báo số, ACK/header, chờ cohort và snapshot age.
- Cùng lịch contact cho baseline/candidate; thiếu contact/deadline thì không commit. Không dùng chia frame trong
  cùng contact để vượt cap tổng, cũng không gọi synthetic utility PASS là chứng minh radio R1.

Chưa launch online: cổng chuyển giao chưa đạt, transport chưa giải quyết. Owner đã nối nghiên cứu P29–31
sau compact; toàn bộ cổng vẫn giữ, kết quả mới ở bàn giao hiện hành.


## 7. Đủ20 client và can thiệp nguyên nhân

[Probe24 đủ20 client](../../methods/full-drift/results/probe24-xa-s42-e15-all.json), đã audit200 lượt predictions→CM→10 metric.

| Arm | Mean group F1 | Benign | Min client |
|---|---:|---:|---:|
| Teacher gốc + Gaussian | 0,823140 | 0,777331 | 0,799597 |
| Teacher gốc + inversion | 0,824205 | 0,777025 | 0,807662 |
| Project48 + Gaussian | 0,828478 | 0,781511 | 0,783014 |
| Project48 + inversion | 0,831036 | 0,782349 | 0,792128 |
| Project48 + inversion hard label | 0,831013 | 0,769802 | 0,797924 |

Teacher-only mean+0,005338 nhưng min−0,016583 ⇒ trượt cổng bảo toàn. Inversion thêm+0,002558,
chưa đạt cổng+0,005. Hard label không tăng mean so soft và benign mất−0,012547 so cùng support soft.
Không kéo dài các adaptation này để tìm round thắng.

Joint predictions20×65.536: đổi teacher/Gaussian sửa35.050, làm hỏng29.685 lượt; thêm inversion sửa10.697,
hỏng9.532. Đây là lượt model×dòng, không phải số record độc lập. Client18 benign→suddenStop từ6 lên1.216,
benign→constantPositionOffset62→711; F1 suddenStop giảm dù true-positive tăng. Lỗi chính ở precision/false-positive,
không thể giải thích đơn giản bằng thiếu số mẫu lớp tấn công.

**Can thiệp estimator:** dùng đúng cùng dòng đã sample, nhân moment từng client×lớp bởi N_kc/m_kc trước SecureSum;
không thay byte/step. [Probe26 teacher](../../methods/full-drift/results/probe26-classweight-teacher-s42.json): teacher group0,8643,
benign0,8177. [Transfer](../../methods/full-drift/results/probe26-classweight-transfer-s42.json), audit200 lượt:
Project48/Gaussian mean0,826598, benign0,789101, min0,790858. So estimator không trọng số: benign+0,00759,
min+0,00784, nhưng mean−0,00188; so matched control mean chỉ+0,00346 < cổng+0,005.
Sai trọng số là yếu tố góp phần, **không đủ giải thích toàn bộ FAIL hoặc plateau phương pháp cũ**.

Các tỷ lệ population ở probe26 là population **fixture2,5%**. Đã đối chiếu manifest full40.014.264 dòng của xa:
benign22,210%, suddenStop1,964%, khác nhẹ fixture21,928%/2,023%; production phải dùng full counts,
không thay bằng counts fixture. Wrapper prior-only đã viết nhưng **chưa chạy**; chưa tách riêng ảnh hưởng prior khỏi mean/covariance.

### Probe27 — support oracle đã xong, chỉ chẩn đoán

[Artifact](../../methods/full-drift/results/probe27-real-support-oracle-s42.json), local s42 đủ20 client,
audit200 lượt predictions→CM→10 metric. Cùng batch, class schedule, teacher, private backbone/BN và số bước;
thay bank inversion bằng dòng thật gộp có cùng lớp. Field arm `inversion` trong artifact được
`support_override` ghi rõ là **real pooled support**, không phải kết quả generator.

| Arm | Mean | Benign | Worst |
|---|---:|---:|---:|
| Gốc/Gaussian | 0,823140 | 0,777331 | 0,799597 |
| Gốc/real support oracle | 0,827264 | 0,786210 | 0,813856 |
| Project48/Gaussian | 0,828478 | 0,781511 | 0,783014 |
| Project48/real support oracle | 0,841418 | 0,792690 | 0,818250 |
| Project48/real support + true-label hard oracle | 0,846903 | 0,725241 | 0,814159 |

Cùng teacher mới, real support thêm+0,012940 mean và+0,035236 worst so Gaussian; inversion chỉ+0,002558.
Nguồn mẫu làm thất thoát phần lớn khả năng chuyển giao ở phép thử này; thêm hard label vẫn gây false positive benign.
Đây là can thiệp support có kiểm, **không deploy được theo R3**, không chứng minh một nguyên nhân duy nhất của final50.

### Probe28 — một/hai query gradient chung đã xong, teacher gate FAIL

[Script](../../../methods/full-drift/builds/fd-probe/source/proj/fd_probe28.py),
[artifact](../../methods/full-drift/results/probe28-head-gradient-s42.json). Client tính summed gradient G,
diagonal Hessian H và global majorizer D trên **cùng snapshot backbone/head**, inclusion weights từ manifest full train.
Decoded aggregate gradient khớp centralized autograd max|error|=1,71644e−9.

Thay đề xuất fixed step0,3 bằng quy tắc có chặn trước khi chạy: softmax Hessian≤0,5I,
`h̃h̃ᵀ ≤ diag(|h̃| ||h̃||₁)`, nên `D_j=0,5 E_w[|h̃_j| ||h̃||₁]` là majorizer cho mọi head.
Primary `d=G/(H+0,01 mean(H))`, `η=<G,d>/<d,Dd>`, update `A←A−0,99ηd`.
Không chạy grid0,1/0,3/1. Two-query tính lại gradient tại head mới, không tái dùng gradient cũ như chính xác.

| Head | Group F1 | Weighted sample train CE |
|---|---:|---:|
| Gốc | 0,841914 | 0,366949 |
| Safe gradient | 0,836962 | 0,360559 |
| Safe diagonal — primary | 0,841337 | 0,365881 |
| Safe diagonal — hai query | 0,841142 | 0,364829 |

Assert train CE không tăng PASS; utility teacher không tăng ⇒ **FAIL**, không launch gradient-only adaptation.
Packet thực gồm normalization + gradient + diagonal Hessian + majorizer + header/CRC:
**33.968 B lên +16.480 B xuống/query**, hai query100.896 B. Số32.940/98.840 B dự toán trước đó chưa tính majorizer,
đã thay bằng số serialize đo được. SecAgg/contact/deadline vẫn chưa proof.

## 8. Giới hạn suy luận và việc chưa mở lại

- R50 đã đo proxy còn tăng trong khi private hấp thụ chậm; không có joint-final correctness/intervention chứng minh duy nhất một căn nguyên.
- CCVR nén48 chiều giữ lợi teacher nhưng không tự chuyển thành lợi private; benign/worst guard ngăn nhận gain trung bình đánh đổi client.
- Weighting và support là hai yếu tố đã can thiệp riêng; prior-only chưa chạy, online BN/backbone chưa được phép suy từ head probe.
- Rà [history §8](synthetic-build-history-audit-2026-10-22.md): unconditional shared-head gradient
  `G(A)=E[p_A(h)h̃ᵀ]−E[onehot(y)h̃ᵀ]` chỉ mang class means/counts trong label term.
  Perturb head triệt tiêu label term trong hiệu gradient; near-optimum cancellation không xác định được synthetic distribution.
  Chưa có căn cứ mở synthesis chỉ khớp một gradient chung. Class-conditional projected gradients cũng **chưa utility/packet PASS**.
- Sau owner tiếp tục: đọc bàn giao/history trước, chốt cơ chế thay đổi và falsifier; không lặp y nguyên các adaptation FAIL.
  Chưa có phương án mới được nhận thay `gkd + la + kdpx`, proxy `q3_zl`.
