# Chẩn đoán nguyên nhân và đối chứng tiếp theo — 31/10/2026

**Ưu tiên tách target khỏi hệ số loss, sau đó mới sửa teacher/generator.** K hiện chưa xác định learned teacher
có lợi hơn augmentation tới mức nào. Owner đã duyệtthựchiện31/10: [specL](screening-l10.md) cụthểhóa đềxuất dưới thành12run
vì codec mới cầncontrolcùngcode. Live [bàn giao](../operations/continue.md); chưa50round.
L đãhoànthành/audit: [results-l10](results-l10.md). Kếtquảủnghộfixed1 vàsofttarget;
lợirealgate nhỏ, timeDelay/tail cònđánhđổi; chưa launchđềxuấttiếphoặc50round.
Giữ [protocol](protocol.md), các [bài học](../../LESSONS.md) và [kết quả K](results-k10.md).

## 1. Bằng chứng mới từ artifact đã đóng băng

Chạy offline, không train, không sửa source K:

```bash
/home/odixe/Tools/miniconda3/envs/nckh/bin/python scripts/diagnose_kd_mechanisms.py \
  --counts runs/j-screening/train_counts.npy --pulls runs/pulls \
  --output runs/k-screening/mechanism-diagnosis.json
```

Script: [diagnose_kd_mechanisms.py](../../scripts/diagnose_kd_mechanisms.py).
JSON lưu SHA256 count file,40snapshotQ và4CM; tính lại từ toàn bộ5343client.
Snapshot sau r được dùng làm teacher cho r+1; Qr10 chưa tác động training r10.

- K8/K9: Qr1–r9 đều chỉ có **lớp13 timeDelayAttack** không khớp argmax. Gate thực thi r2–r10 vì thế chỉ tắt
  KLreal lớp13, giữ15lớp còn lại. Khối lượng lớp13=1.924.480/37.361.273=**5,151% train**.
  H3/K2 cũng có cùng mask này. Chưa có bằng chứng gate thích nghi theo từng input/client trong thí nghiệm này.
- Teacher thực dùng tại r10 của K8/K9: Qr9[13,13]=.342320/.337730;
  Qr9[13,0]=.573633/.578093. Gate vẫn dùng hàng này cho **synthetic** và `ce_t` trong `raw`.
- Seed42, gate so augmentation: benign→timeDelay **13.611 vs180.017**, timeDelay→benign **427.927 vs329.660**.
  Seed43: **12.357 vs177.503** và **429.800 vs331.273**. Đây là hai chiều đánh đổi, chưa phải chứng minh giới hạn feature.
- 848client<256mẫu chiếm **15,871% số client**, nhưng chỉ **0,359% dòng train**.
  Trong gộp Q lớp13, nhóm này chỉ đóng góp0,335% trọng số; lớp14 chỉ0,065%.
  Gộp theo mẫu có thể phù hợp pooled nhưng đại diện yếu cho nhiều client nhỏ; chưa có đối chứng nhân quả về weighting.
- Một epoch tương ứng **2–1489 bước/client**. Dose64 cung cấp1,000–1,684 dòng synthetic/dòng thật tùy kích thước.
  Loss real/synthetic lấy mean riêng; dose32→64 **không đồng nghĩa nhân đôi hệ số loss**. Nó đổi coverage, phương sai và
  hỗn hợp input mà BN thấy. Chênh lệch số bước cũng chưa chứng minh optimizer hỏng.

Không có gradient tách loss, trainF1, heldout prediction hay fullprivateparameterbank của K trong bộpull hiện tại.
Không thể dùng snapshot nhỏ để tái dựng các phép đo đó. Những mục bên dưới là giả thuyết cần kiểm.

## 2. Cơ chế cần phân biệt

Ký hiệu: `C`=CEreal có logit adjustment, `S`=loss synthetic, `R`=KLreal sau gate,
`a=1/(C+ce_t).detach()`. Cả C và ce_t là mean minibatch từng client.

K0/K1 dùng `C+a*Ssoft`; K4/K5 dùng `C+Shard`; K8/K9 dùng `C+a*(R+Ssoft)`.
Do đó augmentation thắng combo **không cô lập** softtarget; nó thay cả target lẫn hệ số.
`ce_t` vẫn tính trên tất cả dòng thật, kể cả dòng bị gate loại khỏi KLreal.

Theo đạo hàm logit, CEreal cho `softmax(z+shift)-onehot(y)`; KDreal cho `softmax(z)-Q[y]`.
Chúng dùng hai hệ xác suất và có thể ngược hướng. Argmaxteacher=y không bảo đảm cùng hướng trên mọi mẫu.
Tuy nhiên logit adjustment đang có chủ đích sửa prior; khác hệ xác suất **chưa đủ gọi là bug**.
Ở hệ số cố định và target detach, KL(Q||p) và softCE(Q,p) có cùng gradient; số loss khác entropyteacher.

| nguyên nhân giả định | ảnh hưởng dự kiến | bằng chứng bác bỏ cần tìm |
|---|---|---|
| Q sai trên lớp khó | truyền thêm benign vào timeDelay, kể cả sau realgate | sửa riêng synthetic target không giảm xung đột gradient/không cải thiện heldout |
| adaptive coefficient không phù hợp | sức nặng synthetic/KD đổi theo lớp, client và độ học được; trộn công trạng teacher với loss scale | fixed vsadaptive không đổi gradient tương đối hoặc chất lượng ở cảhard/soft |
| teacher gộp theo lớp mất cấu trúc input | cùng Q cho mẫu dễ/khó; teacher đúng trung bình vẫn hại một vùng | postepoch/input-aware probe không cải thiện conditional reliability/heldout soQ hiện hành |
| teacher online và prior lệch mục tiêu | Q trộn trạng thái đầu/cuối epoch, train-mode BN/dropout; CE/KD có thể xung đột | postepoch eval và probe prior không giảm lỗi teacher/xung đột hoặc không đổi utility |
| synthetic lệch jointdistribution | hardlabel ép học vùng không giống real; ảnh hưởng BN | kiểm tra joint/flag tốt, can thiệp đúng lỗi không giúp heldout so generator cũ |
| nhiều client nhỏ ít ảnh hưởng teacher | pooled và clienttail có thể ưu tiên khác nhau | weighting khác không giúp tail hoặc hại pooled ngoài cổng |
| temporal drift/feature khó phân biệt | prior hay calibration khác khả năng biểu diễn | train/temporalheldout đều tách tốt nhưng test vẫn kém: cần kiểm shift thay vì kết luận feature thiếu |

## 3. Batch ưu tiên: tách target × hệ số × realKD

Full5343xe,10round/cosine50, DAGSNetW1, dose64copula, mọi thiết lập chung giữ nguyên.
Seed42/43 cùng từng đối chứng. Chưa chọn seed/round/HP tốt nhất theo test.

| cell | synthetic target | hệ số synthetic | realKD | đối chứng trực tiếp |
|---|---|---|---|---|
| C0, K4/K5 đã có | hardlabel | 1 | tắt | augmentation mạnh |
| C1, **mới** | Q[y] | 1 | tắt | C0: softtarget; C3: scale |
| C2, **mới** | hardlabel | a | tắt | C0: scale; C3: softtarget |
| C3, K0/K1 đã có | Q[y] | a | tắt | combo hiện hành |
| C4, **mới** | hardlabel | 1 | gate, hệ số1 | C0: thêm riêng gatedrealKD |
| C5, **mới** | Q[y] | 1 | gate, hệ số1 | C1: thêm riêng gatedrealKD; C6: scale |
| C6, K8/K9 đã có | Q[y] | a | gate, hệ sốa | ứng viên pooled hiện hành |

**4cell mới×2seed=8run**, nếu đủ điều kiện tái dùng đối chứng K.
Hệ số1 chọn vì khớp augmentation và warmup r1 hiện có; không xuất phát từ quét test.
C2 vẫn thuQ để tính a, dù synthetic dùng hardlabel. C4 phải thuQ cho gate, dù synthetic không dùngQ.
R1 mọi cell dùng cùng `C+Shard`, teacher chưa tồn tại; ablation bắt đầu r2. Giữ forward real+synthetic chung.
Không đổi denominator realgate: mean vẫn chia tổng dòngreal, không chia riêng số dòng giữ lại.

Kết quả cần đọc:

- C1−C0 và C3−C2: giá trị softteacher ở hai chế độ scale.
- C2−C0 và C3−C1: ảnh hưởng adaptive ở hard/soft; báo tương tác, không cộng gain tuyến tính.
- C4−C0 và C5−C1: gatedrealKD có giúp khi giữ target/scale cố định?
- C6−C5: scale adaptive có giải thích lợi của gate hiện tại?

Tái dùng K chỉ khi source/default/loss/RNG/grouping/data/precision đúng phép so sánh. Không chạy lại cell đã audit chỉ để đủ bảng.
Nếu phải thay codec/đường train, cần control mới cùng code; sốrun không còn mặc định8.
Không kết luận nhân quả từ so sánh khác seed hoặc khác endpoint.

## 4. Instrumentation: đo ngay trước khi mở rộng batch

Chọn cố định client theo **traincount** và scenario/run, không chọn vì F1test kém.
Probe thưa tại r1/r2/r5/r10 trên những minibatch cố định của các client này; không dùng probe để cập nhật model/RNG/BN.
Nếu chưa chứng minh đo không đổi đường train, chạy shadowprobe độc lập; không sửa ngầm control.

- Log riêng C, KLreal trước/sau gate, S, `ce_t`, a; quantile theo cohort/lớp. Loss của target khác nhau không so trực tiếp như lỗi tổng quát hóa.
- Gradient norm của C, aR, aS và cosine trên **tham số chung**, tính trướcclip; ghi tỷ lệ cosine<0 theo lớp/cohort.
  Gradient trên logit chỉ probe cục bộ bổ sung, không thay gradient tham số. R=0 thì cosine không xác định, không gán0 rồi lấymean.
- Log update norm/appliedsteps, AMPskip, gradient trướcclip, clippingfraction. Tổng trainloss nhỏ không đủ chứng minh tối ưu tốt.
- Đo CE/margin/CM train và temporalheldout trong **nhánh diagnostic riêng**, cùng client/support giữa các cell.
  Trainingreport vẫn train+val đúng protocol. Muốn heldout độc lập phải fit mới trên foldtrain, giữ foldval chưa học;
  không chấm foldval bằng model K đã train fold đó rồi gọi là generalizationgap.
- Dữ liệu/probe/prediction từng mẫu giữ tại client. Báo tổng/cohort scalar và CMgộp; không thêm packetrawdata hoặc privateweights.
  Thời gian probe/eval hạch toán riêng khỏi training; bytes diagnostic tách khỏi method.

## 5. Can thiệp tiếp theo, chỉ mở khi phép đo hỗ trợ

### A. Chặn teacher sai ở kênh synthetic

So trực tiếp gate hiện hành với **synthetic fallback**: nếu argmaxQ[y]≠y thì synthetic target là onehot(y), ngược lại giữQ[y].
Rule dùngQtrain, không hardcode lớp13; giữ realgate, a, denominator, generator và seed nhưcontrol.
Chỉ sửa targetsynthetic; không đồng thời sửa `ce_t`/hệ số/generator.
Đây khác H4(onehotmọilớp,dose32,realkhungated) và K4(bỏQ/adaptive); không lặp thất bại cũ cùng cấu hình.
Kỳ vọng: giảm timeDelay→benign mà giữ phần regularization hữu ích. Bác bỏ nếu không giúp heldout và utility hai seed,
hoặc chỉ đổi falsepositive benign lấy recalltimeDelay mà pooled/tail xấu đi. Báo đủ hai chiềuCM.

### B. Sửa chất lượng teacher trước khi thêm dung lượng packet

1. Cùng classQ/schema/aggregation, thay **online accumulation** bằng một lượt **postepoch eval** trên real local, frozenBN/dropoutoff.
   Tốn localforward; không tăng sốôQ. So với onlinecontrol, giữ toàn bộ loss/generator. Báo bytes và forwardtime.
   Đây là can thiệp gộp về thời điểm+evalmode; nếu có lợi, tách timing/mode tiếp, không quy toàn bộ cho stale.
2. Nếu gradient xung đột tập trung ở client prior lệch: diagnostic2×2 `LAreal bật/tắt × realKD bật/tắt`, synthetic cố định.
   TắtLA là ablationmethod, không thay protocol chung. Đo gradient và heldout trước khi nghĩ tới priorcorrection mới.
   Không tự coi KD phải dùng `z+shift`: teacher hiện ở globalspace, đổi cả student/teacher có thể tạo priorbias khác.
3. Nếu client nhỏ bị teacher đại diện yếu: so gộp hiện hành `w_kc=n_kc` với `w_kc=sqrt(n_kc)` cho n>0, normalize từng lớp.
   Đây là phép thử độnhạy, **không chọn exponent trên test**. Giữ cùng table/byte, không đổi generator hay syntheticloss.
   Chỉ mở sau khi có reliability theo cohort; không tin client2mẫu như expert mạnh chỉ vì cân bằng trọng số.

Không mở lại bucketB8/routergrid: K6/K7 thất bại so combo, chưa có evidence partition phù hợp.
Không tăng anchorbank/teacherensemble chỉ vì agreement đẹp: J đã bác bỏ utility của adaptation cụ thể đó.

### C. Synthetic: kiểm phân phối trước khi ghép generator

Trong nhánh diagnostic foldtrain→foldval, sinh đúng copula hiện hành từ thống kê foldtrain, không fit cảfoldval.
Đo từng lớp/cohort: marginal, tương quan/joint, biên hợp lệ, cờNaN rời rạc và nhất quán với feature liên quan.
Các feature đã scaled: rule sửa phải theo scaler/schema thật, không áp quan hệ vật lý thẳng lên zscore.
Discriminator real-vs-synthetic phải cân bằng lớp, split độc lập, không dùng ID/fold/time làm shortcut.
AUC phân biệt cao chỉ chỉ ra khác phân phối, **không chứng minh** synthetic gây hại cho task.

Nếu có lỗi support rời rạc/joint cụ thể, thử **copula+repair đúng lỗi đã đo** so copula gốc, giữ target/loss/rows nhưnhau.
Repair không rounding bừa cờ về0/1 nếu cờ đã qua transform; phải khớp observedsupport theo schema.
Đo taskheldout và ảnh hưởngBN cùng chất lượngsample. Chỉ có AUC/moment đẹp chưa đủ qua utilitygate.

Để tách dose/BN: sau instrumentation, diagnostic2×2 `dose32/64 × BNrunning update hiện hành/realonly`,
giữ coefficient/target cố định, vẫn cập nhật runningbuffer từreal ở cảtrain/eval path tương ứng.
Thay runningstatistics phải định nghĩa rõ forwardnormalization; không gọi đây là tách toàn bộ BN nếu trainforward vẫn mixed.
Không mặc định triển khai BN/headsplit từngFAILv1; chỉ mở thínghiệm cụ thể khi thấy BNshift và dùngcontrol cùngcode.

Không trộn GMM/MADE/inversion/mix/cluster để dòwin: đã có bài học v1 và H/J; cần cơ chế mới có falsifier.
Không gửi dữ liệu thật sang server hoặc dùng oracle rawsharing làm phương pháp triển khai.

## 6. Đánh giá, codec và thứ tự thực hiện

- Headline sàng lọc r10; cùngLRcosine50. Báo10metric,16lớp, mean/p10/median/min theo xe, cohort<256 và≥16384,
  scenario/run, benign↔timeDelay, bytesbootstrap+upload+download, train/eval/probetime riêng.
- Giữ cổng đã khóa: ΔF1≥.010,p10Δ≥−.010 vsG4 cùngseed. Đồng thời báo ΔvsK4/K5 vàK8/K9;
  vượt nền cũ không tự chứng minh thắngcontrolmạnh. Hai seed chưa đủ gọi significant.
- Officialtest đã được đọc nhiều lần; kế hoạch mới là nghiên cứu thích nghi theo kết quả cũ, không gọi freshconfirmatorytest.
  Khóa cell/HP/endpoint trướclaunch, không tối ưu threshold/best-round trêntest; xác nhận dài hạn/seedmới ở giai đoạn riêng.
- **Codecpreflight:** nhánh classQ hiện cộng `_csum`fp32 rồi gộpfloat64; có tạo bodyfp16 nhưng không decodebody để train.
  Do đó1088B/r là accountingpayload, chưa chứng minh train đúng theo precision truyền thực tế. Conditionalbranch đãdecode, không suy sangclassQ.
  Trước claim hiệu quả giao tiếp thực thi, phải encode→decode thật cảhai chiều, kiểm finite/rowmass/ΔQ/gradient vàbyte thực.
  Nếu thay đường này, khóa thành phiên bản mới và dùngcontrolcùngcodec; giữ nguyênsource/resultsK.
- Thứ tự: **offlineQ/CM đã làm → instrumentation+codecpreflight → factorial8run nếu controltươngthích → canthiệpA/B/C có bằngchứng**.
  Khi đã giải thích cơ chế, chọn gate/augmentation/candidate để kiểm50round cùngseed; không gộp codec/generator/loss mới vào xác nhận dài hạn.
  Mixedarchitecture/contactdrop/compressionbootstrap kiểm sau khi utilitymechanism rõ. Chưa có ủy quyềnlaunch50round từ đềxuất này.

Không việc tay bắt buộc hiện tại. Manifest/HP/sourceL đãkhóa, notebookprivateđãgenerate; launch theo specL, khôngtheo bảngđềxuất8run cũ.
