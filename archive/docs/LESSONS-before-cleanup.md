# LESSONS — bài học kỹ thuật (dữ liệu, model, train, đánh giá, code)

Đọc sau [CONTEXT](CONTEXT.md), cùng [bài học vận hành](docs/operations/lessons.md). Bài học kế thừa:
`../mobihet-features/LESSONS.md` (dataset v4), [history/v1/LESSONS.md](archive/legacy-v1/LESSONS.md) (MobiHet v1, grep theo chủ đề; FL/KD/synthetic/đánh giá trên DAGSNet) và [bài học vận hành v1](archive/legacy-v1/docs/operations/lessons.md).

## Kế thừa — áp dụng thẳng cho repo này

- Model client chỉ học nhãn của chính nó đạt đỉnh trên test có prior toàn cục sau 1–2 epoch rồi giảm (mobihet 13/09). Ở đây model client
  chấm trên test của chính xe ⇒ prior khớp hơn, nhưng Sybil test 17,67 % vs train+val 8,80 % vẫn lệch theo thời gian (fullcount31/10).
- Luôn báo F1 train cạnh F1 test; sụp đột ngột khi đổi cấu hình là dấu hiệu tối ưu hỏng (HGB L2 = 0, mobihet-features 28/10).
- Val "xe chưa thấy" ≈ val cùng xe khi train gộp nhiều xe; không lộ được học thuộc theo xe (mobihet-features 29/10).

## Dữ liệu

- `pyarrow.dataset(<v4>/train)` lỗi "Unable to merge: Field scenario has incompatible types: large_string vs string" (cột trong file là
  `large_string`, partition hive suy `string`) ⇒ đọc từng file bằng `pq.read_table` (`core/data.py`).
- 77 đầu vào = 11 × 7 ⇒ DAGSNet giữ lưới 11 vị trí (`view(B, 11, 7).transpose(1, 2)`).

## Compile

- **`torch.compile` + AMP fp16 làm gradient NaN ở nhánh AlexNet grouped** (29/10): forward khớp eager (Δlogit 2e-3) nhưng
  backward của `amax` (maxpool 2, ceil) cho NaN ở mọi bước ⇒ optimizer bỏ mọi bước, model đứng ở init, FedAvg smoke acc 0,047 /
  loss "0,028" (chỉ là tổng các bước được áp = 0). fp32 compile và eager không bị. Sửa: `torch.maximum(x[..., 0::2], x[..., 1::2])`.
  Chặn tái diễn: `local.compile_gate` (một bước thật, so gradient compile vs eager, FAIL ⇒ eager) + cảnh báo `skipped_frac` mỗi round.
  Test tương đương fp64 eager không bắt được lỗi chỉ có ở đường compile + fp16.
- Shape tĩnh cho train (nhóm đệm tới đủ G client rỗng), shape động cho eval; mỗi shape mới = một lần compile ~2 phút.

## Tốc độ (đo 29/10, RTX 3050 4 GiB local)

- DAGSNet đơn, AMP, batch 8.192: full 46k mẫu/s eager, 59k compile; W 0,375: 78k / 113k.
- Grouped (G model/bước, bmm channels-last) compile: ~45k slot/s full width ≈ 77 % model đơn; eager chậm hơn ~2,5×.
  Lần compile đầu ~2 phút. Hiệu suất thực = slot thật / slot (padding): 0,85 ở batch 64 / 8.192 dòng mỗi bước grouped trên phân bố
  client v4 (nhóm theo kích thước); phần mất chủ yếu do client lớn (95k dòng = 1.487 bước tuần tự).
- **Stride của tensor đầu vào cũng là guard của `torch.compile` tĩnh** (29/10): `mask = valid[:, s*b:(s+1)*b]` có stride phụ thuộc số bước
  của nhóm ⇒ mỗi nhóm một lần recompile, quá 8 lần dynamo **im lặng chạy eager** (probe Kaggle FedAvg W 1: r1 1.243 s thay vì vài phút).
  Sửa: `.contiguous()` mỗi bước, `recompile_limit = 64`; kiểm bằng `TORCH_LOGS=recompiles` (còn 2 lần hợp lệ: dropout 0 của gate, no_grad eval).
- Nạp parquet trên Kaggle: mỗi rank đọc cả 410 file từ `/kaggle/input` mất 10–30 phút ⇒ một lần `data.build_npy` trong notebook rồi
  `V4_NPY` memmap theo rank.

## Chi phí MobiHet v4

- 30/10: `none` là **544 B mỗi chiều** = 1.088 B/xe/round, không phải 544 B tổng. Phải cộng bootstrap input stats:
  S4 Gaussian 394.368 B/xe hai chiều; G4 copula 3.311.744 B. Tổng 10 round tương ứng 405.248 B và 3.322.624 B;
  50 round nếu payload giữ nguyên: 448.768 B và 3.366.144 B. Không kết luận rẻ hơn mọi arm chỉ từ byte steady-state.
  H1 tách 8 nhóm làm bootstrap dense lớn hơn nữa; utility chưa có, chưa coi H1 là cải tiến đã PASS.

## Synthetic/KD v4 — audit H và chẩn đoán 30/10 tối

- H0 paired copula−gauss +0,00546 seed43, cùng dấu +0,00529 seed42; hai seed chưa đủ kết luận ý nghĩa thống kê.
  H1–H4 đều FAIL gain+0,010 vs G4. H3 tăng32→64 dòng có pooled+0,00371/p10+0,03457; lợi mô tả, chưa PASS.
  Cluster8 tăng bootstrap lên23,73MB/xe mà chỉ +0,00076 F1. Mix/onehot giảm pooled.
  [Audit đủ10metric](archive/docs/research/screening-h10.md).
- Full train label scan37.361.273 dòng + CM G4 own-test: lớp absent local chỉ26.999/9.347.679 dòng test (0,289%).
  Không thể quy toàn bộ lỗi pooled cho thiếu lớp. Nhóm<256 mẫu/xe meanF1=0,587, nhóm≥16384=0,880; đây là tương quan.
- Q[y] là teacher nhãn, không phải hàm theo input. G4 r10 Q[timeDelay,timeDelay]=0,279, argmax=benign;
  timeDelay→benign460.575 lỗi test. Có nguy cơ KD củng cố nhầm, nhưng chưa có ablation để chứng minh nhân quả.
  J4 bỏ riêng KL real, J1 dùng teacher từng mẫu có đối chứng cùng support J0. Giữ dataset77 feature/protocol.
  [Bằng chứng và falsifier](archive/docs/research/screening-j10.md).

## J và thiết kế thông tin trao đổi — 31/10

- J0–J4 đều FAIL gain+0,010. J1 teacher theo mẫu hơn cùng-bankQ[y] +0,002123, nhưng bankQ[y] mất−0,004072
  vsstreamingG4. Teacherargmaxagreement~0,88–0,90 không đủ chứng minh utility; không tiếp tục selector chỉ vì probeđẹp.
- J4 bỏrealKD pooled+0,003629 nhưng timeDelayF1+0,214792, **15lớp còn lại đều giảm**; 2895clientkémhơn/2272tốthơn,
  nhóm<256mẫumeanΔ−0,008366. Cần xét regularization và classnegative-transfer, không coi bỏKDreal là thắngcho từngclient.
- ComboH3+J4 cộngthuần≈0,880524 vẫn dưới cổng0,883188; phải đo tươngtác, không cộngΔrồi tuyênbốPASS.
- BảngconditionalQ[y,b(input)] giữ schemaoutput16lớp chung, không gửi featuretừngmẫu. Router3sign fixedB8 chỉ làpartition,
  chưa có bằngchứngđúngdecisionboundary; train-onlypreflight125/128bucketcólớp, entropySybil0,251nats cho thấy lệch mạnh.
  [K spec](archive/docs/research/screening-k10.md), [thiết kế từ bài toán](archive/docs/research/federated-design.md).

## K kiểm chứng hai seed — 31/10 ~10:27

- Label-agreement realKDgateK8/K9 F1=.883935/.884584, gain+.010747/+.010076 vsG4cùngseed ⇒PASScổngr10cảhai,
  nhưngmargin43chỉ+.000076. Hai seed không đủgọi ổn định/significant; chưa50round.
- Gatehơnaugmentationchỉ+.000586/+.000250 F1, p10thấphơn. GatebenignF1~.861, augmentation~.832;
  augmentationtimeDelayF1~.315, gate~.14. Chọnphươngpháp phải báođánhđổilớp/xe,khôngchỉpooled.
- Augmentation-only(unitCEreal+CEsynthetic,khôngQ/raw) hơncomboKD cả2seed(F1/p10),khônglearnedmessage saubootstrap.
  Vẫn dùngglobalcopula/prior3311744B/xe,khônggọilocal-only/khôngFL. K4PASS42,K5FAIL43sátngưỡng;khôngnớicổng.
- Dose64×noRealinteractiondương+.001856/+.001629,nhưngcomboK0/K1vẫnFAILgain. ConditionalB8khônghơncombo
  (−.000572/−.000019)cả2seed; thêm8256B/r chưa cóutilitygain. KhôngquétB/routertrêntestđểtìmwin.
- 100roundCM/10metric/support và100snapshotfiniteđãaudit; chưaprediction→CM/fullprivateweightsbankrebuild.
  [Kresults](archive/docs/research/results-k10.md).

## Chẩn đoán cơ chế sau K — 31/10

- SnapshotQr1–r9 của K8/K9 chỉ reject lớp13; gate thực dùngr2–10 do đó tắt KLreal trên5,151% dòngtrain,
  không phải sample/inputconfidencegate. QtimeDelay sai vẫn vào syntheticKL và `ce_t`/adaptivecoefficient.
- 848xe<256mẫu chiếm15,871%client nhưng chỉ0,359%trainrows. Sampleweightedteacher có thể ưu tiên pooled khác tail;
  chưa có causalablation weighting. Mỗi epoch2–1489bước/client; dose64 không đồng nghĩa nhânđôi lossweight.
- Augmentation vscombo đổi cảtarget vàadaptivecoefficient; phải đối chứngfactorial trước khi quy gain cho bỏteacher.
- Nhánh classQ hiện gộp sum chưadecodefp16body; byteaccounting không chứng minh utility của wireprecision thực tế.
  Giữ nguyênsourceK; nếu triểnkhai encode/decode thật, cần control cùngcodec mới.
  [Bằng chứng, công thức và đềxuất](archive/docs/research/root-cause-experiments.md).

- Lcodecmeanfp16 encode/decodeclassQ cầncontrolcùngprecision; khôngso vớilegacy rồi quy toàn bộΔcho loss.
  Unitserialization/gradientcells/nonmutatingprobePASS; legacyCPU2round3casebitexact, real4xeW1compiledAMP2roundPASS.
  Shadowprobe postepochfp32/dropout0 vàF1train đãfit không chứng minh actualtraininggradient hay independentgeneralization.
  [Lspec](archive/docs/research/screening-l10.md); utilitysauđóxem [Lresults](archive/docs/research/results-l10.md).

## L đã audit — 31/10 ~17:08

- Fixed1 hơnadaptive ở hard/noReal,soft/noReal,soft/realgate cảhai seed; bỏadaptive trênsoftgate tăng+.003014/+.002713F1.
  Softtarget hơnhard khi cùngfixed/noReal +.002967/+.002527. Kaugmentationthắngcombo khôngchứngminhlearnedQvôích:
  target vàlossweight phải tách. [Lresults](archive/docs/research/results-l10.md).
- L6/L7softfixedgateF1=.887030/.887433,PASSgainr10cảhai; L0/L1softfixednoReal=.886315/.886861cũngPASS.
  Realgatechỉthêm+.000715/+.000572 trênsoftfixed, chưacơsởgọicầnthiết/ổnđịnh. L4/L5p10cao hơn, pooledthấphơn.
- Probe34clientr10: adaptiveusedweightmedian2,38–3,88, weightedsyntheticgradientmedian2,35–4,89×realCE;
  phùhợphypothesisauxiliarylossquámạnh, khôngphải actualAMPgradient/clippingrate toànmạng. Fixedsoft tăngcảtrain/testF1;
  khôngquytoànbộlỗi chooverfit. Traingap chỉmôtả vìtrain+valđãfit.
- L6/L7meanΔxe<256 vsaugmentation vẫn−.007724/−.007348, dùpooledtốt. Khôngđồngnhất pooledgain với từngclientđược lợi.
- TimeDelayprior5,151%train+val→5,162%test gầnnhưkhôngđổi; train/testF1L0=.694/.283 vẫnchênhlớn.
  Classprioralone chưa giảithíchđượclớpkhó. SybilfullFLtrain+val8,80%, số8,0%cũđãđượcsửa.
- R1cùngwarmupmath nhưngF1giữacácLsessioncùngnominalseedspread.000516/.000588; nguyênnhânchưaisolate,
  khôngphảiCI/noisebound. KhôngsuyCPUbitexact thành2T4bitexact; lợismallgate/codec cầnlặpcấu hình.
- 12run×10round,testCM10metric/48trainCM/1632proberecord/120snapshot PASS; fullprivatebank/predictionreplay/50round
  vẫnchưakiểm. Khôngđổisource đãchạy hoặc nớicổng choL2/L3/L8/L9FAIL.

- 31/10 cohort audit: absent-local chỉ0,289%test toànmạng nhưng nhóm848xe<256 có1555/34827testrows=4,465%,
  median8/16lớp trongtrain,9776testrows thuộc localclasscount1–16. Không dùng pooledtỷ lệ để phủ nhận thiếu lớp
  ở xe nhỏ; đây là mô tả support, chưa chứng minh cơ chế cá nhân hóa có ích. [M chẩn đoán](docs/research/screening-m20.md).
