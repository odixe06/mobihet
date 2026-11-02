# Bài học kỹ thuật còn áp dụng

Đọc sau CONTEXT. Bằng chứng hiện tại: [M20](archive/docs/research/results-m20.md), [chẩn đoán tiếp](docs/research/diagnosis.md).
Archive giữ hồ sơ cũ; không nạp lại trong phiên thường. Không bỏ bài học dưới đây vì đã archive run.

## Hợp đồng dữ liệu và đánh giá

- 77 input=11×7, đã scaled; không chuẩn hóa lại. Đọc từng file bằng `pq.ParquetFile` để tránh schema hive
  `large_string`/`string`. Thứ tự feature và label mapping phải đóng băng.
- Train FL gồm train+val,37.361.273dòng; own-test9.347.679dòng. F1train chỉ mô tả fit, không phải independentvalidation.
- MacroF1/xe chỉ tính lớp có trong y_true; pooled đủ16lớp. Mean/p10/std giữa xe cần đọc cùng support từng xe/lớp.
- Sybil8,80%train+val→17,67%test; timeDelay5,151→5,162% gần giữ nguyên. Không quy lỗi timeDelay cho classprior alone.
- Absent-local chỉ0,289%test toàn mạng, nhưng848xe<256mẫu có median8/16lớp,1555/34827testrows=4,465% absent,
  9776testrows thuộc lớp có1–16mẫu local. Pooled không phản ánh đủ khó khăn của xe nhỏ.
- Giữ headline endpoint đã khóa; không chọn best test round hoặc HP theo test. Nhiều batch đã đọc officialtest,
  không gọi đây là freshconfirmatorytest. Hai seed chưa đủ significance.

## Train/compile và artifact

- `torch.compile`+AMPfp16 từng tạo NaN gradient ở AlexNet maxpool dùng `amax` dù forward khớp; dùng `torch.maximum`
  theo cặp. Compilegate phải kiểm backward trên realdata; theo dõi applied/skippedsteps, không chỉ finiteweights/loss.
- Slice batch phải `.contiguous()`: stride theo group gây recompile và fallback eager im lặng. Static train/dynamic eval;
  shape/stride mới có thể tốn compile. Local sm86 không chứng minh T4sm75.
- Hai rank giữ privatebank khác nhau: fullresume phải lưu từngrank, cùng round; budgetstop cần đồng thuận tránh deadlock.
  Snapshot nhỏ loại privateweights, không phải checkpoint đầy đủ. Exactresume RNG chưa được chứng minh.
- CEreal dùng LA, KD dùng rawlogit có chủ đích; khác hệ xác suất chưa tự chứng minh bug.
- V20 V0/V1 dừng r1 tại guard`skipped_frac==0` sau2T4compilePASS. P0/P1 đãaudit có skip rất nhỏ mọiround,
  còn P2/P3augmentation có skip ởroundmuộn: localfixture skip0 không chứng minh fullnetwork skip0.
  `core/local.py` dùng fixedloss_scale4096, không dynamicGradScaler; chưa được gọi skip là warm-up.
  OriginalVfailed mất skipfraction vì guard ném lỗi trướclog. ProbeT4full5343 mới đo3/513368 và5/513368,
  mọi xe có appliedstep, weights/BN/loss finite; đây không giá trịkhôi phục của hai failuregốc.
  Một xe193trainrow vẫn mất1/4step: globalfraction nhỏ không bảo đảm mọi xe ítảnhhưởng.
  RecoveryCE chỉ sửa guard về common≤1% +positiveapplied từngxe +finite, lưucounters mỗiround;
  không đổi optimizer/AMP/fixedscale hoặc gọi rootcauseoverflow đãsửa. Giữ oldfail và dùng slug/source riêng.
- Shadowprobe34xe là postepochfp32/trainBN/dropout0 trên clone, không actualAMPgradient hoặc clippingfraction toàn mạng.
- Audit phải dùng artifact thật: sourceoverlay/config/LR/IDs/support, CM→10metric, finite tensors/teacherlag.
  Dataset hiện chưa có fullcontentfingerprint và chưa prediction→CM replay; phải nói rõ phạm vi.
- Q20: chuẩn hóa cùng fp16body trên CPU/CUDA có thể lệch2.384e-7, vượt atol2e-7 dù target lưu đúng.
  Probe100snapshot T4 khớp decodeCUDA local; auditor dùng khoảng theo từng entry từ gamma16 của sumfp32 và u của division,
  giữ exactbody/aggregate/source/signature checks, không nới atol chung. Test storedboundary và targetcorruption phải cùng qua.
  Đây là roundingconsistency, không proof exactCUDAreplay mọi runtime hoặc lý do trainlại.

## Những hướng đã thử và điều cần giữ

- Copula hơn Gaussian trên nền cũ cảhai seed khoảng+.0053F1; không suy rằng generator đã tối ưu khi loss thay đổi.
  Cluster8/mix và các adaptation GMM/MADE/BNsplit từ v1 không có utility đã chứng minh; không mở lại thiếu giả thuyết mới.
- J: anchor/input-aware teacher chỉ hơn cùng-bankcontrol nhưng không hơn streamingcontrol; teacher agreement đẹp không đủ.
  Bỏ realKD giúp timeDelay nhưng làm15lớp khác giảm. Không cộng gain từ các ablation để dự đoán combo thắng.
- K: bucket8 không hơn combo, tăng byte vô ích trong cấu hình đã thử. Augmentation mạnh, vẫn dùng bootstrap toàn mạng;
  không gọi augmentation là local-only hoặc zeroFL.
- L factorial: fixedlossweight1 hơn adaptive ở hard/soft/realgate cảhai seed; softtarget vẫn hơn hard khi giữ scale.
  Vì vậy augmentation thắng combo cũ không chứng minh learnedQ vô ích. Adaptive có shadowauxgradient quá lớn,
  nhưng cosine/norm riêng không chứng minh nguyên nhân duy nhất.
- Lrealgate chỉ thêm~.0006–.0007pooled trên softfixed/noReal; chênh sessionr1 cùng công thức~.0005–.0006,
  chưa isolate nguyên nhân, không CI/noisebound. Không suy CPUbitexact thành T4bitexact.
- Q[y] cùng cho mọi mẫu cùng lớp, không biết input. ArgmaxQ đúng vẫn có gradient ngược CE ở nhiều mẫu;
  correctargmax không phải chứng minh teacher hữu ích. Gate chỉ xét lớp, không sampleconfidence.
- Qr dùng từ r+1; realgate cũ chỉ chặn lớp13 trong r2–10, Q sai vẫn vào synthetic/adaptiveweight.
  Đổi target, scale, generator, codec cùng lúc sẽ mất khả năng quy nguyên nhân.
- Legacy K tạo bodyfp16 để đếm byte nhưng gộp float chưa decode; không gọi đó là wireprecisionproof.
  L/Mmeanfp16 đã encode/decode và cần control cùng codec.
- Chi phí Q=544B/chiều=1088B/xe/r; bootstrapGaussian394368B, copula3311744B hai chiều.
  Không kết luận giao tiếp rẻ chỉ từ packet mỗi round; bootstrap/transport chưa end-to-endproof.
- Không coi pooledF1 tăng là mọi xe/lớp được lợi, hoặc gap giảm vì xe mạnh suy giảm là fairness tốt.

## M20 đã audit đầy đủ

- Control vẫn tăng r15→20; không bão hòa đồng loạt. Fallback dao động/plateau khi Q đổi argmax.
  Q thu ở r13 tác động r14, không r13. Hai kênh realgate và syntheticfallback cùng đổi: tương quan chưa chứng minh nhân quả.
- Argmax đúng nhưng margin nhỏ vẫn mở KD; cần tách hai kênh trước khi thử gate liên tục/hysteresis.
- Ranking/cohort đổi r10→20; không suy kết luận endpoint10 sang20 hoặc chọn best round.
- Server sqrt(n) tăng pooled cảhai seed, xe<256 chưa hưởng nhất quán. Localρ theo count chưa hơn uniform ổn định.
- Own-test seen-class có thể tăng trong khi absent-class xe nhỏ giảm; nâng pooled/tail không tự giải quyết thiếu lớp.
- Artifact M20 đã kiểm320snapshot/320testCM/96trainCM/3264shadowprobe; giới hạn bằng chứng vẫn như mục audit ở trên.

- M20: gain sqrt(n) pooled tập trung84–87% vào timeDelay; realKD giảm timeDelay nhưng nâng15lớp còn lại. Không gọi gain tổng là cải thiện đồng đều.
- Gap train/test timeDelay vẫn rất lớn sau cân theo support test cùng xe/lớp, cả xe nhiều mẫu. In-sampletrain không tách overfit/temporalshift.
- Uniform hóa offdiagonal giữ Qyy nhưng đổi entropy và có thể đổi argmaxgate; hoán vị offdiagonal giữ entropy nhưng phá quan hệ lớp.
  Muốn kiểm nội dung Q phải tách các confound này, giữ rawgate/lịch teacher phù hợp.
- Negative logitgradient không đồng nghĩa parametergradient tổng âm hoặc KD có hại. KL/hardCE khác entropyconstant; trainloss thấp không đủ chứng minh học tốt hơn.
- 1epoch không là cùng số bước: xe nhỏ2–4step/r. Tăng KD dựa count chưa thay được phân tích lượng tối ưu và generalization.

## N20 factorial đã audit

- [N20](archive/docs/research/results-n20.md): realKD giảm timeDelay ở cả soft/fallback nhưng nâng other15/mean/p10.
  Không tắt/tăng KD đồng loạt từ pooled alone; fallback gain nhỏ chưa nâng đồng đều client/cohort.
- Fallback khi realKDoff vẫn đổi target đột ngột và dao động sớm. Bác bỏ “chỉrealKD gâydaođộng”;
  bỏreal không tự xóa chatter, nhưng C vẫn học r15→20. Không gọi tất cả client/model plateau sau20round.
- Realgate13đóng vẫn giảm timeDelay ngayr2 khi bật realKD các lớp khác. Cần tách crossclasseffect/feedback;
  sửa confidence/gate13 riêng chưa chứng minh đủ. Teacherargmax đúng hoặc gate lớp yếu bị chặn không bảo vệ mọi decisionboundary.
- N matchedsameclient/class train/testgap vẫn lớn ở xe nhiều mẫu; không quy toàn bộ cho thiếu lớp hoặc ít step.
- Factorial liveFL đo total effect vì Qtrajectory khác; replay cùngteacher cần thiết để tách nội dung/tiếp nhận khỏi feedback.
  Hai seed/sessionr1spread không là significance/noisebound; không chọn C vì pooled hơn vài phầnvạn.
- Replay: reencode Q đãnormalized có thể lệch~1e−4; giữ Q nguồn, phục hồi packet tương đương và kiểm decode GPU
  trước train. CPU/CUDA reduction có thể lệch1ULP; không sửa teacher để ép CPUbitexact. Permutation phải khóa SHA theo runtime.
- Replay preload toàn lịch teacher là chi phí setup riêng. Không vừa tính cachehistory vừa tính lại từng downlink,
  hoặc coi packet-equivalent là đo liveFL/contactcost.

## O20 replay đã audit đầy đủ

- [O20](archive/docs/research/results-o20.md): fullQ/realgate hơn noReal pooled cảhai seed khi cùng teacher nguồn,
  nhưng vẫn giảm timeDelay và tail/xe nhỏ chưa lợi nhất quán. Không biến kết luận Nlive thành “realKD luôn hại pooled”.
- ReplayrealKD vẫn hại13 từr2 dù gate13đóng và Q chung; không cần feedback để có hại cục bộ. Đổi lịchteacher
  từ livegate sang nguồn noReal phục hồi một phần timeDelay; không phân rã causal hoặc gán mọi lỗi cho feedback/gate13.
- FullQ hơn diagonal-preservinguniform và entropy-preservingpermutation pooled cảhai seed. Có utility của offdiagonalidentity
  trong cấu hình này; không bỏ offdiagonal trước proofmới. Uniform lại lợi meanxe<256/timeDelay: pooledutility không đồng đều.
  UniformO là chia đều offdiagonal/giữdiagonal; khác uniformM12 trộn hard/softρ=.5, không cộng gain hoặc coi cùng can thiệp.
- R0replaycontrol đạt tripwire so N0/N1 ở cảhai seed, không bitexact hoặc significance. R1 vẫn học r15→20;
  gap train/test vẫn lớn cả xe nhiều mẫu. Chưa chọn phương pháp tổng, liveFL/mixedarchitecture/r>20 chưa kiểm.

## Đối chiếu nguồn trước P20

- Owner xác nhận v1 chỉ tham khảo66feature/20–100client, không chuyển gain/ngưỡng/causalclaim sang v4/5343xe.
  V1 teacher tốt hơn nhưng transfer/online không giữ gain; BNsplit/head-only đã FAIL. Không coi teacher là bottleneck duy nhất.
- V4 scaler định nghĩa6NaNflags0/1 và mapping featurethiếu→0. Audit bounded419840trainrow không vi phạm;
  fixture copula hiện sinh100%cờ khác0/1, projection đưa về0%. Đây là fidelityconstraint, **chưa utilitygain**.
- Teacher classQ phụ thuộc trueclass, không phải classifier inputdependent. Không chấm Q[y_true] rồi so accuracy với neuralproxy
  như cùng nhiệm vụ. P dùng evaluator-only proxy/studentCM trên cùng diagnosticrows, không dùng để chọn luật/HP.
- DAGSNet/grouped/protocol hiện khớp byte repo benchmark. LwFedNIDS đãprune39494parameter và V2X-HtFL461456parameter
  có headriêng trong kết quả cũ; tách khỏi bảng tham chiếu đồng nhất395408parameter. [Đối chiếu](docs/research/benchmark-r20-reference.md).

## P20 — teacher, support và clientobjective

- [P20 gốc](docs/research/results-p20.md) đã đóng gate32run:31×20round/P21r0. [Recovery P21](archive/docs/research/results-p21-recovery.md)
  bổ sung seed43 đủ20round, không thay originalfailure bằng probe1round hoặc gọi freshretry là resume.
  P21 lỗi số học trước serverfit không là utilityFAIL của hybrid; guardPASS/recovery không chứng minh rootcause đã sửa.
- Augmentation là phần gain lớn nhất so localCE; local-only có accuracy0 trên absentrows xe nhỏ, synthetic giúp đáng kể.
  SoftclassQ vẫn thêm utility pooled cảhai seed; không gọi learnedteacher vôích hoặc augmentation zeroFL.
- Cùngbank, inputtarget hơn classmean cảpooled/low256/p10; nhưng bankhard vẫn thua streaminghard pooled,
  inputteacher vẫn dưới streamingclassQ. Teacherresolution và sampling/support là hai vấn đề cần tách; livecontrast không cô lập feedback.
- Serverproxy C8 chưa hơn inputensemble C7; trên34xe diagnostic, neuralteacherF1 thấp hơn private student ở cùng realrows.
  FitKL/syntheticagreement không chứng minh realteacherquality. Paired teacher đúng/student sai là cơ hội, không count lỗiKD đã sửa.
- Modelcarrier vs outputcarrier cùngbank qua tripwire cảhai seed, không bitexact. Model tốn nhiều byte hơn mà không utility ổn định;
  streamingquery qua proxy còn giảmpooled cảhai seed. Không mặc định tăng kích thước teacher hoặc downweights sẽ tốt hơn.
- BinaryrealKD tăng meanxe nhỏ nhưng giảm absentclassaccuracy và timeDelay cảhai seed. Margin hơn binary/strength pooled,
  nhưng low256/p10 không lợi cảhai seed và absent vẫn dưới noReal. Confidence lớp hoặc ítdata chưa đủ để ép mọi xe theo teacher.
- C2 và xe nhỏ vẫn tăngr15→20; C9pooled giảm trong khi meanxe nhỏ tăng. Không gọi một curve là plateau toàn hệ thống.
  Gap seenclass train/test tồn tại cả xe>=4096; chưa tách overfit/temporalshift bằng in-sampletrain.
- Projection đúngNaNflag không tăng pooled ổn định. Không coi syntacticconstraint là jointmanifold hoặc nguyên nhân utility chính.
  Mixture/countρ, postepochQ, BNsplit/head-only và v1 khácdataset đã có bài học; không đổi tên để thử lại thiếu hypothesis mới.
- Bootstrap vẫn chi phối compactQ20round. Báo đủ setup/two-waybytes; timer server/scoring nằm trong train,
  không cộng lại hoặc gọi walltime2T4 là latency xe. Hai seed/repeatedofficialtest không là significance/confirmatoryproof.
- [Pairedclientdiagnostic](archive/docs/research/p20-client-transfer.md): thêm softQ hơn augmentation pooled nhưng2207/2240xe giảmF1,
  chỉ~38%xe<256 tăng. Margin so C2 vẫn có1059xe giảm cảhai seed. Mean/p10/cohortgain chưa là mọi xe tốt lên;
  số xe giảm phải đọc cùng độ lớn delta/support, không chọn method từngxe bằng officialtest.
- C10hybrid (.5ensemble+.5onehot khi fitproxy) hơn C8pooled+.010208/+.008540 hai seed; vẫn dưới C2streamclassQ
  −.002447/−.004481. Xe<256 chưa hơn proxy/ensemble nhất quán; so C2 có1412xe giảm cảhai seed.
  Cải thiện objective server có utility trong setting đã thử, không đủ chọn proxy làm teacher mặc định hoặc tăng KD cho mọi xe.
  Teacher C10 trên34xe diagnostic vẫn dưới student, teacherđúng/studentsai không phải count lỗi KD đã sửa.

## Q20 — ba khâu và độ tin cậy từng xe

- [16run Q20](docs/research/results-q20.md) đủ20round/hai seed, querycontrol−stream qua tripwire, nhưng fullremote CM/Q
  không bitexact dù localfixture bitexact. Querycontrol cũng có1227xe delta âm cảhai seed; không biến negativecount
  thành causalKDfailure hoặc dùng hai seed làm noisebound/significance. Báo thêm độ lớn/support và repeatedcontrol.
- Mixedsupport dùng classQ riêng chưa nâng pooled/p10 ổn định. Cùng support, inputtarget nâng meanxe<256/p10 cảhai seed;
  so querycontrol, pooled gần giữ nguyên, meanxe nhỏ tăng. Có utility resolution/weakcohort, chưa chứng minh utility/byte thắng compactQ.
- Sqrtweight chỉ cho inputensemble giảm pooled cảhai seed, xe nhỏ trái dấu; gain sqrtclassQ ở M không chuyển sang mọi teacher.
  Count/confidence không là expertquality; cần validation sạch trước qualityweightserver.
- Margin trong mixedinput nâng pooled/mean/p10 so stream, nhưng absentaccuracy xe nhỏ và timeDelay giảm.
  RealKDtradeoff tồn tại sau inputresolution; teacher tốt theo classargmax/agreement chưa đủ bảo vệ clientobjective.
  Live contrasts gồm feedback; receptionreplay cùngQ/inputhistory mới tách được feedback khỏi cách học.
- Margin có1005xe delta âm cảhai seed so stream, nhưng querycontrol cũng có biến động percar lớn.
  Mô tả giảm>.05 cảhai seed là75xe margin/29xe control, không causalcount hoặc CI.
  Mean/p10 tăng và std giảm không bảo đảm mọi xe/absentclass tốt hơn.
- Xe nhỏ vẫn tăngr15→20; gap matchedtimeDelay còn cả xe>=4096 và control. Chưa plateau chung hoặc phân biệt overfit/shift
  khi train+val đãfit. Q không proxyteacher, không có realclassifierteacherqualitydiagnostic để gọi agreement là accuracy.

## R20/V20 — độ ồn, replay tiếp nhận, validation sạch

- [R20/V20](docs/research/results-r20-v20.md): ba phiên cùng công thức/seed lệch pooled ≤.0005, p10 ≤.006, low256 ≤.003,
  timeDelay ≤.0035; ~250 xe dao động >.05 mỗi chiều. Đếm "số xe giảm" và chênh dưới các ngưỡng này không là bằng chứng.
- Teacher cố định, realKD vẫn −.05 timeDelay/lớp vắng: tác hại không cần feedback. Live gấp đôi do vòng phản hồi
  (Q[13,13] r20 .739 noReal → .581 binary). **Margin ≤ binary/strength khi teacher cố định**; lợi live của margin là
  bảo vệ teacher, không phải tiếp nhận tốt hơn. Sửa kết luận Q20 "margin hơn binary" thành hiệu ứng trajectory.
- Val sạch: augment−CE +.0226 pooled (nguồn lợi chính, không do test); mixed_input−augment +.0025 pooled/+.009–.014 p10;
  margin−mixed pooled ≈ ồn nhưng timeDelay −.077, lớp vắng −.06/−.07. Loại realKD khỏi mặc định.
- Gap timeDelay train→val có cả ở CE thuần (xe≥4096 .886→.267): giới hạn dữ liệu/đặc trưng, KD chỉ làm nặng thêm.
- Cơ chế hại là phần target-class của KL theo lớp; phần non-target (quan hệ lớp) mới là utility O20. Tách hai phần trước
  khi bỏ KD (W20 NCKD). NCKD có gradient 0 lên logit lớp thật — kiểm bằng test, không chỉ công thức.
- Bootstrap copula: 76% byte là histogram dày, 12% là moment Gaussian không dùng trên đường copula. Histogram xe mã thưa
  không mất mát ~74 KB thay 1,26 MB; tổng server không đổi nên không cần chạy lại để chứng minh.
- Shadow diagnostics tự dựng ctx ngoài local_epoch: hook mới trên ctx phải có fallback, nếu không preflight chết ở diagnostics.

## W20 — non-target KD, step floor, bootstrap nén

- Ô "PASS gate" có thể chỉ dời ngưỡng giữa cặp lớp nhầm (synth hard+NCKD: timeDelay recall +.06, precision −.09,
  benign→timeDelay ×1,85, Q[13,13] giảm). Với cặp nhầm khó, đọc precision/recall và nhầm chéo, không chỉ F1 lớp đó.
- Bỏ phần target-class của KD cắt feedback (Q[13,13] live = noReal) nhưng KD trên dòng thật vẫn đè cả nhóm lớp vắng
  (lỗi dồn sang lớp xe có). Hại timeDelay và hại lớp vắng là hai kênh khác nhau.
- Đếm xe vượt độ ồn lặp chỉ hợp lệ khi ô giữ nguyên dòng RNG. Step floor đổi độ rộng permutation → xe không chạm floor
  dao động bằng mức đổi seed dù trung bình ≈0; so với spread đổi seed hoặc trung bình theo cohort.
- Thiếu tối ưu ở xe nhỏ là nguyên nhân lớn của p10/lớp vắng: bước chỉ synthetic (0 byte, +3,5% bước) nâng xe<256 +.02,
  lớp vắng +.13 mà dữ liệu local vẫn đúng 1 epoch.
- Nén bootstrap (CDF uint16, fp16 Cholesky, uplink thưa) giữ utility; nhưng uplink thưa không dùng được với secure aggregation.
