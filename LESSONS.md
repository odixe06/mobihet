# Bài học kỹ thuật còn áp dụng

Đọc sau CONTEXT. Bằng chứng hiện tại: [M20](docs/research/results-m20.md), [chẩn đoán tiếp](docs/research/diagnosis.md).
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
- Shadowprobe34xe là postepochfp32/trainBN/dropout0 trên clone, không actualAMPgradient hoặc clippingfraction toàn mạng.
- Audit phải dùng artifact thật: sourceoverlay/config/LR/IDs/support, CM→10metric, finite tensors/teacherlag.
  Dataset hiện chưa có fullcontentfingerprint và chưa prediction→CM replay; phải nói rõ phạm vi.

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
