# P20 — sự cố số học trước round1, 01/11/2026

P21 `accanniez/mh-p21-proxy-hybrid-s43/1` lỗi trước commit round1. Chưa có kết quả performance;
không dùng sự cố này để kết luận proxy teacher hoặc hybrid distillation không hiệu quả.

## Bằng chứng đã kiểm

- Pipeline dừng16:40:30+07, exit1 sau khi pull P21; snapshot16:40 còn31kernel RUNNING.
  RUNNING là trạng thái API, chưa chứng minh các kernel đó khỏe hoặc đã qua round1.
- Output đã pull: config, log và sourceprovenance;8overlaySHA khớp manifest P20, đúng seed43/full5343/W1.
  Receipt `runs/p-study/failure-p21/audit.json`; inventory remote riêng ở cùng thư mục.
- Log: rank0 compilegate `logit_maxdiff=nan`, `grad_rel=nan`, `bn_maxdiff=nan`, chuyển eager;
  rank1 compilegate PASS. Config rank0 ghi `backend=eager`, không đạt cổng compiled của batch.
- Crash trong `round_bank → _probs → roundtrip(prob,1,r) → encode`;
  `np.isfinite(a).all()` không đạt. Codec đã chặn output không hữu hạn trước tổng hợp và fit proxy round1.
- Fullinventory exactversion có47file, đều hiện diện local; không có CM/round/snapshot hoặc resume checkpoint.
  Receipt `runs/p-study/failure-p21/remote-inventory.json`. Không có private state tại điểm lỗi để replay chính xác hoặc xác định xe lỗi.

## Giả thuyết và kiểm chứng

Phân biệt input không hữu hạn, overflow khi inference AMP, state/BN không hữu hạn và khác biệt eager/compiled.
Compilegate có NaN không tự chứng minh nguyên nhân của lỗi upload sau localepoch; có thể là hai biểu hiện khác nhau.
Không tự sửa codec bằng nan-to-zero, bỏ client, hạ gate hoặc đổi loss để vượt kiểm tra.

Probe `scripts/diagnose_p20_nonfinite.py` chỉ dùng train:64row/xe từ64xe lớn mỗi rank,
seed42/43, DAGSNetW1/dropout0, so eager FP16/FP32, ghi lớp BN đầu phát sinh nonfinite.
Fixture có fingerprint; tie selection/row order bounded. Không backward, không train/test adaptation,
không replay state P21 hoặc thay proof2T4. Artifact đã audit:2fixtureSHA đúng,8forward hữu hạn/BN hữu hạn,
RTX3050sm86/torch2.13. Không tái hiện lỗi; chưa phân biệt được runtime/compiled/backward/postepochBN.
Receipt `runs/p-study/numerics-probe/verification.json`.

Probe tiếp `accanniez/mh-p21-numerics-t4-20261101/1`:source P frozen + diagnosticwrapper,
full5343/2rank/seed43/cùng round1, giữ fallback backend như driver gốc. Wrapper ghi compilegate từngrank,
finiteprob từnggroup; nếu gặp NaN, kiểm parameters/BN/synthetic và inferenceFP32 tại cùng state/input,
trace BNFP16. Không own-test eval, không export rawrows/privateweights, không W&Bkey.
Đây là reproductiondiagnostic, không thay P21benchmark hoặc tạo endpoint mới. Joblaunch/watch/pull ở bàn giao.

Sau reboot đã collect/audit probe T4: cả2rank compilegatePASS,84group bao phủ đúng5343xe mỗi xe một lần,
mọi probabilitygroup hữu hạn; round1 hoàn tất, skipped0, teacherSHA giống nhau giữa rank.
Sourceprovenance đã đối chiếu từngfile; receipt `runs/p-study/numerics-t4/verified-diagnostic.json`.
**Chưa tái hiện lỗi P21 cả trên T4; chưa có bản sửa số học được chứng minh.** Probe không chấm F1,
không phải20roundbenchmark. Cần đối chiếu các kernel P20 khác và điều kiện runtime/backend của run lỗi;
không coi một lượt reproductionPASS là bằng chứng lỗi tự hết hoặc teacher hiệu quả.

Worker phục hồi chỉ collect/audit exactversion đã push; không launch hoặc sửa source scientific đang chạy.
Sau probe cần tách lỗi triển khai khỏi utility của teacher, kiểm active-client forward/BN/probability finite,
đúng seed/rank và productionshape trước thiết kế recovery. Nếu không có checkpoint, không gọi rerun là resume.

## Quyết định chạy lại sau audit toàn bộ

Chỉ P21 thiếu kết quả huấn luyện;31run khác đã PASS_BOUNDED20round. Cần hoàn tất C10seed43 để kiểm lợi ích
hybrid đã thấy ở seed42, không cần chạy lại toàn batch. Sáu checkerQFAIL đã xử lý bằng auditor, không phải trainingfailure.
Collector/watchjob thất bại hoặc mất do reboot cũng không là lý do restart những kernel đã audit.

P21 không checkpoint: lần phục hồi phải bắt đầu r1, giữ20round/cosine50/seed43/full5343/W1 và toàn bộ HP cũ.
Trước launch mới cần xác minh guard dừng cảhai rank khi compilegateFAIL, capture metadata trạng thái/BN/probability
không hữu hạn, và kiểm đường finite giữ nguyên phép học. Guard không thay NaN bằng0, bỏ client hoặc đổi precision/loss.
Giữ source/log lỗi cũ; dùng version/slug mới và kiểm quota/access trước push.
Tại lúc đóng phân tích **chưa launch retry, chưa có bản sửa số học được chứng minh**. Probe T4 là bằng chứng mới cho bước recovery,
không thay20round hoặc bảo đảm lỗi không lặp lại. Phân tích31run vẫn tiếp tục, C10 ghi rõ chỉ một seed.

## Recovery đã chuẩn bị theo yêu cầu chạy bù

`scripts/run_p21_recovery.py` bọc đúng Method cũ: guard dừng cảhai rank trước SGD khi compilegateFAIL,
kiểm privateprobability mỗi group trước codec và chỉ khi lỗi mới ghi metadata + FP32counterfactual.
Không thay đổi giá trị finite, target, loss, HP, seed, tham gia, codec hoặc backend của run đã qua gate.
Guard không phải lời giải nguyên nhân NaN gốc; nó chặn đường fallback chưa qua chuẩn batch và tăng khả năng chẩn đoán lần tới.

`scripts/preflight_p21_recovery.py` đã PASS realv4/seed43/3round,173train/46testrow, W1/compiledAMP,
teacher1024/proxy200step: state tensors/BN/CM/method+envRNGbitexact, rejected eager/peerfailure và NaNpacket.
Source trước/sau khớp; `runs/p-study/recovery-p21/preflight-verification.json` đã kiểm actualartifact.
Đây là local sm86; T4probe cũ fullround1 là bằng chứng bổ sung, không proof mọi20round hoặc xác định rootcause.

[Manifest recovery](p21-recovery-launch.json) giữ original8overlay/SHA/HP, thêm guardentrypoint, slug mới
`accanniez/mh-p21-hybrid-s43-recovery1/1`. Worker launch/watch1200s/pull/audit ở bàn giao;
chỉ pushreceipt mới xác nhận remote, chưa nhận seed43F1 trước fullaudit20round.

## Recovery đã đóng, 02/11/2026

`accanniez/mh-p21-hybrid-s43-recovery1/1` đã collect/audit20round, cảhai T4 compilegatePASS,
20finiteguardrecord/rank và skippedfraction0. Đã tái kiểm artifact/source/CM/teacherchain, dựng lại20proxy snapshot strict.
[Kết quả bổ sung](results-p21-recovery.md) hoàn tất C10seed43; originalP21FAIL và báo cáo31run giữ nguyên.
Guard/recoveryPASS chưa tái hiện hoặc xác định nguyên nhân NaN gốc, không gọi là numericalrootcausefix.
Job đã đóng, không gọi lại launcher/collector từ callback muộn; receipt và bước tiếp ở bàn giao.
