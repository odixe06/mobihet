# R20 — repeated controls, frozen reception, clean validation

Owner02/11 yêu cầu tiếp tục các kiểm nghiệm sau [Q20](results-q20.md).
Giữ DAGSNetW1/full5343/20round/cosine50/seed42–43/batch64/1epoch/AdamWreset/AMPfp16/clip1.
Không sửa source/manifest P/Q hoặc chọn HP/best round bằng officialtest.

## Thiết kế trước chạy

| nhánh | cell | run | câu hỏi / dấu hiệu bác bỏ |
|---|---|---:|---|
| repeat | stream repeat A/B, mỗi seed42/43 |4| So hai session mới cùng seed/công thức với Q0/Q1; mô tả spread pooled/percar và support. Không gọi spread là CI hoặc trừ noise để suy causalcount. |
| replay | mixedinput noReal/binary/margin/strength ×2seed |8| Cùng Q và inputtarget history từ Q6/Q7; tách reception khỏi feedback. Margin không hơn binary/strength hoặc absent/timeDelay vẫn hại → không coi margin bảo vệ mọi xe. |
| validation riêng | realCE+LA/augmentation/mixedinput/margin ×2seed |8| Fresh foldtrain→foldval, mọi control cùngsplit; bootstrap/prior/generator chỉ foldtrain. Gap còn trên val chưa fit → vấn đề generalization chưa chỉ do teacherfeedback. |

Triển khai theo gate: phase đầu chỉ8replay; repeat và validation chuẩn bị độc lập, chỉ push sau gate/quota riêng.
Owner xác nhận chỉ dùng16account hiện tại. Livequota09:06/02Nov chỉ3account≥6.9h, không đủ12run cap6h.
Ưu tiên đủ4reception×2seed; R0–R3repeat còn pending, không gọi đã launch hoặc có repeatedcontrol mới.
Manifest từng phase khóa trước push. Nếu quota không đủ toàn phase: giữ gate, báo cụ thể, không hạ round hoặc reserve.
R0/R1 = repeatA seed42/43; R2/R3 = repeatB seed42/43; R4–R11 = replay theo thứ tự trên.
Nhánh validation không thay headline train+val của protocol và không trộn kết quả vào benchmark.
Chưa triển khai qualityweight/memory/gradientprojection; chờ validation sạch trước expert reliability.

## Replay: chính xác cái gì cố định

Chỉ lấy aggregate teacher train-only Q6/Q7 đã audit: Q1…Q20, bankbodyfp16, decoded inputtargets,
bankX SHA và source snapshotSHA. Không dùng metric/test để chọn history, không privateweights/BN/realrows.
Q6/Q7 được chọn vì noReal mixedinput control đã định trước, không vì endpoint tốt nhất.
Mọi cell tiếp tục localtrain/upload/liveaggregation và scoring như Q; output live chỉ diagnostic,
downlink thay bằng cùng lịch nguồn. Qr/inputtargetr dùng từ r+1, cố định trong round; r1 hard synthetic chung.
Bankr tái sinh theo seed/version, phải khớp exact sourcebankSHA trên runtime; giữ authoritative decoded targets,
không reencode normalizedQ hoặc dùng CPUatol chung để sửa target. Nếu CUDAdecode không exact: dừng trước train.
NoReal control replay−Q6/Q7 tripwire |Δpooled|≤.001, |ΔtimeDelay/p10/low256mean|≤.01 cảhai seed.
Nếu fail: audit trước utility inference; không rerun/tune tìm PASS.
Binary/margin/strength giữ đúng Q20 formula; strength countweightedmass bằng margin từngxe cùngQ.
Contrast trong replay giữ teachertrajectory, nhưng private models/BN thay đổi; không tách riêng từng gradientclass.

## Chi phí, artifact, phạm vi

Repeatstream544B/chiều/r; bootstrap3311744B; tổng3333504B/xe/20round.
Replay giữ live-equivalent upload/down33344B/chiều/r, tổng4645504B gồm bootstrap.
Frozen history preload chỉ setup counterfactual riêng: báo decoded cache/source modulebytes,
không cộng cache rồi tính lại cùng downlink thành chi phí liveFL, không tuyên bố tiết kiệm liveFL.
Phase8replay cap5.3h/run,42.4account-hour; reserve15%→48.76h.
Q20 logelapsed mixedinput4.172/4.385h, real4.756/4.458h, margin4.031/4.079h, strength4.332/4.529h.
Cap5.3 lớn hơn worstlog~32.6phút, notebook giữ reserve0.4h và driver worst-round guard.
Không speedchange/HPchange; nếu budgetstop trước20 thì incomplete, kiểm checkpoint trước recovery.
Chỉ chấp nhận wholephase nếu từng owner đủcap×1.15; không ghép quota rời giữa owner để giả đủslot.
Repeat/validation chưa cấp ngân sách hoặc thay protocol, không suy tốc độ mới trước đo.

## Phase repeat — tài khoản bổ sung 02/11

Owner bổ sung tài khoản và cho phép push kernel còn cần; giới hạn16account trước đây được mở cho tài khoản mới đãlưu.
Inventory riêng `runs/r-repeat/quota.json`: sáu owner mới đều serveridentity/quotaPASS30h, không active/unknown trong latest10.
Chạy đúng4control R0/R1repeatAseed42/43, R2/R3repeatBseed42/43; không thêm phương pháp chỉ để lấp quota.
Cap5.3h/run, tổng21.2h +15%reserve→24.38h; Q0/Q1 measuredelapsed4.685/4.347h, vẫn giữ notebookreserve0.4h.
Mỗi run một owner ngoài16owner R/V đang có; hai owner mới còn lại giữ cho recovery/kiểm nghiệm tiếp.
HP giữ nguyên Q0/Q1, chỉ `r_kind=repeat` để factory trả nguyên Live; không attach frozen history.
Preflight R hai seed×3round đã chứng minh repeatA/B bitexact với Live trên realfixture; kiểm lại sourceSHA và45+7testreceipt.
Không trainlại localpreflight hoặc sửa runtime để dùng quota mới. Compile/T4/full5343/20round vẫn phải kiểm remote.
Manifest `r20-repeat-launch.json` khóa trước push, batch/watch/receipt riêng `runs/r-repeat`, artifact chung`artifacts/r20` với slugriêng.
Hai repeatedsession mỗi seed mô tả spread và độ ổn định cohort/percar; không là CI/noisebound hoặc causalharmcount.
W&B online/private inlinekey theo quyền đãcó; versionhistoryprivate giữ key. Poll1200s/pull/audit20round qua codex-long-jobs.

Preflight realv4W1/2seed/3round: repeat vs Q bitexact; replay vs manual injection cùngteacher bitexact;
r1 bốn receptionbitexact, receiveweights/mass, source tamper/teacherlag/codec, state/load strictrebuild,
diagnostics giữ RNG/state. Local sm86 không chứng minh T4/full5343bitexact.
Runtime require2T4/allrankcompilePASS/nonfiniteguard; exactowner/slug/version1, sourcebase+overlay/notebookSHA.
Audit20CM→10metric/xe/phân phối, source, finite snapshots, liveaggregates và consumed replaychain,
bankSHA, local support/cohort/absent/timeDelay, slopes15→20, byte/time và startupW&B.
Small snapshots không fullprivateweights/predictionreplay/fullcontentfingerprint/DP/contactproof.
Poll1200s bằng codex-long-jobs worker; COMPLETE phải pull/audit mới kết luận.

## Reference theo thành phần

Kế thừa [method](method.md): FD/KD classQ, shared inputconsensus thích nghi từ
[FedMD](https://arxiv.org/abs/1910.03581). R replay/reception là chẩn đoán dự án, không serverproxyfit
hoặc triển khai FedDF/PCGrad/FedAUX. Không gán paper cho thành phần chưa chạy.

## Phục hồi đóng gói trước training

Attempt đầu nhận400 tạiSaveKernel; toàn bộ8owner-ownedlisting đủpage sau lỗi xác nhận chưa có8slug.
Chưa xác định servervalidationlimit cụ thể; không gọi trainingfailure hoặc restart.
Giữ source/notebook/intent/log/manifest gốc ở `runs/r-study/push-recovery/attempt1` và notebookfolders gốc ignored.
Payload chuyển từ module4918846B sang blobprivate3684087B (decoded3993600B), wrapper7368B.
DecodedSHA108d314e8fc6e12393a1e5c8a6abe0ba9b6b20f4307f8e3ec68aa89e136db920 không đổi; không đổi model/loss/HP/history.
Input mỗi owner private, listingbytes và downloadSHA phải khớp trước push; CUDAdecodeexact kiểm trước cachebuild.
Slug mới suffix`-input1`, version1; manifest mới `r20-input1-launch.json` sau preflight.
Worker không coi packagingfixPASS là kernel đãlaunch/hoàn tất; receipt/startup/artifact vẫn bắt buộc.

Owner đồng ý nối task trên các account hiện có: completion phải tiếp tục next phase đãủyquyền, kiểm quota/slot và artifactgates trước push.
Không reset quota bằng nối session; runpartial chỉ resume fullprivatebank hai rank đãaudit, không dùng smallsnapshot.
Nếu phase tiếp không đủ whole-pairbudget, ghi blocker/quota thiếu; không hạ reserve/round hoặc launch một seed để lấy win.
