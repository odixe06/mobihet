# Chẩn đoán và bước tiếp

Ưu tiên own-test từng xe; nâng chất lượng xe ít mẫu/non-IID trong băng thông hạn chế, schema hỗ trợ khác kiến trúc.
Đọc [Q20](results-q20.md), [P21 bổ sung](results-p21-recovery.md), [P20 gốc](results-p20.md), [O20](results-o20.md), [N20](results-n20.md), [phân tích M](analysis-m20-mechanisms.md), [đối chứng tiếp](diagnostic-next.md) và [LESSONS](../../LESSONS.md); không nạp lại hồ sơ archive để bắt đầu phiên.

P20 gốc31đủ20/P21r0, recovery bổ sung C10seed43 đủ20: inputteacher hơn classmean cùngbank, nhưng chưa hơn streamingclassQ;
proxythuần chưa hơn ensemble; hybrid hơn proxythuần pooled hai seed, vẫn dưới streamingclassQ và chưa nâng đều xe nhỏ.
Modeldownlink không thêm utility ổn định. RealKD/margin có tradeoff absentclass/xe yếu,
không mặc định tăng teacherweight theo ítdata. Hướng hiện hành và falsifier ở [kết quả P](results-p20.md).
Chẩn đoán [transfer từngxe](p20-client-transfer.md) và [đề xuất toàn pipeline](proposed-method-p20.md)
tách support, expertweightserver và clientreception; chưa đổi kiến trúc hoặc nhận phương pháp tối ưu.

Q20 đã đủ16×20round: supportmix riêng chưa hơn ổn định, inputresolution nâng weakcohort, sqrtinputweight chưa giúp.
Margin nâng pooled/mean/p10 nhưng hại absent/timeDelay, thêm39.36%totalbyte. Controlquery cũng đổi percar;
ưu tiên đo spread và receptionreplay cùngteacher trước quy causalharm, validation sạch trước reliabilityserver.

## Điều đã biết

- Distillation có giá trị nhưng phụ thuộc loss scale/target; L đã tách được fixed1 khỏi adaptive. Không tăng KD bừa cho xe nhỏ.
- Q theo lớp không biết input, argmaxđúng chưa bảo đảm hữu ích; packet nhiều hơn hoặc teacherconfidence đẹp chưa chứng minh utility.
- Pooled và lợi từng xe/lớp khác nhau; thiếu lớp tập trung ở xe nhỏ. Cần đọc mean/p10/support/cohort và pairedclientdelta.
- Mmetric cho thấy vẫn cải thiện10→20; riêng fallback dao động/plateau khi gate chuyển. Fullsnapshot audit PASS đủ16×20.
- Server√n có gainpooled nhưng tail chưa nhất quán; postepochQ chưa tăngpooled; localρ chưa hơn uniform ổn định.
- N factorial đã tách đủ4ô: realKD hại timeDelay nhưng lợi other15/mean/p10 ở cảhai syntheticpolicy;
  fallback khôngreal vẫn dao động sớm. Bác bỏ “chỉrealKD gâydaođộng”, chưa có phương pháp thắng đều.
- N realKD hại timeDelay từr2 dù gate13đóng; cần kiểm crossclasseffect/feedback, không chỉ confidence lớp13.
  Gap train/test vẫn lớn ở xe≥4096; không chỉ thiếu mẫu/lớp.
- O8×20 đã audit: giữ Q nguồn, realKD tăng pooled/other15 nhưng vẫn hại13; teachertrajectory khác giúp phục hồi
  một phần so liveN. Uniform/permutation đều giảm pooled; uniform lại giúp meanxe nhỏ. Chưa có winner đều.

## Registry kiểm chứng và đề xuất

| vấn đề/bằng chứng | cơ chế cần tách | phép thử và dấu hiệu bác bỏ |
|---|---|---|
| fallback đổi gate rồi giảmtimeDelay | ngưỡng argmax mở lại realKD hay đổi synthetic target | N20 factorial đã audit: cảhai kênh có thể dao động, bỏreal chưa xóa daođộngfallback; không có winner đều; gate liên tục còn pending |
| realKD hại13 khi directgate13đóng | crossclasseffect và feedbackteacher | O20đã kiểm: cùngQ vẫn hại13, đổiteachertrajectory phục hồi một phần so Nlive; chưa isolate từng class/BN hoặc phân rã causal |
| Qargmax đúng nhưng margin nhỏ vẫn mởKD | mức tin cậy nhị phân gây chuyển target đột ngột | P/Q đã thử margin: nâng một số own-test/mean/p10 nhưng absent/timeDelay còn hại. Tiếp replay cùngteacher, không hardcode lớp/threshold theo test |
| M12 đổi cảsynthetictarget vàrealKDweight | lợi đến từ mixture hay giảm realKD | giữ một kênh, thay kênh kia; không cộng gain hoặc chọn hệ số theo test |
| Qpacket chứa cảoffdiagonal | tri thức lớp hay chỉlabel-smoothing | O20đã kiểm: fullQ hơn uniform/permutation pooled cả2seed; uniform lại lợi xe nhỏ. Giữ structure trước testcompression/adaptation mới |
| train tăng nhưng test có gap | tối ưu, temporalshift hay featureoverlap | nhánh fit mới foldtrain→foldval chưa học/statfit; không gọi trainF1 hiện tại là validation |
| bootstrap3,31MB chi phối | thông tin synthetic nào thật sự cần | chỉ mở compression/generator ablation khi xác định bottleneck; giữ cùngloss để không confound |

Đề xuất là chẩn đoán, chưa phương pháp đãchốt. Không hardcode lớp13; rule chỉ dùng thông tin train/local/server đã cho phép.
Giữgenerator/codec/protocol giữa từng phépso. Không reopen bucket8/anchor/GMM/MADE thiếu bằng chứng mới.

Chi tiết cell, confound, falsifier và cổng tại [diagnostic-next](diagnostic-next.md); Nfactorial/Oreplay đã chạy, gate/clientquality/validation còn pending.

## Thứ tự thực hiện

1. M/N/Ofullaudit và đủ10metric đã xong; giữ artifactN/O chưa backup, không recollect/restart.
2. Dựa P/O, ưu tiên teacherquality/support và tiếp nhận không làm mất khả năng nhận diện lớp chưa thấy;
   temporalvalidation sạch trước qualityweightserver. C10 đã đủ hai seed, chưa đủ utility để thay control; không đổiHP tìm win.
3. Khóa cell/endpoint/HP trước launch;10–20roundchẩn đoán không chứng minh50round hoặc mixedarchitecture.
4. Bước đang chạy, quyền và việc tay chỉ ghi ở bàn giao sống. M/N/O/P/Q và recovery P21R1 đã audit/phân tích;
   chưa launch phase mới hoặc tự đổi headlineprotocol. Hướng tiếp/falsifier tại kết quả Q.
