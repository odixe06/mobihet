# Chẩn đoán và bước tiếp

Ưu tiên own-test từng xe; nâng chất lượng xe ít mẫu/non-IID trong băng thông hạn chế, schema hỗ trợ khác kiến trúc.
Đọc [M20](results-m20.md) và [LESSONS](../../LESSONS.md); không nạp lại hồ sơ archive để bắt đầu phiên.

## Điều đã biết

- Distillation có giá trị nhưng phụ thuộc loss scale/target; L đã tách được fixed1 khỏi adaptive. Không tăng KD bừa cho xe nhỏ.
- Q theo lớp không biết input, argmaxđúng chưa bảo đảm hữu ích; packet nhiều hơn hoặc teacherconfidence đẹp chưa chứng minh utility.
- Pooled và lợi từng xe/lớp khác nhau; thiếu lớp tập trung ở xe nhỏ. Cần đọc mean/p10/support/cohort và pairedclientdelta.
- Mmetric cho thấy vẫn cải thiện10→20; riêng fallback dao động/plateau khi gate chuyển. Fullsnapshot audit PASS đủ16×20.
- Server√n có gainpooled nhưng tail chưa nhất quán; postepochQ chưa tăngpooled; localρ chưa hơn uniform ổn định.

## Đối chứng cần khóa, chưa launch

| vấn đề/bằng chứng | cơ chế cần tách | phép thử và dấu hiệu bác bỏ |
|---|---|---|
| fallback đổi gate rồi giảmtimeDelay | ngưỡng argmax mở lại realKD hay đổi synthetic target | factorial realgate dynamic/sticky × syntheticfallback dynamic/sticky; bác bỏ nếu không giảm dao động hoặc hại own-test/tail |
| Qargmax đúng nhưng margin nhỏ vẫn mởKD | mức tin cậy nhị phân gây chuyển target đột ngột | sau factorial, so alpha liên tục theo margin Qyy−maxQkhác; không hardcode lớp/threshold theo test; bác bỏ nếu dao động hoặc own-test/tail không tốt hơn |
| M12 đổi cảsynthetictarget vàrealKDweight | lợi đến từ mixture hay giảm realKD | giữ một kênh, thay kênh kia; không cộng gain hoặc chọn hệ số theo test |
| Qpacket chứa cảoffdiagonal | tri thức lớp hay chỉlabel-smoothing | fullQ vs giữdiagonal/phân phối offdiagonal đồngđều; kiểm utility trước compression/payloadmới |
| train tăng nhưng test có gap | tối ưu, temporalshift hay featureoverlap | nhánh fit mới foldtrain→foldval chưa học/statfit; không gọi trainF1 hiện tại là validation |
| bootstrap3,31MB chi phối | thông tin synthetic nào thật sự cần | chỉ mở compression/generator ablation khi xác định bottleneck; giữ cùngloss để không confound |

Đề xuất là chẩn đoán, chưa phương pháp đãchốt. Không hardcode lớp13; rule chỉ dùng thông tin train/local/server đã cho phép.
Giữgenerator/codec/protocol giữa từng phépso. Không reopen bucket8/anchor/GMM/MADE thiếu bằng chứng mới.

## Thứ tự thực hiện

1. Mfullaudit, bảng10metric mọi round và đường học/cohort/teacherlag đã xong; giữ evidence nhẹ để phân tích tiếp.
2. Tổng hợp số đo vào results, bài học bền vào LESSONS; chọn giả thuyết có falsifier.
3. Khóa cell/endpoint/HP trước launch;10–20roundchẩn đoán không chứng minh50round hoặc mixedarchitecture.
4. Bước đang chạy, quyền và việc tay chỉ ghi ở bàn giao sống. Owner hiện ưu tiên backup/cleanup trước khi launch thêm.
