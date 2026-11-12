# Nghiên cứu hiện hành

Đọc [proposal](proposal.md) để biết quyết định; [method](method.md) để biết code CBB (Core + KM16 + lite6 + QQ8) làm gì;
[protocol](protocol.md) là cấu hình chung, không sửa riêng cho ứng viên.

| câu hỏi | tài liệu chính |
|---|---|
| Cải thiện F1 / phương pháp mới: bắt đầu từ đâu, họ nào đã đóng? | [sổ tay xây dựng phương pháp](method-building-brief.md) |
| Cơ chế X đã thử chưa, ở study nào, kết quả? | [index phương pháp đã thử](tried-methods-index.md) |
| A1/B19 thử sửa generator theo nhầm lẫn thế nào, vì sao dừng? | [B19](screening-boundary-b19.md), [sổ tay §5](method-building-brief.md) |
| Phương pháp chính là gì, thành phần nào giữ/bỏ? | [proposal](proposal.md) |
| Core được chọn ra sao, bank r50 thế nào? | [F50](results-f50.md) |
| Lossless CDF đã chứng minh gì và bao nhiêu byte? | [L50](results-l50.md) |
| Oracle/prior/Q/floor đã kết luận gì? | [results-y50](results-y50.md) |
| Generator nào đã thử, vì sao đóng băng KM16? | [catalog generator](generators.md) |
| Luật và kết quả sim fold val b1–b8 (KM ra đời thế nào)? | [screening-sim-foldval](screening-sim-foldval.md) |
| KM full-scale r50 đạt gì? | [screening-km50](screening-km50.md) |
| Codec lite6, sim b9/b10, pilot full-scale lite6? | [screening-km-lite](screening-km-lite.md) |
| Đề xuất cải tiến ngoài generator, sim b11, privacy? | [catalog đề xuất](improvement-proposals.md) |
| Batch theo thứ tự: B13 module, B14 uplink, B15 SAM, B16 mean-KD, B17 macro, B18 nguồn ngoài fit — FAIL gì? | [B13](screening-module-b13.md), [B14](screening-uplink-b14.md), [B15](screening-sam-b15.md), [B16](screening-mean-b16.md), [B17](screening-macro-b17.md), [B18](screening-source-b18.md) |
| B20 giảm compute tại xe (SH/SM/SD): luật, sim, full-scale? | [B20](screening-compute-b20.md), [phụ lục luật](screening-compute-b20-addendum.md) |
| Băng thông từng gói, codec study, combo r20/r50 (CBB vs CQ8)? | [screening-codec](screening-codec.md) |
| Nút thắt hiện tại (TD 44% thiếu hụt macro) và bằng chứng còn hiệu lực? | [sổ tay §3](method-building-brief.md), [diagnosis](diagnosis.md) |
| So benchmark và đối chiếu bài gốc? | [benchmark r20](benchmark-r20-reference.md), [nguồn generator](generator-primary-sources.md), [papers](../../papers/README.md) |

Job/round/account đang chạy chỉ ở [continue](../operations/continue.md). Hồ sơ đã đóng:
[archive 07/11](../../archive/docs-20261107/README.md) (M→X, C10/W20/X20, probe CPU generator),
[archive 09/11](../../archive/docs-20261109/README.md) (standard/forest, KM fp32, Y50 design, launch receipts cũ),
[archive 13/11](../../archive/docs-20261113/README.md) (bài rà/khảo sát 08–11/11 dẫn tới B13–B18, đã gộp vào sổ tay).
Chuyển chỗ tài liệu không là chạy lại, sửa gate hoặc promote phương pháp.
