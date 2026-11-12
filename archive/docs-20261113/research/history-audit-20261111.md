# Audit hướng cải tiến trước B16 — 11/11/2026

Owner yêu cầu rà kỹ nội dung cũ rồi mới thử. Đã tìm có mục tiêu active docs/code,
`archive/docs/research` và `archive/legacy-v1`; không coi archive là trạng thái hiện hành.
[Legacy audit](history-audit-legacy-20261111.md) ghi riêng nguồn còn/thiếu và protocol cũ.
Không cần mở lại mọi thí nghiệm hoặc train lại để xác minh tên ý tưởng.

## Kết luận cơ chế, không chỉ tên

| Hướng | Đã thử gì thật | Chứng cứ / kết quả | Quyết định |
|---|---|---|---|
| Mean theo lớp | P20 C6 `bank_class`: teacher per-input được mean thành Q lớp, rồi KL **từng dòng**. C7 giữ per-input. P29 legacy cũng KL từng dòng với cùng class target. | [Code tạo target](../../../methods/mobihet_study.py), [loss thực thi](../../../methods/mobihet_wstudy.py), [P20](../../docs/research/results-p20.md). C7−C6 macro +.009095/+.008153 nhưng vẫn kém stream C2; không bằng chứng nên mở lại bank. | Chưa tìm thấy exact `KL(Q || mean p_student)` trong nguồn local. Có thể kiểm một loss khác, giữ carrier/compute/byte. |
| Bank/query/proxy/target theo vùng | P20 Q20 F50, K10 và b11 QK đã có; b11 QK gần0. F50 bỏ bank vì gain r20 không giữ tới50. Proxy không tăng đơn điệu theo độ mạnh. | [P20](../../docs/research/results-p20.md), [F50](../../../docs/research/results-f50.md), [b11/catalog](../../../docs/research/improvement-proposals.md). | Không mở lại bank/Qregion/proxy bằng tên “teacher chi tiết”. |
| Divergence / force KD | B13 reverse/symmetric KL/4JS đều FAIL hai seed. Tăng synthetic weight/row và giảm synthetic support đã có, không miễn phí. | [B13](../../../docs/research/screening-module-b13.md), [sim](../../../docs/research/screening-sim-foldval.md), [LESSONS](../../../LESSONS.md). | KL-of-mean là đổi constraint, cũng có thể chỉ yếu KD. Phải có giảm-dose control nếu muốn diễn giải cơ chế. |
| Server weights / uplink | M20 sqrtcount gain nhỏ trên recipe cũ; b13 CBB sqrtcount FAIL. B14 GP/LG actualwire audit PASS nhưng utility FAIL cả2seed; sharpen Q không đủ. | [M20](../../docs/research/analysis-m20-mechanisms.md), [B14](../../../docs/research/screening-uplink-b14.md), [artifact](../../../runs/sim-kaggle-b14/wire-audit.json). | Không tune meanlog/entropy/count. Precision/out-of-fit là estimand khác, cần nguồn đo hợp lệ. |
| Second moment | Legacy P30 khớp mean teacher + `Cov(p_teacher,x)` khi tối ưu INPUT; joint−mean −.000232, FAIL. Input covariance KM đã có. | [Legacy audit](history-audit-legacy-20261111.md). Không raw code/JSON P30 local để tái audit. | Chưa thấy direct `E[ppᵀ|class]` experiment, nhưng “chưa thử” không đủ lý do launch: chưa chứng minh moment đó dự báo utility. |
| Teacher theo recipient/domain | Existing `_client_clusters()` chia scenario/run cho generator, không personalized Q expert. Legacy personalized branch/P6/ConflictGraph chủ yếu proposal; ProxyFuse đã FAIL. | [Code Base](../../../methods/mobihet.py), [legacy matrix](history-audit-legacy-20261111.md). | Còn khả năng, nhưng chưa diagnostic domain→error ngoài classprior. Chưa launch clustering/weight grid. |
| Temporal validation / quality | **V20 đã train**, full5343 DAGSNetW1, train32.685.166 → internalval4.676.107, hai seed. CE recovery và aug/mixed/margin đều có closure. V20 là diagnostic, **chưa dùng val/OOF để weight teacher**. | [V20 design](../../docs/research/screening-v20.md), [R/V result](../../docs/research/results-r20-v20.md). CE TD trainR .886→val .267 ở xe lớn; margin gain macro .0006/.0005 nhưng TD−.077/−.076. | Sửa phát biểu “temporal chưa thử”. Không lặp validation/margin gate. Quality teacher ngoài fit vẫn mở, chi phí/independence phải khóa riêng. |
| Reception gate / gradient conflict | R replay giữ teacher đã tách livefeedback: real/binary/margin/strength hại TD ngay cả fixedteacher. Legacy P29 gradient cosine là diagnostic, không PCGrad utility. | [R/V](../../docs/research/results-r20-v20.md), [legacy audit](history-audit-legacy-20261111.md). | Không gọi margin/gradient âm là bằng chứng nên chặn peer update. Không mở lại gate đãFAIL. |
| Generic optimization / BN / sampler | SAM b15 đủ20r/2seed audit PASS nhưng utility FAIL. BNsplit/recalibration/head-only và EMA đã thử. Real stratified sampler natural-client chưa tìm thấy exact; legacy replay interleave khác. | [SAM](../../../docs/research/screening-sam-b15.md), [prior evidence](../../docs/research/prior-evidence-p20.md), [legacy audit](history-audit-legacy-20261111.md). | Đóng SAM adaptation, không tune rho. Sampler generic hạ ưu tiên vì fairness và chưa evidence. |

## Đánh giá đầy đủ tới đâu

V4: đọc source tạo/nhận target và báo cáo có khóa gate, đối chiếu closure/analysis/audit JSON còn local.
Index [evidence-index.json](../../../runs/history-audit-20261111/evidence-index.json) ghi SHA, tồn tại, kiểu dữ liệu;
không biến việc hash JSON thành predictions→CM replay. P20/V20/M20 có recipe/endpoint khác CBB sim:
không chuyển trực tiếp gain, threshold hoặc verdict. B13–B15 đã tái audit artifact thực trong phiên này;
full CM→10metric/report, source/config và counter/wire chỉ ở scope được từng auditor ghi.
Legacy thiếu rawsource/JSON: dùng reported historical audit, không tuyên bố đã tái xác minh toàn bộ.
“Chưa tìm thấy” giới hạn ở lịch sử local được rà; không khẳng định novelty literature hoặc lịch sử đã mất.

## Chọn thử nghiệm hẹp

**B16: KL-of-mean student**, control CBB mới, thêm **half row-KL** làm dose comparator.
Không thêm thông tin teacher, không sửa support/synthetic fidelity, không bảo đảm giải benign future FP.
Mean objective cho phép row cùngclass khác prediction mà mean khớp Q; realCE quyết định phân bố cá thể.
Rủi ro: lỗi cá thể bù nhau, giảm regularization, xe ít mẫu thiếu anchor; Jensen gap không phải utility.

Evidence→mechanism→falsifier: b13/b14 loss/pooling FAIL + classQ broadcast hiện tại ép cùng target từngrow
→ đổi thứ tự reduction, không đổi divergence/carrier → cần macro≥+.002 **từngseed** và giữ guard cũ.
Nếu chỉ loss giảm hoặc benign tăng đổi TD/p10 thì FAIL; không tune/sweep lại adaptation.
Half-dose coefficient .5 cố định làm comparator, không giá trị “tối ưu”; nó chỉ kiểm một dose, không loại mọi giải thích do dose.
Nếu MK không hơn HK, không gán lợi riêng cho meanconstraint. Utilitygate so từng arm với CBB, không với nhau.

Moment2/recipient/OOF được giữ là hướng cần diagnostic tiếp; không launch đồng loạt vì thiếu causal evidence/support/cost.
Thiết kế, localgates, budget và thang scale riêng ở [B16](../../../docs/research/screening-mean-b16.md).

**Cập nhật sau thử nghiệm B16:** MK/HK đềuFAIL hai seed, không lênpilot/tune. [Kết quả](../../../docs/research/screening-mean-b16.md).
Phân biệt lịch sử ở trên vẫnđúng; trạng thái “chưa exact objective” chỉ mô tả trướcB16, không còn currentstatus.
