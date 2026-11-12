# Rà cơ chế tri thức sau góp ý SAM — 11/11/2026

Owner: SAM có lợi tối ưu tổng quát, lo ngại công bằng với baseline; **vẫn cho chạy thử hiện tại**,
muốn ưu tiên module distillation/server và vị trí khác. Không hủy b15 hoặc sửa nguồn đang khóa.

SAM phải báo riêng **CBB+SAM**, kèm compute/ablation; gain không tự chứng minh aggregation/distillation CBB tốt hơn.
So FLmechanism nên giữ optimizer/LR/batch chung. Method có thể khác compute do cơ chế (APFed fusionpass,
CANPerFL50+10epoch đã ghi trong code), nên so cả performance/compute/byte; không mặc định mọi kháccompute đều invalid.
Chưa có quyền hoặc nhu cầu chạy SAM cho toàn bộ baseline chỉ để kiểm lo ngại này.

## Vị trí nào còn đáng xem

Các nhánh dưới đây được rà lại theo yêu cầu owner tra lịch sử: [audit v4](history-audit-20261111.md),
[audit legacy](history-audit-legacy-20261111.md). Phân biệt đã train/diagnostic/proposal với exact mechanism chưa tìm thấy.
Mainmodel chỉDAGSNet, message không phụ thuộc hiddenbasis.

| vị trí | thay đổi cụ thể | vì sao khác nhánh FAIL / rủi ro | cost và kiểm loại |
|---|---|---|---|
| **loss tiếp nhận thống kê Q** | KL theo meanprediction nhóm synthetic cùngclass, thay meanKL từngrow | thay thứ tự reduction, không chỉ đổi KL→JS/RKL; Q là mean nhưng từngrow bị ép cùngtarget. Rủi ro yếu đi regularization, cho phép lỗi cá thể bù nhau | **B16 MK/HK FAIL cả2seed**; cùngforward/64slot/floor/schema nhưng không đủutility. Đóng adaptation |
| **nội dung uplink** | gửi thêm moment2 probability trên realrows để biết biến thiên trong mỗi class/client | GP/LG chỉ thay meanprob/meanlog, không định danh distribution. Variance không là correctness, và trainvariance không là generalization | diagmoment2 thêm tối đa512B/chiều/round chưaheader/nén; fulltriangle16×136fp16=4352B/chiều. Chỉ lên khi biến thiên mang tín hiệu và utility giữ rare/TD |
| **teacher của người nhận** | server tạo Q riêng cho xe/nhóm domain từ cùng Q8client, có globalfallback | thay expertpopulation theo recipient, khác Q(y,KMregionx) và localmargin gate đãFAIL. Rủi ro mất transfer lớp vắng hoặc nhóm hóa chỉ đổi prior | có thể giữ packetQ16×16/xe và sốforward; servercompute/metadata/broadcast khác. Cần chứng minh domain dự báo khác biệt lỗi, rồi mới khóa grouping/weights |
| **đo tri thức ở đâu/khi nào** | kiểm Q trên trainblock thời gian/nhóm sự kiện tách thật trong shadow, hoặc recency-stat trong train | modelown-test tương lai; randomrow ngoàifit không là temporalvalidation. Recency-weightQ không tự gọi quality ngoàifit; thay nguồn đo, không thay generatorK | timestamp có sẵn; sidecar partition/count/extra inference. Main vẫn fulltrain. Bác bỏ nếu chỉ do classprior hoặc thiếu support |
| **server weight chất lượng theo class** | thêm count/error/precision ngoàifit, rồi mới weight expert thay count đơn thuần | Qdiagonal giống recallsoft; không đo FP từ lớp khác vào classc. Sqrtcount/GP/LG đãFAIL, không thay entropy bằng tên reliability | cross-fit/shadow có compute/bootstrap đáng kể. So weight với cùng packetnguồn trước kiểm livefeedback |

## 1. Distillation khớp đúng mức thông tin đã gửi

Code [WMixin._loss_stream](../../../methods/mobihet_wstudy.py) dùng meanKL từngsyntheticrow;
[Base aggregation](../../../methods/mobihet.py) tạo Q từ classmean realprobability, countweighted.
Nền64synthetic cân bằng16class: mỗi client/step có4row/class, labels được giữ cùngslot.

Hiện tại:

`Lrow = mean_c mean_i KL(Q[c] || p(x_ci))`.

Ứng viên:

`Lmean = mean_c KL(Q[c] || mean_i p(x_ci))`.

Không mean giữa **client**: mỗi modelclient vẫn objective riêng; không mean logits rồi gọi meanprob.
GiữQ/rawstudentlogits/realCE+LA, coefficient1 và r1hardCEsynthetic. Q chỉ dùng từr+1 nhưcũ.
P20/P29 đã dùng target mean nhưng vẫn per-rowKL. Exact KL-of-mean student chưa tìm thấy trong nguồn được rà.
[B16](../../../docs/research/screening-mean-b16.md) khóa C/MK/HK×2seed; HK giảm rowKL .5 làm dose comparator, chỉ từ r2.

Jensen cho `Lmean≤Lrow`; bỏ penalty độ phân tán quanh mean. Đây cũng là thay cường độ/constraint,
không claim gain nếu có chỉ do “sửa semantics”. Teacher mô tả realmean; mean trên4synthetic là estimatornhiễu,
và syntheticdistribution khácreal: không assert đây là objective đúng duy nhất.

Toy Q=[.8,.2], p1=[.9,.1], p2=[.7,.3]: Lrow≈.03507 nhưng Lmean≈0;
gradient theo logits của Lmean≈0, của Lrow khác0. Đây là khác biệt cơ chế, **không bằng chứng VeReMi utility**.
[Số/gradient](../../../runs/distillation-review-20261111/algebra.json).

[DIST NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/da669dfd3c36c93905a17ddba01eef06-Abstract-Conference.html)
cho một tiền lệ về giảm ràng buộc khớp prediction, dùng inter/intraclasscorrelation;
**công thức Lmean ở đây không phải DIST**. Không tự dựng intra-instance teacherrelation từ Q hằng rồi coi là thông tin đã quan sát.

## 2. Client đang nén mất loại thông tin nào

Hai phân phối output có cùngmean[.8,.2] nhưng variance0 và[.04,.04]; server chỉ có Q không phân biệt được.
Diagmoment2 `E[p_j²|y]` thêm16×16scalars; so với mean `E[p_j|y]` cho variancepopulation
`E[p²]−E[p]²`. Fullcov dùng `E[p pᵀ|y]`, triangle136/class.

Phân biệt quan trọng:

- **Betweenclient variance** có thể tính từ các Qclient và count đang nhận, nếu server giữ chúng; không cần upload mới.
- **Withinclient/per-row variance** không khôi phục được từ Qclient; moment2 thực sự là thông tin uplink mới.
- Trainprobability variance không tự là lỗi ngoài mẫu. Không dùng thấpvariance hoặc thấpentropy làm qualityweight vô điều kiện.

Nếu thử variance distillation, phải khóa estimator/normalization/support/quantizer/coef và expectation alignment real→synthetic;
không cộng thêm loss tùy ý hoặc tăng byte mà chưa xác định muốn phân biệt tình huống nào.
Ưu tiên diag trước về byte không phải kết luận fullcov vô ích; chưa launch cảhai.

## 3. Server có nhất thiết gửi cùng teacher cho mọi xe

CBB cá nhân hóa weights nhưng Q là global16×16 giống nhau. Có thể giữ teacher toànmạng cho lớp thiếu,
đồng thời aggregate expert phù hợp recipient khi có dữ kiện domain khác biệt.
Ví dụ `Q_k[c] ∝ Σ_j n_jc a_kj,c q_j[c]`, a dựa commoninputstatistics/quality đã kiểm;
không dựa hiddenvectors khácbasis, không mean parameters giữa architecturekhác.

Đây là adaptation **personalized expert aggregation**, có tiền lệ
[KT-pFL NeurIPS2021](https://proceedings.neurips.cc/paper/2021/file/5383c7318a3158b9bc261d0b6996f7c2-Paper.pdf)
về personalized linearcombinationsoftpredictions. Paper dùng publicdata để tính softprediction;
CBB không được tự giả định publicrealdata hoặc copy coefficientlearning mà thiếu query/gradient hợp lệ.
Không claim reproduction KT-pFL hoặc lý thuyết paper bảo đảm classQ adaptation.

Không gọi QK/K10 chưa thử: cácnhánh đó đổi target theo **vùnginputsample**; ở đây đổi **expert của recipient**.
Tuy khác estimand, vẫn phải kiểm không chỉ tái tạo prior hoặc cô lập nhómthiếuclass.
Gom expertgroup nhỏ có thể tránh computeN²; chưa chọn sốgroup/topk/kernelbandwidth trước diagnostic.
Tính riêng byteunicast vsbroadcast; mỗi xe nhận1table không nghĩa tổng mạngbroadcast nhưcũ.

## 4. Nguồn đo phải phản ánh mục tiêu tương lai

Schema thật của parquettrain có `rcv_time_ns`, `message_id`, `sender_id`, `scenario/world/run/client_id` ngoài77features.
Phiên này chỉ xác minh schema. **V20 đã chạy temporal internalvalidation** full5343/2seed;
[thiết kế](../../docs/research/screening-v20.md), [kết quả](../../docs/research/results-r20-v20.md).
V20 xác nhận gap ngoài fit nhưng chưa dùng quality trên val để weight teacher; không gọi temporalvalidation là chưa thử.
Không đưa world/sender thật làm inputclassifier; giữ77features đãkhóa.

LESSONS đã có attack–benign twin quaworld. Vì vậy rowdisjoint A/B chưa chứng minh sự kiện/thời gian độc lập;
countsupport99,8% trong [crossfitfeasibility](crossfit-feasibility-20261111.md) chỉ áp dụng giả định splitrow cân bằng.
Nếu đổi groupedmessage/blocktime split phải tính lại coverage/mincount. Không gọi randomcrossfit là dự báo temporalshift.
Đo hậu thời điểm train trong **traininternal** phải bảo vệ shadow khỏi bootstrap/Q của phần bị giữra;
mainmodel vẫn dùngfulltrain; không nhìn officialtest để chọn block/weight.

## 5. Quality aggregation cần nhìn precision, không chỉ confidence đúngclass

Muốn giảm benign→class11, chỉ biết accuracy/diagonal trên rowtrueclass11 là thiếu:
precision còn phụ thuộc các lớp khác bị dự đoán thành11. Q8mean mọi trueclass + count có thể dựng **softprecisiontrain**,
nhưng nó vẫn in-sample. Muốn dùng như quality cần statsoutofit/hardCM và count/support phù hợp.

[CRAD](https://arxiv.org/html/2609.00446v2) dùng classvalidationaccuracy (recall) và consensusfiltering;
paper trao peer modelsnapshots để đánhgiá tại recipient. Không phù hợp bytehạn chế nếu copy nguyên;
“precision” của paper là **inversevariance**, không PPV/TP÷predictedpositive.
Không lấy paper làm chứng cứ đã dùng FP-awarequality; matchedn, accuracy .05/.95 có cùngweight,
consensus còn có thể loại specialist tốt. [Rà nguồn và diagnostic mới](teacher-source-survey-20261111.md).
Không gán cho CBB khả năng đánhgiá peer trên dữ liệu local khi không có model/query.

## Quyết định hiện tại

Ngoài chínhdistillation/server, còn các vị trí cần phân biệt với cơ chế FL:

- **Định danh/mức độc lập/độ mơ hồ dữ liệu:** audit exact77featureduplicate nhưng khácnhãn,
  train/twin/group overlap và conditionalerror. Đây là diagnostic, chưa phát hiện lỗi dữ liệu mới;
  không tự relabel/thêmfeature/split khác cho riêngCBB. Nếu đổi dataset/commonfeatures thì baseline cũng phải dùng cùngversion.
- **Sampler/thứ tự realbatch:** class-interleaving không replacement có thể phân tán classhiếm trong epoch,
  giữ mỗirow đúng1lần/batch64. Khác oversampling/extraepoch, nhưng là kỹ thuật generic nhưSAM;
  chưa evidence gain, phải báo ablation riêng và ảnh hưởng BN/Qonline, không ưu tiên hơn cơ chế trao tri thức.
- **State/init/normalization:** optimizerstate hoặc pretraining có thể đổi hội tụ, nhưng thay commontraining/cost;
  localEMA/BNsplit đãFAIL, không gọi chúng chưa thử. Không coi đây là đường tắt công bằng hơn SAM.

Ưu tiên diagnostic dữ liệu/temporal vì có thể đổi cách đo tri thức mà vẫn giữ dataset/maintraining;
không tự chạy một grid commontraining mới khi câu hỏi đang tập trung cơ chế FL.

B15 đủ20r/2seed: SAM FAIL macro+p10 cả2seed, thêmTD seed42; đã đóng, không pilot/tune.
[Results](../../../docs/research/screening-sam-b15.md); nguồn gốc14file giữ frozen-source trước cập nhật tài liệu.
**B16 đã thử Lmean/HK**, utilityFAIL cả2seed; không tune hoặc ưu tiên launch lại lossnày.
Tiếp theo ưu tiên đo nguồnquality ngoài fit, rồi mới personalizedexpert/diagmomen2 khi có diagnostic.
[Temporalprobe](crossfit-feasibility-20261111.md) xác nhận support/prior khác countbalancedhalf; chưa teacherutility. Không mở lại divergence/sqrtcount/logpool/Qregion/BNsplit/generator/floor đãFAIL bằng tên mới.
Mỗi arm mới cần controlCBB **khôngSAM**, sourcefreeze/localgates và utilitygate cũ trước sim2seed→pilot20→r50two.
Trạng thái B16/localgate/push thực ở [continue](../../../docs/operations/continue.md); chưa kết luận giải được gap APFed.

B16 results: [thiết kế/kết quả](../../../docs/research/screening-mean-b16.md). Giảmloss đúngJensen không tăngmacro; per-row constraint có thể hữuích.
