# Chẩn đoán tiếp sau P20 — trạng thái và thiết kế đề xuất

**P đã đóng audit và phân tích:** [kết quả/bài học](../../../docs/research/results-p20.md), [audit](audit-p20.md), [byte/time](cost-p20.md).
Registry: margin/strength đã thử, pooledgain nhưng xe yếu/absent chưa an toàn; inputteacher matchedbank có utility;
proxythuần/modelcarrier chưa lợi; projectionflag không ổn định. [Hybrid đủ hai seed](results-p21-recovery.md)
hơn proxythuần pooled nhưng vẫn dưới streamingclassQ, xe nhỏ chưa hơn ensemble nhất quán. Chưa có phương pháp tổng.
Thứ tự hiện hành: validation train-only sạch → replay cùngproxy để táchsupport/feedback
→ servermemory/querybudget và clientreception có đối chứng. Thiết kế/falsifier ở cuối kết quảP;
các mục bên dưới giữ lý do và can thiệp M/N/O/P đã biết, không đọc như mọi cell còn chưa chạy.
Seed thiếu đã phục hồi/audit20round; originalP21FAIL giữ nguyên. [Đề xuất sau P](proposed-method-p20.md)
khóa thứ tự kiểm support→targetresolution→serverreliability→clientreception.
[Q20](../../../docs/research/results-q20.md) đã audit/phân tích16×20: inputtarget nâng weakcohort, sqrtinputweight chưa giúp;
margin tăng pooled/mean/p10 nhưng hại absent/timeDelay. Tiếp: repeatedcontrol/receptionreplay cùngteacher,
validation sạch trước qualityweight. Nhánh validationreliability/memory/gradientprojection chưa triển khai.

Cơ sở: [O20 đã audit](results-o20.md), [N20](results-n20.md), [phân tích M](analysis-m20-mechanisms.md).
Owner01/11 tiếp tục ủy quyền batch song song: [P20](screening-p20.md), [audit v1/features/benchmark](prior-evidence-p20.md).
P triển khai các giả thuyết tiếp nhận/teacherinput/proxy/support; không mở confidencequalityweight trước validation hợp lệ.
Giữ5.343xe, DAGSNetW1,20round/cosine50, seed42/43 cho screening FL; own-test từng xe vẫn chính.
Không đổi data split/optimizer/LR riêng một method; không chọn HP bằng officialtest.
Các phép bên dưới là giả thuyết của dự án, chưa gán tên bài báo hoặc gọi là phương pháp thắng.

## 1. Tách realKD khỏi synthetic fallback — đã chạy N20

N20 hoàn tất factorial4ô×2seed,20round. Thiết kế đã khóa ở [screening-n20](screening-n20.md):

| | synthetic softQ | synthetic fallback |
|---|---|---|
| realKD off | N0/N1 | N4/N5 |
| realKD argmax-gate | N2/N3 | N6/N7 |

Cùng copula64, weight1, codecmeanfp16, countaggregation,20round;8/8run đủ artifactaudit.
**Câu hỏi:** giảm timeDelay/dao động đến từ realKD, fallback hay tương tác? Đọc cả main effect và interaction,
không cộng gain từ các cell. Q có feedback theo round: đây là total effect của cơ chế FL, không phải giữ nguyên teacher để đo riêng gradient.
**Dấu hiệu bác bỏ:** bỏ realKD không làm giảm dao động tương ứng; hoặc tác động fallback xuất hiện tương tự khi realKDoff.
Báo riêng15lớp còn lại, xe nhỏ/absent/rare; không chỉ pooledF1.

**Kết quả:** realKD giảm timeDelay ở cảhai syntheticpolicy nhưng nâng other15/mean/p10; fallback khi noReal
vẫn dao động sớm, nên bác bỏ giải thích chỉrealKD gâydaođộng. Không có winner đều các cohort.
Đặc biệt realKD làm giảm timeDelay ngayr2 khi realgate13 đang chặn: cần tách crossclasseffect và feedbackQ,
không chỉ xử lý ngưỡng teacher13. N20 kiểm total effect, chưa isolate cùngteachertrajectory.

## 2. Teacher replay tách crossclass/feedback và nội dung offdiagonal — O20 đã audit8×20

Teacher Q1…Q19 từ train-only messages của N0/N1 đã lưu và đối chiếu SHA/seed/lag.
Đã chạy4ô×2seed, dùng lịch Q của đúng seed làm input cố định, cùng copula/sample schedule/codec/protocol:

- R0: fullQ, realKDoff, syntheticsoft — replaycontrol;
- R1: cùng fullQ, realKDargmaxgate, syntheticsoft — so R0 để tách tiếp nhận realKD khỏi feedbackteacher;
- R2: realKDoff, syntheticsoft, giữ Qyy/chia đều offdiagonal — cùng xác suất đúng nhưng entropy thay đổi;
- R3: realKDoff, syntheticsoft, giữ Qyy/hoán vị cố định15entry offdiagonal — cùng entropy/khối lượng,
  phá quan hệ tên lớp. Permutation độc lập dữ liệu đã khóa manifest/hash và kiểm runtime.

R0/R2/R3 không realKD và synthetic luôn soft; R1 gate tính từ rawQ. Không để argmax sau biến đổi tự mở gate:
Qyy<.5 có thể sai argmax ở fullQ nhưng đúng argmax sau uniformization. Nếu dùng gate ở phép khác, phải tính từ Q gốc và giữ riêng.
**Câu hỏi1:** realKD còn giảm timeDelay khi Q được giữ chung? R1−R0 còn giảm cho thấy không cần feedbackQ để tạo hại;
nếu giảm yếu/mất thì feedbacktrajectory quan trọng. Chưa isolate từng trueclass hoặc BN/gradientpath.
Ở round2, directKD13 trong R1 phải bị chặn theo rawQ1; kiểm từ artifact, không suy argmax sau training.
**Câu hỏi2:** learned offdiagonal mang quan hệ hữu ích hay chủ yếu lan truyền confusion? R0 hơn R2 chưa tách entropy;
R0 hơn R3 mới có bằng chứng cho quan hệ tên lớp trong cấu hình replay.
**Bác bỏ:** R1−R0 không lặp tác động dự kiến; hoặc R0 không hơn R2/R3 ở endpoint/cohort đã khóa.
Không suy “offdiagonal vô ích mọi cấu hình”.
Replay là counterfactual chẩn đoán, không phải server triển khai đã chứng minh. Cảhai bên vẫn tính cùng payload đầy đủ;
chưa tuyên bố nén packet hay cộng gain replay vào liveFL.
R0 phải audit teacherinput=sourceQ[r−1] và controlkhớp trong tolerance đãkhóa; không gọi CPUfixture/T4sessions là bitexact.
Owner đã yêu cầu tiếp tục kiểm chứng. Mã/preflight/quota và thiết kế O20 tại [screening-o20](screening-o20.md);
trạng thái push/kernel/worker chỉ ở bàn giao.

**Kết quả đã audit:** cùngQ, realKD vẫn hại timeDelay nhưng nâng cả15lớp còn lại và pooled cả2seed;
đổi teachertrajectory phục hồi một phần so liveN. Feedback không cần thiết để có hại cục bộ, nhưng teachertrajectory có ảnh hưởng.
Không gán riêng gradientclass/BN hoặc phân rã tỷ lệ causal. FullQ hơn cả uniform/permutation pooled;
uniform lại lợi xe nhỏ/timeDelay. R0tripwire cả2seedPASS. Chưa chốt liveFLmethod hoặc offdiagonal vôích.

## 3. Kiểm tổng quát hóa trước khi nâng “độ tin cậy teacher”

Gap recall timeDelay tồn tại cả xe nhiều mẫu và sau cân lại theo cùng xe/lớp. Cần nhánh dữ liệu chẩn đoán riêng:
fit foldtrain → chấm foldval theo thời gian, mọi prior/bootstrap/generator fit chỉ foldtrain; không lấy officialtest để chọn luật.
So real-only, augmentation và KD trên cùng nhánh. Đây không phải headline M; áp split/selection mới vào benchmark đòi hỏi
đổi protocol chung và chạy lại đối chứng liên quan, chưa thực hiện.

Phân tích feature theo lớp/thời gian/world, kiểm nhiễu nhãn/session và shortcut trước kết luận thiếu capacity hoặc không thể phân biệt.
**Dấu hiệu:** gap xuất hiện ở val chưa fit trước khi nhìn test, lặp qua cohort/seed. Gap chỉ trên in-sample vs test chưa tách overfit/feature shift.
Không lấy trainaccuracy hoặc Qconfidence làm qualityweight server đã được xác thực.

Probe BN chỉ trên clone checkpoint tương thích: giữ weights, BN-recalibration bằng localtrain rồi so CM trước/sau theo quy tắc khóa trước;
không fit BN bằng test. Cần privatebank đúng rank; snapshot nhỏ M hiện có không chứa privateweights.
Đây là kiểm giả thuyết batch cuối real/synthetic hiện tại, không mở lại toàn bộ BNsplit cũ. Nếu không có gain ổn định thì dừng hướng này.

## 4. Cải tiến từng kênh, không coi gate13 là nguyên nhân duy nhất — chưa chạy

**Gate liên tục theo lớp, không thêm byte:** `a_c=max(0, Qcc−max_{j≠c}Qcj)`.
Thử đổi realKDweight riêng hoặc synthetic mixture `(1−a_c)onehot(c)+a_c Qc` riêng trước thử kết hợp.
Khác gate nhị phân: khi margin đi qua0, weight/target liên tục. Không hardcode lớp13 hoặc chọn threshold theo test.
**Bác bỏ:** target/metric ít dao động hơn nhưng own-test/tail không cải thiện, hoặc15lớp còn lại bị hại.
Marginweight cũng giảm strength KD; gain chưa chứng minh continuity là nguyên nhân. Cần đối chứng strength hoặc replay
với lịch weight được khóa từ train-onlyQ trước khi gán nguyên nhân. N20 cho thấy realKD các lớp khác có thể tác động13
ngay khi gate13đóng; phải báo perclass/cohort, không chỉ số flip13.

**Tiếp nhận theo mẫu local:** thử chỉ distill khi teacher có xác suất trueclass cao hơn student, với weight detached;
cần giữ cùng quy ước rawlogit/LA và control cùng compute. Đây là giả thuyết chống kéo lùi mẫu đã học, không bảo đảm teacher đúng.
Không dùng testconfidence, không gửi input/feature hay output từng mẫu lên server.

**Thiếu bước ở xe nhỏ:** nếu can thiệp, so thêm syntheticKD với control cùng số bước/compute; không so model được train nhiều hơn
rồi gán gain cho teacher. Đổi optimizer persistence/localepoch phải theo protocol chung, không lách bằng method flag.

**Server:** tách “đại diện nhiều client” khỏi “độ tin cậy trên tương lai”. Chỉ thêm qualityscalar/class sau khi có phép ước lượng
train-only hợp lệ, kiểm support/độ bất định; báo byte thực của scalar/normalizer mới. Không chỉ sharpen confusion đang sai.
O cho thấy teachertrajectory có ảnh hưởng nhưng không chứng minh EMA/sharpen/postepoch đã đủ; postepoch M chưa hơn control,
không mở lại timing chỉ từ replaygain. Không dùng class13test để hardcode rule; giữ cấu trúc offdiagonal đã có utility.

## Cổng trước launch

Khóa cell, source/config/data fingerprint, teacherlag, permutation và endpoint trước chạy; không chọn thử cảmọi combo để tìm win.
Kiểm loss/gradient đúng công thức và control tái lập; quota/2×T4/W&B theo skill. Báo đủ10metric, lớp, pairedclient/cohort,
bootstrap+upload/download/time. Bất kỳ giới hạn nào của artifact/DP/mixedarchitecture phải giữ nguyên phạm vi.
Backup/cleanup trước N đã xong. N20 đã đóng; runtime/jobownership và trạng thái backup N chỉ ở [bàn giao](../../../docs/operations/continue.md).
Mục2 O20 đã hoàn tất/audit; mục3validation còn đề xuất, mục4 một phần đang kiểm trong P20 theo thiết kế riêng.
Trạng thái launch/job duy nhất ở bàn giao; không chạy thêm launcher ngoài worker sở hữu P.
