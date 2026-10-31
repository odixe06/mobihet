# Chẩn đoán tiếp sau M20 — thiết kế đề xuất, chưa launch

Cơ sở: [phân tích cơ chế](analysis-m20-mechanisms.md), [M20 đã audit](results-m20.md).
Giữ5.343xe, DAGSNetW1,20round/cosine50, seed42/43 cho screening FL; own-test từng xe vẫn chính.
Không đổi data split/optimizer/LR riêng một method; không chọn HP bằng officialtest.
Các phép bên dưới là giả thuyết của dự án, chưa gán tên bài báo hoặc gọi là phương pháp thắng.

## 1. Tách ảnh hưởng realKD khỏi synthetic fallback

M đã có3/4ô; thiếu realKDoff+fallback. Cần batch đối chiếu đồng thời để giảm nhiễu khác session:

| | synthetic softQ | synthetic fallback |
|---|---|---|
| realKD off | control giống M2/M3 | ô còn thiếu |
| realKD argmax-gate | control giống M4/M5 | control giống M6/M7 |

Cùng copula64, weight1, codecmeanfp16, countaggregation,20round;4ô×2seed=8run nếu được khóa để chạy.
**Câu hỏi:** giảm timeDelay/dao động đến từ realKD, fallback hay tương tác? Đọc cả main effect và interaction,
không cộng gain từ các cell. Q có feedback theo round: đây là total effect của cơ chế FL, không phải giữ nguyên teacher để đo riêng gradient.
**Dấu hiệu bác bỏ:** bỏ realKD không làm giảm dao động tương ứng; hoặc tác động fallback xuất hiện tương tự khi realKDoff.
Báo riêng15lớp còn lại, xe nhỏ/absent/rare; không chỉ pooledF1.

## 2. Kiểm thông tin offdiagonal Q bằng teacher replay

Teacher Q1…Q19 đã lưu từ train-only messages của M, có fingerprint/seed/lag. Dùng cùng lịch Q cho các đối chứng chẩn đoán:

- fullQ;
- giữ Qyy, chia đều offdiagonal: cùng xác suất đúng nhưng entropy thay đổi;
- giữ Qyy, hoán vị cố định15entry offdiagonal: cùng entropy/khối lượng, phá quan hệ tên lớp. Chọn permutation độc lập dữ liệu trước chạy.

RealKDoff, synthetic luôn soft trong phép này, cùng generator/sample schedule và seed. Không để argmax sau biến đổi tự mở gate:
Qyy<.5 có thể sai argmax ở fullQ nhưng đúng argmax sau uniformization. Nếu dùng gate ở phép khác, phải tính từ Q gốc và giữ riêng.
**Câu hỏi:** learned offdiagonal mang quan hệ hữu ích hay chủ yếu lan truyền confusion? FullQ hơn uniform chưa tách entropy;
fullQ hơn permutation mới có bằng chứng cho quan hệ tên lớp trong cấu hình replay.
**Bác bỏ:** fullQ không hơn hai đối chứng ở endpoint/cohort đã khóa. Không suy “offdiagonal vô ích mọi cấu hình”.
Replay là counterfactual chẩn đoán, không phải server triển khai đã chứng minh. Cảhai bên vẫn tính cùng payload đầy đủ;
chưa tuyên bố nén packet hay cộng gain replay vào liveFL.

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

## 4. Chỉ cải tiến sau khi biết kênh gây hại

**Gate liên tục theo lớp, không thêm byte:** `a_c=max(0, Qcc−max_{j≠c}Qcj)`.
Thử đổi realKDweight riêng hoặc synthetic mixture `(1−a_c)onehot(c)+a_c Qc` riêng trước thử kết hợp.
Khác gate nhị phân: khi margin đi qua0, weight/target liên tục. Không hardcode lớp13 hoặc chọn threshold theo test.
**Bác bỏ:** target/metric ít dao động hơn nhưng own-test/tail không cải thiện, hoặc15lớp còn lại bị hại.

**Tiếp nhận theo mẫu local:** thử chỉ distill khi teacher có xác suất trueclass cao hơn student, với weight detached;
cần giữ cùng quy ước rawlogit/LA và control cùng compute. Đây là giả thuyết chống kéo lùi mẫu đã học, không bảo đảm teacher đúng.
Không dùng testconfidence, không gửi input/feature hay output từng mẫu lên server.

**Thiếu bước ở xe nhỏ:** nếu can thiệp, so thêm syntheticKD với control cùng số bước/compute; không so model được train nhiều hơn
rồi gán gain cho teacher. Đổi optimizer persistence/localepoch phải theo protocol chung, không lách bằng method flag.

**Server:** tách “đại diện nhiều client” khỏi “độ tin cậy trên tương lai”. Chỉ thêm qualityscalar/class sau khi có phép ước lượng
train-only hợp lệ, kiểm support/độ bất định; báo byte thực của scalar/normalizer mới. Không chỉ sharpen confusion đang sai.

## Cổng trước launch

Khóa cell, source/config/data fingerprint, teacherlag, permutation và endpoint trước chạy; không chọn thử cảmọi combo để tìm win.
Kiểm loss/gradient đúng công thức và control tái lập; quota/2×T4/W&B theo skill. Báo đủ10metric, lớp, pairedclient/cohort,
bootstrap+upload/download/time. Bất kỳ giới hạn nào của artifact/DP/mixedarchitecture phải giữ nguyên phạm vi.
Backup/cleanup còn chạy ở [bàn giao](../operations/continue.md); tài liệu này chưa cho biết kernel đã được push.
