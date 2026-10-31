# V2 — bằng chứng, hợp đồng FL và kế hoạch kiểm chứng

Ngày 2026-10-11. Trạng thái: **đề nghị, chưa thực nghiệm**. [Index V2](README.md).
Mọi ngưỡng và giá trị cấu hình đề nghị trong bộ V2 phải khóa thành manifest trước khi có run;
không thay cổng cũ của P1/P2/S-a.

## 1. Bằng chứng quyết định hướng nghiên cứu

| đã biết | suy luận được phép | không được suy |
|---|---|---|
| H2 full tốt hơn diag; own-Gaussian kém Gaussian gộp, [P1](../probes/pilot-p1.md) | cấu trúc quần thể ở input hữu ích; cần giữ thông tin liên client | mọi cải tiến mô hình sinh đều tốt; mean đủ thay covariance |
| Proxy DAGSNet tốt hơn proxy nhỏ, FedAvg cao hơn private, [P1](../probes/pilot-p1.md) | cần giữ sức chứa và tri thức tác động toàn private model | model lớn vẫn tốt sau nén cực mạnh hoặc teacher nào cũng hữu ích |
| DML B0 không nhận proxy mới vẫn hơn local, [P2](../probes/pilot-p2.md) | phải có đối chứng không trao đổi và đếm phiên bản đã nhận | phần tăng đó là lợi FL qua kênh |
| KME/DM fit synthesis tốt nhưng không vượt đối chứng, [S-a](../probes/pilot-new-fl.md) | đo lợi dự đoán và đối chứng cơ chế, không chỉ loss synthesis | đã cô lập nguyên nhân bank tĩnh; mọi condensation đều thất bại |
| positionMirroring được FedAvg cải thiện, Gaussian chưa cải thiện | còn ranh giới chưa truyền tốt | đã chứng minh chỉ cần một quan hệ vật lý cụ thể là giải được |
| Receiver/session có shortcut, [dataset](../../../../knowledge/dataset.md) | báo group/old, per-class, nguồn tham gia và lệch nguồn | Dirichlet client là một xe/receiver độc lập thực tế |

Số liệu gốc giữ ở report được dẫn; không tạo bảng kết quả mới từ suy đoán. Hai seed là kiểm tra lặp lại tối thiểu,
không phải bằng chứng ý nghĩa thống kê hoặc triển khai đường thật.
Scaler hiện fit toàn train80%, gồm holdout nội bộ được tách sau; đây là preprocessing leakage cần công bố.
Official test20% không dùng chọn phương án. Mỗi private được chấm trên cùng holdout toàn cục,
không diễn giải metric đó là hiệu năng trên phân bố riêng của xe.

## 2. Bất biến và thuật ngữ

- **Private**: IDS triển khai ở xe, nhận 66 feature và xuất 16 logit. Hiện dùng DAGSNet 395.024 tham số.
- **Proxy**: DAGSNet cùng kiến trúc trên các xe, tham số được học cộng tác; private không phải gửi tham số riêng.
  Khác kiến trúc private trong tương lai không làm thay đổi giao thức proxy.
- **Mạng tỷ số** ở V2-C: vẫn DAGSNet 66→16, nhưng logit học nhiệm vụ data-versus-noise theo lớp.
  Không diễn giải 16 đầu ra này là xác suất IDS đã hiệu chuẩn.
- **Version**: trạng thái tri thức bất biến, có hash/schema/base version. **Round** là một epoch private;
  **contact** là cơ hội truyền; **version update** chỉ xảy ra khi đủ payload và cohort. Ba đơn vị không đồng nhất.
- **Cohort**: tập ≥5 client có payload hợp lệ cùng base/version/schema. Mỗi client đóng góp tối đa một lần/object.
  Số 5 là ngưỡng mô phỏng vận hành, không tự chứng minh chống suy ngược hoặc DP.
- **FL** ở đây: client tính loss/thống kê trên dữ liệu local; chỉ cập nhật/thống kê gộp hoặc synthetic hợp lệ đi qua RSU;
  tri thức gộp quay lại làm thay đổi private. Không có bước server huấn luyện trên raw data tập trung.

Giữ R1 contact hữu hạn, R2 ở thiết kế, R3 không raw local record rời xe, R4 66 feature/protocol hiện hành.
Không tree học được, không learned MLP/generator/encoder phụ. H2 moment gộp là nền đã có, không phải một GMM mới.
Không thêm temporal/map feature, không bật M4b, không chia lại partition.

## 3. Hợp đồng giao tiếp áp dụng cho cả ba hướng

### 3.1 Ai biết gì

Xe giữ raw record, nhãn, private model, gradient từng minibatch và buffer BN riêng. RSU chỉ dùng tổng cohort
của thống kê/cập nhật; không đọc từng client delta/sketch/hệ số trước gộp. Ngoại lệ synthetic từ noise đã chốt
không mở rộng thành quyền đọc histogram, latent hay loss từng record.

Secure aggregation mô phỏng chỉ là ranh giới truy cập của simulator. Chưa có cryptographic protocol triển khai,
chưa có DP hoặc bảo đảm chống client độc hại. Thêm nhiều query cohort có thể lộ qua phép hiệu giữa tổng;
không dùng nhóm gần giống nhau để tái dựng đóng góp cá nhân.

IDS phát hiện misbehavior của dữ liệu V2X; không đồng nghĩa thuật toán FL đã chống poisoning/Sybil participant.
Đề xuất hiện giả định client huấn luyện trung thực. Bảo vệ aggregation Byzantine là bài toán chưa kiểm nghiệm.

### 3.2 Gói và đồng hồ

Mỗi object có header đề nghị 32 B: loại/schema, object ID, base/new version, độ dài và checksum rút gọn.
Hash toàn source/model/fixture phục vụ audit lưu riêng; hash/manifest cần cho client xác nhận từ xa cũng phải trả byte.
Frame tối đa 1.024 B payload, header 16 B, ACK 8 B theo mô phỏng hiện có.

Đối với object ứng dụng dài P byte đã gồm header object:

`B_wire_min = P + 24 × ceil(P / 1024)`

Đây chỉ là lượng truyền thành công một lần. Phải cộng handshake, cohort roster, prior, seed/schema bootstrap,
retransmit, ACK/control khác và phần frame đã phát trước deadline. `B_crypto` tách riêng, chưa được coi bằng 0
khi tuyên bố chi phí secure aggregation thực. Chỉ có thể kết luận R1 dưới đúng simulator và overhead đã khai báo.

1. Xe dựng object ngoài contact, gắn base version; ngừng thay nội dung khi bắt đầu truyền.
2. Cursor tiếp tục frame còn thiếu; frame dở trả phí nhưng không commit. Nhận lặp không áp dụng lặp.
3. RSU đóng cohort trước, hoặc dùng cohort được định danh rõ; không trộn object khác base/seed/schema.
   Thời hạn đề nghị 5 round từ mở cohort; thiếu 5 client thì hủy hoặc lập cohort mới, không giảm ngưỡng.
4. Aggregate chỉ xuất hiện sau các contact tạo nên nó. Không cho xe đầu round nhận tri thức được xe cuối round gửi.
5. Xe chỉ kích hoạt object đủ checksum/dependency. Bản vá cần base đầy đủ; model nửa cũ nửa mới không hợp lệ.
6. Xe chưa nhận đủ vẫn train local. Hàng đợi bắt kịp có giới hạn; hết thời hạn thì ghi miss, không tải snapshot miễn phí.

B0≈4,5 kB, B2≈231 kB là **trung vị budget simulator**, không phải băng thông đường thật hay bảo đảm mỗi contact.
Mặc định thiết kế 1 contact/xe/round, kế thừa Q3 đang mở. Không tăng contact để cứu hướng thất bại.
Baseline unicast: phát cùng nội dung cho 20 xe vẫn tính 20 lượt nhận. Không mặc định RSU broadcast miễn phí.
Một bộ điều phối RSU logic theo simulator hiện tại; nhiều RSU độc lập chưa có protocol đồng bộ.
Nếu thêm backbone RSU–RSU/cloud, byte, độ trễ và quyền nhìn tổng phải được định nghĩa trước; không ngầm có server hợp nhất tức thì.

### 3.3 Prior và BN không được biến thành kênh miễn phí

Prior toàn cục lấy từ count gộp train, truyền qua kênh và version hóa. Chưa nhận prior: dùng prior local theo
fallback khóa trước; báo số client đủ điều kiện `gp`. Không chấm tất cả client bằng prior có sẵn ở evaluator.
Count cá nhân không lộ riêng: denominator phải nằm trong aggregate, dù simulator biết partition để dựng fixture.

V2-A/B chọn **BN buffer riêng ở xe** cho proxy để tránh gửi full running state kèm từng gói nhỏ.
Tham số affine BN vẫn được học/gộp. Sau nhận weight mới, xe tính lại buffer trên local train theo lịch cố định;
không dùng holdout/test. Đây là thay đổi so với H1 cũ, cần baseline dense cùng BN policy.
Hash version chung chỉ cam kết tham số học được; checkpoint/audit phải lưu cả BN riêng từng xe.
Không gọi proxy ở RSU là cùng một hàm với mọi proxy trên xe khi BN khác nhau.
Nếu dense control với chính sách này trượt cổng chất lượng, dừng carrier; không mặc định số proxy P1 còn giữ nguyên.

## 4. Thiết kế đánh giá chung — đề nghị

### Giai đoạn M: kiểm cơ chế trước huấn luyện lớn

Chỉ khi có yêu cầu code: local math/fixture nhỏ, qua guard RAM. Kiểm cộng được, codec/decode thật,
interrupted frame, duplicate ACK, base mismatch, cohort<5, resume bitwise trên cùng thiết bị.
Đánh giá giả thuyết riêng từng phương án ở file tương ứng; không dựng hàng loạt notebook trước cổng này.

### Giai đoạn U: trần chất lượng không giới hạn contact

Kaggle; cùng fixture pilot hiện có, 20 client, α=0,5, 2 seed42/43, 20 round×1 epoch;
private batch512, cosine1e-3→1e-5, AdamW wd0/reset mỗi round, AMP/clip như S-a.
Group/old chấm r5/10/15/20; headline r20 group gp, thêm head bắt buộc.
Proxy/ratio optimizer khác phải ghi rõ; không giả vờ tổng compute ngang khi có thêm model hoặc forward.

Cổng chung đề nghị, phải đạt **cả hai seed**:

- Private `group gp − local gp ≥0,03`, old gp tốt hơn local, benign F1 không giảm quá0,03.
- Đạt contrast riêng từng phương án. `head` dùng chẩn đoán, không thay headline sau khi thấy điểm.
- Báo teacher/proxy riêng; model phụ tốt nhưng private không cải thiện ⇒ chưa đạt mục tiêu IDS ở xe.
- Có counterpart không nhận aggregate mới, nhưng giữ cùng compute/lịch local khi khả thi.
  Nếu lợi không hơn counterpart này thì chưa chứng minh lợi từ trao đổi tri thức mới.

Không đưa prior hiệu dụng mới vào cùng arm với compression/density ratio. Nếu `gp` có vấn đề, mở nghiên cứu
calibration riêng với cổng riêng; không chuyển sang báo head để gọi hướng hiện tại thắng.

### Giai đoạn C: deadline và codec thực

Chỉ sau U. Mỗi hướng thắng chạy B0/B2 và phiên bản không giới hạn bằng **cùng** codec/prior/BN/cohort policy.
Đối chứng local có cùng quy tắc prior; thêm đối chứng chỉ nhận prior và H2 nền nếu hướng dùng H2.

Gọi `ΔU=F1(method,U)−F1(control,U)`, `ΔB=F1(method,B)−F1(control,B)` với control hợp lý đã khóa.
Đề nghị giữ `ΔB/ΔU≥0,70` nếu `ΔU>0`; đồng thời `ΔB≥0,01`, benign guard còn qua.
Báo thêm gain so local riêng để không nhầm retention của cơ chế với toàn bộ gain H2/DML có sẵn.
Ít nhất16/20 xe phải thực sự nhận ≥1 aggregate version mới trước r20; báo median/p90 first-use và số version hữu ích.
Không có đủ activation ⇒ fail khả thi kênh, dù local training vẫn tăng điểm.

Không lấy tổng byte/20 để che việc phiên bản đầu đến r19. Báo F1 theo round và byte tích lũy,
deadline miss, tuổi snapshot, catch-up debt, nguồn tham gia và độ lệch số lần phục vụ theo client.
So Pareto quality–byte–compute, không so hướng nhiều epoch với baseline ít epoch mà bỏ qua compute.

### Giai đoạn F: protocol đầy đủ

Chưa được mở trong phiên docs. Sau khi chọn hướng và được yêu cầu: theo [protocol](../../../research/protocol.md),
50 round/headline r50, toàn train80%, official test20% đúng lịch; không tune sau khi xem test.
R2 đa kiến trúc phải được chủ dự án cho phép riêng; hiện chỉ lập luận qua giao diện, chưa có số chứng minh.

## 5. Bằng chứng cần lưu nếu triển khai

- 10 metric: accuracy; precision macro/micro/weighted; recall macro/micro/weighted; F1 macro/micro/weighted.
  Tính từng confusion client rồi mean; không thay bằng metric từ confusion gộp.
- Prediction, confusion từng client, weights+BN từng client/version/round, config, source/fixture/payload hashes.
- CE real, CE synthetic/KD/NCE tách riêng; LR, applied/skipped, proxy version, cohort contributions, byte theo loại.
- Per-class, benign, positionMirroring/timeDelay; đường học; seed riêng, không gọi min–max hai seed là CI.
- Verify từ đúng source đã chạy; tái-inference kiểm mẫu checkpoint được định trước; nói rõ nếu chỉ xác minh prediction lưu.
- W&B API key và kết nối **online trước train**, log mỗi round; thiếu key/network thì fail. Key không vào docs/log.
  Local chỉ smoke; Kaggle pilot ưu tiên quota ít còn đủ budget, đọc quota lúc triển khai.

Không có kết quả hoặc chi phí GPU mới trong bộ đề xuất này. Không có thao tác tay cần làm hiện tại.
