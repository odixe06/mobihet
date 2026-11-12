# Rà module CBB — 10/11/2026

Rà code/tài liệu/artifact có sẵn; kết quả mới theo [b13](../../../docs/research/screening-module-b13.md). Không mở `archive/`.
Nền = DAGSNet W1, KM16 w0 .5 + lite6 + QQ8 ([proposal](../../../docs/research/proposal.md)).
Owner trong phiên mới: cho đổi module và tăng compute; phương pháp phải dùng được với nhiều kiến trúc,
nhưng thí nghiệm đợt này chỉ DAGSNet; performance trước, báo communication/cost frontier, chưa có byte cap cứng.
Không đưa MLP/RFF classifier hoặc model phụ khác vào thí nghiệm. Nhận định về gain dưới đây là giả thuyết.

## Điều code thực sự làm

- Mỗi bước ghép real và synthetic trong **một** forward, BN chung; CEreal dùng logit adjustment,
  synthetic dùng logit thô. Q uplink là xác suất online của real rows trước optimizer update tại bước đó,
  thu trong train mode với dropout và BN của batch ghép. Không phải teacher out-of-sample hoặc forward cuối epoch.
  [stream loss](../../../methods/mobihet_wstudy.py), dòng 117–140;
  [BN](../../../core/grouped.py), dòng 57–73.
- Uplink là mean xác suất theo true class, sau codec; server cộng với trọng số số dòng/lớp.
  Q chứa đóng góp của chính client nhận teacher. Q r chỉ dùng từ r+1.
  [round aggregation](../../../methods/mobihet.py), dòng 505–632;
  [round teacher signatures](../../../methods/mobihet_next.py), dòng 131–155.
- QQ8 thực thi ở `mobihet_sim._round_private`: sửa số liệu ngay sau local epoch,
  rồi lượng tử Q downlink sau aggregate. Patch `mobihet_study_wire.roundtrip` không đổi stream CBB.
  [QQ8](../../../methods/mobihet_sim.py), dòng 789–833.
- `q_agg=sqrt_count` có implementation/test trong Base, nhưng Live CBB hiện assert `q_agg == "count"`;
  `q_collect=postepoch` cũng bị study assert online. Chỉ thêm `--set` chưa đủ làm ứng viên hợp lệ.
  [Base knobs](../../../methods/mobihet.py), dòng 124–143;
  [study constraint](../../../methods/mobihet_study.py), dòng 98–99;
  [Live constraint](../../../methods/mobihet_next.py), dòng 29–33;
  [wire tests](../../../tests/test_mobihet_teacher.py), dòng 89–154.

## Tránh nhầm cơ chế cũ thành đề xuất mới

| module | bằng chứng thực | phạm vi kết luận |
|---|---|---|
| sqrt-count Q | M10/M11 vs M4/M5, seed42/43: macro +.001482/+.001158; benign −.000513/−.000720; p10 +.001029/−.002576 | đã chạy; recipe cũ có realKD gate, không CBB; không gọi ứng viên mới hoặc PASS CBB |
| postepoch Q | M8/M9 vs M4/M5: macro −.000156/−.000104; benign −.000979/−.001115 | đã chạy; extra real forward, không có bằng chứng cải thiện macro |
| hard CE + NCKD synthetic | W4/W5 vs Q0/Q1: timeDelay recall +.0604/+.0612, precision −.0895/−.0842; mean F1 xe 64–256 dòng −.00118/−.00117 | đã thử, tradeoff; không gọi NCKD mới; bảng này không suy macro endpoint từ cohort mean |
| Q theo vùng input | bucket8 K10 và KM-cluster QK b11; QK ≈0 cả hai seed | họ đã đóng; thêm bucket không là thông tin mới |
| Q mềm, Q từ synthetic | QT2 b11 −.0041/−.0042 macro; QS hại suddenConstantSpeed | không mở lại qua đổi tên temperature/consensus |
| realKD, bank, EMA, prior | có kết quả trong [proposal](../../../docs/research/proposal.md), [F50](../../../docs/research/results-f50.md), [catalog](../../../docs/research/improvement-proposals.md) | không cộng các gain riêng; bank r20 không giữ tới r50 |
| BNsplit/projection/proxy cũ | [diagnosis](../../../docs/research/diagnosis.md) ghi đóng, không mở archive để suy diễn thêm | BN chung hiện tại là sự thật code, chưa chứng minh nguyên nhân; không gọi BNsplit chưa thử |

Nguồn số M20: [analysis](../../../evidence/m20-summary/analysis.json), mục `effects` và `arms.M4…M11.final`;
[audit](../../../evidence/m20-summary/audit.json), [diagnostic audit](../../../evidence/m20-summary/diagnostic-audit.json).
Contrast đúng ở [analyse_m20](../../../scripts/analyse_m20.py), dòng 132–138.
Nguồn W20: [analysis](../../../runs/w-study/analysis.json), `pairs.W4-Q0`, `pairs.W5-Q1`;
[closure](../../../runs/w-study/closure.json) xác nhận pulled/audited;
[cell definition](../../../scripts/w20_common.py), dòng 4–9.
Các số này dùng official test nhiều lần trong lịch sử, không là xác nhận độc lập.

## Nhánh có thể triển khai trước

Mỗi nhánh thay một cơ chế trên CBB cùng seed/control. Giữ 64 synthetic/bước, floor16, một epoch real,
protocol, split, kiến trúc và endpoint. Không lấy candidate metric để đổi gate hoặc tune tiếp.

| ưu tiên | nhánh | bằng chứng → cơ chế | gain giả thuyết → dấu hiệu bác bỏ | chi phí và heterogeneity |
|---|---|---|---|---|
| 1 | **sqrt-count trên CBB**; capped-count chỉ sau khi khóa riêng | số dòng client chênh lớn; sqrt-count cũ có gain nhỏ → giảm quyền chi phối của client nhiều dòng trong teacher, giữ nguyên real probabilities | macro tăng và FP benign→attack giảm; bác bỏ khi pooled/cohort không cải thiện, hoặc benign/timeDelay giảm quá guard | không thêm model forward/byte schema; teacher vẫn 16×16, hỗ trợ kiến trúc khác; phải sửa assertion và test exact weighted decoded packets |
| 2 | **JS hoặc reverse-KL trên synthetic** | full KL(Q‖p) ép cùng Q[y] lên mọi synthetic của lớp; synthetic lệch ranh giới là bằng chứng đã có → đổi hình học gradient để giảm áp lực từ synthetic không khớp student | benign FP giảm, macro tăng; bác bỏ khi chỉ dời precision↔recall timeDelay hoặc mất absent/rare/c11 | một forward hiện tại; thêm phép toán 16 lớp, 0 byte mới; kiến trúc độc lập. JS và reverse-KL là hai arm; không đồng thời đổi target/outer weight/codec |
| 3 | **same-class real–synthetic mixup** | model riêng fit train tốt nhưng own-test kém; bỏ synthetic hại → nối vùng real local với synthetic cùng nhãn để giảm học thuộc | macro/benign precision tốt hơn với cùng số forward; bác bỏ nếu c11/timeDelay/cohort nhỏ giảm hoặc gain chỉ có train | dùng đúng các row slots hiện có, không model phụ/byte mới. Phải khóa cách ghép khi lớp thiếu; dùng fallback synthetic gốc. Không mix cờ/atoms rồi coi là vật lý hợp lệ |
| 4 | **teacher dựa lỗi prequential / cross-fit** | Q online từ row thật vẫn in-sample qua nhiều round; confidence train cao chưa là generalization → đo chất lượng trên phần real chưa dùng để cập nhật tại thời điểm đo, gửi tổng error/count theo lớp làm trọng số | teacher quality ngoài fit và benign FP tốt hơn; bác bỏ nếu confidence tăng mà macro không tăng, hoặc xe ít mẫu không có đủ support | cần thêm forward hoặc lịch đo rõ ràng; chỉ gửi tổng/count, mọi backbone dùng được. Cross-fit thực sự cần model/prefix độc lập và kế toán compute; không gọi row pre-update một bước là fresh validation |

**Kết quả 11/11:** lựa chọn1 và2 đã thử ở [b13](../../../docs/research/screening-module-b13.md), mọi arm FAIL gate r20 cả2seed;
giữ các công thức/giả thuyết ban đầu trong bảng để đối chiếu, không coi là đề xuất chưa thử.
Lựa chọn 1 là **adaptation có căn cứ**, không novelty algorithm. Nhánh3 sau đó đã thử ở
[b14](../../../docs/research/screening-uplink-b14.md), cùng GP/LG; **cả3FAIL cả2seed**, không pilot/tune.
[Phân tích sau b14](b14-analysis-20261111.md) cập nhật ưu tiên; bảng trên giữ giả thuyết trước chạy.
Nhánh 4 là hướng thông tin mới rõ hơn nhưng thiết kế tốn công và có rủi ro dùng hết
local train trong protocol; cần khóa cách tạo đánh giá nội bộ mà không lấy official test hoặc gọi fold-val sim thành fresh confirmation.

Chi tiết tối thiểu khi định nghĩa arm:

- **JS:** `m=(Q+p)/2`; JS gốc `0.5 KL(Q||m)+0.5 KL(p||m)`, dùng trực tiếp biểu thức khả vi, target Q detach.
  Kiểm gradient theo logit của công thức tham chiếu. JS có thang gradient khác KL; outer weight cố định1 là adaptation đã khóa,
  gain/fail không tự chứng minh nguyên nhân do hình dạng loss thay vì cường độ gradient. B13 khóa **4JS**,
  giữ outer1, để khớp độ cong tại p=Q với forward KL; xem [định nghĩa arm](../../../docs/research/screening-module-b13.md).
- **Reverse-KL:** `KL(p||Q)` phải clamp Q như baseline, tính fp32 và giữ gradient qua p; mode-seeking có thể làm mất lớp hiếm.
  Không tự làm sắc/làm mềm Q. Đọc c11, absent và timeDelay cạnh macro; không suy "mode-seeking" thành gain FL.
- **Same-class mixup:** synthetic slot lớp y chỉ ghép real row có y của chính client và batch đã có;
  tránh thêm epoch hoặc nhìn test. Nhãn cùng lớp không tự chứng minh soft target giống nhau: cần khóa target và LA semantics.
  Nếu target hoặc số bước thay, ghi rõ không còn isolate input augmentation. Padded/floor-only slot phải fallback hữu hạn.
- **Reliability:** độ tự tin/entropy từ chính train logits không là độ đúng ngoài mẫu. Weight cần ngưỡng count tối thiểu,
  không dùng cell một dòng, có fallback pooled Q; gửi thêm thống kê phải tính đủ hai chiều/header.

## Nhánh sau, chưa nên launch ngay

- **Leave-one-client-out Q:** subtract phần đóng góp client khỏi tổng server có thể giảm self-feedback; chưa có bằng chứng
  đây là nút thắt. Không cần aligned hidden/model. Phải giữ packet contribution/count để subtract đúng sau codec,
  tránh trừ xác suất normalized từ Q8 rồi mặc định khôi phục exact aggregate; thêm downlink scale/count nếu cần.
- **Thông tin variance/disagreement:** chỉ gửi second moment xác suất/entropy theo lớp tốn ít byte hơn bank1024,
  nhưng mean/confidence không chứng minh expert tốt. Cần kiểm có khả năng dự báo lỗi ngoài fit trước dùng làm weight.
- **Tăng floor:** compute cap mới bỏ làm mở được đánh đổi, nhưng b12 đã thử F24…F64 và không PASS cả hai seed theo luật cũ.
  Muốn nghiên cứu frontier mới phải ghi là mục tiêu/gate mới trước run, giữ nguyên verdict cũ; không tune để đổi FAIL thành PASS.
- **RFF/MLP/feature-prototype model phụ:** không thuộc thí nghiệm DAGSNet-only hiện tại. Hidden prototype khác kiến trúc chưa
  có không gian chung; không tự cộng hidden vector khác basis hoặc private parameters.

## So APFed/CANPerFL đúng phạm vi

- [APFed implementation](../../../../fl_dagsnet_veremi_v2/methods/apfed.py), dòng 1–15, 56–87: private base được trộn với global base,
  classifier lấy global, thêm một pass học hệ số fusion trước local epoch; server FedAvg tham số. DAGSNet chỉ là backbone
  dùng chung trong benchmark. Mang parameter fusion sang CBB sẽ phá hỗ trợ kiến trúc khác; chỉ mượn mục tiêu giảm local drift,
  không coi việc copy module là phương pháp heterogeneous.
- [CANPerFL implementation](../../../../fl_dagsnet_veremi_v2/methods/canperfl.py), dòng 1–11, 24–31: FedAvg50 rồi **10 epoch cá nhân hóa**.
  Không gọi headline CANPerFL là cùng đúng 50 local epochs với CBB. Báo compute/communication của cả hai pha.
- Gap .9078→.9119 vẫn là benchmark descriptive (CBB hai seed, APFed một seed); chưa là significance.
  CBB hơn ở một số lớp/cohort, đặc biệt timeDelay; không đổi guard để chỉ giữ pooled rank.

## Kiểm chứng trước train

Khóa arm/HP/primary endpoint/guard/byte kế toán trước push. Code gate phải chứng minh knob chạy trên stream QQ8 thật,
knob-off đúng CBB, packet aggregate đúng decoded bytes, finite backward và mỗi client có applied step.
Sau đó sim200 fold-val hai seed → full-scale r20 cosine50 cùng-round control → resume r50 + seed43 r50 theo thang hiện hành.
Utility/10metric từ CM đủ support; density/probe loss/entropy/teacher agreement không thay utility.
Không khẳng định đã thử hết không gian module; báo ma trận nhánh đã thử/chưa thử và frontier performance/compute/byte.
