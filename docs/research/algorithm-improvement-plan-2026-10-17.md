# Cải thiện huấn luyện FL cho IDS IoV: cập nhật client và trao đổi tri thức

Ngày: **17/10/2026**. Trạng thái: **đề xuất kỹ thuật, chưa triển khai hoặc launch**.

**Ưu tiên:** giữ DML + m4b làm nền; sửa cách private tiếp nhận teacher; phát triển Gaussian + m4b; kiểm FedDyn/SCAFFOLD trên proxy để giảm drift. Câu hỏi chính: **client cập nhật model thế nào, cần nhận tri thức gì từ các client khác, và tri thức ấy có cải thiện private model hay không?** Không tiếp tục nguyên trạng các adaptation đã FAIL.

**Owner chốt mới nhất trong phiên, thay ưu tiên contact trước đó:** cải thiện huấn luyện trước; kịch bản số lượng contact bổ sung sau. Giai đoạn A kiểm chất lượng với trao đổi logic đầy đủ, đồng bộ, **chưa áp deadline/contact cap**. Giữ cùng tần suất tổng hợp giữa candidate và control để cô lập cách học; không cần chốt số contact vật lý để viết hoặc kiểm thuật toán. Giai đoạn B mới kiểm compression, contact và khả năng triển khai. `sub4/sub16` là bằng chứng về ảnh hưởng tần suất tổng hợp, chưa chứng minh R1 thực tế.

## 1. Mục tiêu, phạm vi và mức chắc chắn

- Mục tiêu chính kế thừa quyết định 16/10: **mean-client macro-F1 của private model, rule `gp`, trên group holdout toàn cục**. Proxy tốt hơn chỉ là bằng chứng phụ; phải chuyển thành lợi cho private.
- R1 vẫn là yêu cầu triển khai sau: giới hạn byte/deadline, bootstrap, hai chiều, metadata, cohort, ACK và truyền lại. **Chưa dùng R1 để loại hướng có triển vọng huấn luyện**; luôn ghi loại/kích thước thông tin phải trao đổi để đánh giá sau.
- R2: private có thể khác kiến trúc. Proxy chung được cùng kiến trúc; private không average tham số/hidden state với proxy hoặc client khác.
- Giữ 66 vị trí input, scaler, partition và class map hiện tại. m4b đặt 13 tọa độ về 0 trong không gian đã chuẩn hóa, **có thay đổi thông tin đầu vào**; không được ghi là full66 nguyên trạng.
- Không đưa record thật/record biến đổi/activation hay gradient từng record ra khỏi xe. Không giả định có public dataset. Secure sum mô phỏng không phải bằng chứng secure aggregation mật mã hoặc DP.
- Giai đoạn hiện tại vẫn DAGSNet. Kiểm đa kiến trúc là bước xác nhận riêng, chưa được phép coi là đã đạt R2 từ giao diện lý thuyết.
- Mọi ngưỡng/cấu hình mới dưới đây là **đề nghị khóa trước thực nghiệm**, không thay các cổng lịch sử. Không dự báo F1 tuyệt đối hoặc bảo đảm đạt một mức IDS chưa được owner đặt.

Đã đối chiếu [CONTEXT](../../CONTEXT.md), [LESSONS](../../LESSONS.md), [bàn giao mới nhất](../operations/continuation-2026-10-15-evening.md), [danh mục phương pháp](../proposals.md), [bằng chứng lịch sử](../archive/overview/evidence-summary.md), [protocol](protocol.md), [dataset](../../knowledge/dataset.md), [DAGSNet](../../knowledge/architecture.md), thiết kế/kết quả FullDrift, AntiMem, Gaussian, DML, Hybrid và các hướng nghiên cứu 13–16/10.

Phân tích dùng báo cáo **đã audit sẵn trong checkout**, không audit lại toàn bộ tensor/predictions và không kiểm trạng thái remote phiên này. Seed43 vòng 6–7 vẫn **chưa có kết quả audit trong tài liệu đã đọc**; không suy thành PASS vì thời gian dự kiến đã qua. Riêng source FullDrift dùng cho các nhận xét mã ở §3 khớp SHA256 source đã thu của session vòng 5: `2b36282b093ad097410249fc01fadfc3772292f09c99a5628e366678903765e8`.

### 1.1. Những đặc điểm FL bắt buộc giữ

1. **Dữ liệu phân tán:** mỗi client chỉ train trên dữ liệu của mình và tri thức/synthetic được tạo từ thông tin được phép nhận. RSU không train bằng bản sao record của tất cả xe.
2. **Học local và cộng tác tách được:** local optimizer cập nhật model; server tổng hợp thông tin nhiều client rồi gửi trở lại. Candidate phải hơn local/no-peer và own cùng recipe; augmentation/EMA tự nó không phải lợi FL.
3. **Private độc lập:** kiến trúc, weights và BN riêng không cần cùng shape với client khác. Nếu cần tham số chung cho tổng hợp, chỉ proxy làm giao diện; không biến private thành bản FedAvg đồng dạng bắt buộc.
4. **Thông tin có nguồn gốc:** mỗi update/moment thuộc train local, có phiên bản và trọng số tổng hợp rõ. Không dùng holdout/test cập nhật proxy, moment, teacher hoặc prior huấn luyện.
5. **Giữ boundary chia sẻ hiện tại:** không upload record/activation/gradient từng record; server học từ tổng cohort được mô phỏng. Aggregate model update không đồng nghĩa không có rủi ro riêng tư; chưa claim DP/mật mã khi chưa triển khai.
6. **Non-IID vẫn còn:** giữ partition gốc, không trộn dữ liệu giữa client để làm bài toán dễ hơn. Đánh giá mỗi private model, không thay bằng model server rồi gọi là client đã học tốt.

### 1.2. Cần trao đổi thông tin gì, tác động lên model nào?

| Hướng | Client → RSU | RSU → client | Cập nhật tại client | Thông tin cộng tác mới so local |
|---|---|---|---|---|
| DML nền; KD cap/frozen teacher | Update proxy, BN/state cần thiết; private không gửi | Proxy gộp và version | CE local + KL từ proxy/teacher vào **toàn private backbone và head** | Hàm phân loại đã học từ nhiều client, không chỉ prior |
| Gaussian + m4b/EMA | Theo lớp: count, tổng input, tổng tích ngoài | Mean/covariance lớp gộp, count/prior và schema | CE real + CE synthetic trên toàn private; EMA là recipe local | Cấu trúc input–nhãn bậc hai ngoài dữ liệu riêng của xe |
| Hybrid private-only synthetic | Cả proxy update và moment | Proxy gộp + moment gộp | Private học cả teacher trên real và hard-label synthetic; proxy giữ objective đối chứng | Kết hợp decision function và support input quần thể; phải kiểm bổ sung thật |
| FedDyn-proxy | Proxy update/model; history local không gửi | Proxy đã hiệu chỉnh bởi history server | Proxy thêm linear/quadratic correction; private nhận tri thức qua KL | Cải thiện cách tổng hợp/hướng tối ưu của tri thức proxy hiện có |
| SCAFFOLD-proxy | Proxy delta **và control delta** | Proxy **và global control vector** | Sửa gradient proxy bằng `c−c_i`; private học qua KL | Ước lượng sai lệch hướng cập nhật giữa client và quần thể |
| Response consistency | Không thêm thông tin ngoài kênh FL nền | Không thêm | Hai view local + consistency; backbone bớt phụ thuộc receiver | Không tạo tri thức cộng tác mới; chỉ giúp tận dụng kênh đã có |
| Buffered proxy, giai đoạn B | Proxy delta kèm base version/transaction | Snapshot có version | Giữ local learning, đổi thời điểm tiếp nhận/tổng hợp | Làm tri thức đến đúng lúc khi hệ thống không đồng bộ |

**Lựa chọn có căn cứ:** ưu tiên function knowledge trong proxy và covariance input vì đã có utility nội bộ. Control vector là thông tin tối ưu mới đáng thử vì drift có bằng chứng. Chưa ưu tiên mở thêm kênh prior/risk scalar/prototype mean vì các controls cũ cho thấy chúng thiếu decision boundary. Các paper và khác biệt adaptation được dẫn tại §4–6.

### 1.3. Chu kỳ cập nhật logic cho giai đoạn A

```text
Khởi tạo: partition/scaler/class-map cố định; private_i riêng; proxy chung nếu phương pháp cần.
Mỗi round:
  RSU phát tri thức gộp của round trước: proxy / moments / control vector.
  Mỗi client giữ dữ liệu tại chỗ, cập nhật private và proxy theo loss của arm.
  Client gửi đúng payload đã khai báo; private weights không bị average.
  RSU tổng hợp cohort bằng rule đã khóa; tạo phiên bản tri thức round kế tiếp.
  Evaluator độc lập chấm private trên holdout; kết quả không đi vào optimizer.
```

Gaussian trên dữ liệu tĩnh là ngoại lệ có chủ ý: snapshot moment có thể chỉ tổng hợp một lần, rồi tiếp tục local replay; không giả vờ có tri thức mới mỗi round. Đối chứng DML-own cũng giữ hai model và cùng số bước để không nhầm lợi mutual learning nội bộ thành lợi giữa client.

## 2. Phân loại hướng còn tiềm năng và hướng nên dừng

### 2.1. Bằng chứng mạnh nhất hiện có

Các số dưới là **group holdout, full train trừ holdout, seed42**, trừ dòng ghi rõ pilot. Không trộn với official test hoặc full train80% không holdout.

| Hướng | Bằng chứng | Diễn giải phục vụ đề xuất |
|---|---|---|
| DML + m4b | ep6 private gp 0,7010 so 0,6655; ep15 0,7323 (Δ≈+0,036); proxy ep15 0,8510 | Còn khả năng học. Tác động input mạnh hơn các biến thể KD đã thử; chưa biết r50/seed43 |
| DML + swappos | ep6 private 0,6985, tăng 0,033 | Đối chứng thay nuisance thay vì bỏ hoàn toàn; không mặc định tốt hơn m4b |
| DML m4b + q4 | ep6 private 0,7010, gần như không đổi so dense | Có bằng chứng utility của lượng tử hóa; wire/contact còn cần kiểm |
| DML m4b + sub4 + q4 | ep6 private 0,7319, hơn m4b 0,031 | Teacher được cập nhật thường hơn giúp; **không chứng minh một contact/round đạt cùng lợi** |
| Gaussian + m4b | ep6 private 0,6553 so 0,5882; benign +0,142 | Nhánh ít phụ thuộc model chung, thích hợp R2; chất lượng hiện thấp hơn DML m4b |
| Gaussian + swappos + private EMA | ep6 private EMA 0,6768 so non-EMA 0,6404 | Có tín hiệu ổn định quỹ đạo. Không cộng số này với lợi m4b để dự báo kết hợp |
| Hybrid | Pilot r20 hai seed hơn từng thành phần, interaction âm; full cũ phẳng và owner dừng | Chỉ xem xét thiết kế mới sau khi có bằng chứng riêng; không nối run full đã dừng |

Nguồn: [FullDrift](../methods/full-drift/results.md), [AntiMem hai seed](../methods/anti-memorization/results.md), [Hybrid pilot](../archive/methods/gaussian-proxy-hybrid/builds/hybrid-pilot/results/report.md). Các phép so mới phải dùng số chưa làm tròn trong CSV.

Ba cơ chế có bằng chứng: **receiver shortcut**, **drift khi học local lâu**, **private hấp thụ chưa hết tri thức proxy**. Chưa đủ bằng chứng quy mọi lỗi cho một nguyên nhân. Ví dụ m4b cải thiện mạnh nhưng private–proxy vẫn cách khoảng 0,119 tại ep15.

### 2.2. Không tiếp tục adaptation hiện tại

| Nhóm | Vì sao dừng | Điều kiện tối thiểu nếu nghiên cứu lại |
|---|---|---|
| ClassTail, CCR, class-mass/scalar score summaries | Gần local, không chở đủ ranh giới; CCR pooled−CE chỉ +0,0017 | Phải bổ sung thông tin phụ thuộc input/hàm thực sự; chỉnh trọng số scalar chưa đủ |
| BoundaryCells, kernel anchors, anchor residual | Coverage tốt nhưng targets yếu/mềm; không qua utility so own/shuffle/Gaussian | Phải chứng minh teacher trên holdout thật mạnh hơn nhãn Gaussian; không chỉ tăng ô/anchor |
| CountSketch và repair, seeded subspace, random-coordinate/rolling/projection/PowerSGD adaptations | Codec/contribution hoặc norm contraction không giữ utility dense | Chỉ xem lại khi có nguyên nhân cô lập và cách sửa vượt gate teacher fidelity; q4 là đối chứng mạnh hơn hiện tại |
| KME/Fourier, latent-mean và gradient synthetic books | Synthesis/đạo hàm/chi phí PASS nhưng quality FAIL | Đầu tiên chứng minh synthetic mang decision boundary, không kéo dài synthesis loss |
| Gaussian ratio, MALA, support/geometry, Nyström adaptations | Support/fidelity hoặc compression không chuyển thành chất lượng | Không mở lại bằng đổi precision/rank/sampler tùy ý; cần cơ chế mới và matched control |
| Joint energy DAGSNet | Sampler/cost FAIL; chưa có utility r20 đầy đủ | Giải quyết cost/support trước; không gọi là đã chứng minh bất khả thi về F1 |
| NTD, hidden KD đã thử | Private Δ −0,042 và +0,006 ở full; không qua gate | Không quét tiếp cùng loss để tìm điểm thắng |
| FedProx μ0,01; FedAvgM β0,9 | Prox không tăng utility, sub4 còn giảm; AvgM phân kỳ | Không bác toàn họ. Thiết kế mới phải khác cơ chế hoặc có chẩn đoán bước tối ưu rõ |
| EMA DML/proxy, ProxyFuse hiện tại | EMA làm chậm quỹ đạo đang học; fusion không hơn proxy | Không tự chuyển headline sang ensemble để cứu private |

Nguồn/route tới reports: [danh mục đầy đủ](../proposals.md), [bằng chứng archive](../archive/overview/evidence-summary.md), [LESSONS](../../LESSONS.md). **FAIL cấu hình không chứng minh toàn họ thuật toán “không thể cải thiện”.** Với bằng chứng hiện có, kết luận đúng là không đáng chi thêm ngân sách cho cấu hình ấy.

## 3. Hai điểm từ source cần đưa vào thiết kế mới

### 3.1. q4: utility đã kiểm, wire bytes chưa được chứng minh đầy đủ

Trong [fd.py](../../methods/full-drift/builds/fd-probe/source/proj/fd.py), `quantize` tính `nbytes = ceil(numel*bits/8) + bytes_scale`; q4 vì vậy được báo 208.260 B/chiều, 416.520 B hai chiều. Nhưng `q` trả về `int8`; `aggregate_quantized` ghi `q.numpy().tobytes()` vào `latest_packet`, chưa pack hai nibble/byte. Buffer/integer có tính vào bộ đếm nhưng không được serialize đầy đủ cùng packet trong nhánh này; upload cộng tensor đã decode trực tiếp vào `qsum`.

Hệ quả: các run chứng minh hành vi học với lượng tử q4 trong mô phỏng; **7,65× là tỷ lệ theo bộ đếm codec dự kiến, không phải phép đo byte end-to-end của packet hiện tại**. Không sửa source/notebook đã freeze. Build tương lai cần:

1. Codec thực: pack/unpack signed int4, scale, padding lẻ, BN buffers/counters, schema, base/target version và checksum.
2. Chỉ cập nhật model từ bytes đã decode; cấm đường tắt truyền tensor ngoài payload.
3. Đếm `len(serialized_packet)` và byte frame thật; kiểm bit/shape/scale parity với tensor q4 cũ.
4. Contact bị cắt chỉ lưu mảnh; snapshot hoạt hóa sau đủ gói/checksum. Bootstrap và resync khi thiếu base cũng tính phí.

Đây là cổng R1 bắt buộc trước khi kết luận phương pháp vừa một contact. Không cần chạy lại utility chỉ để xác nhận bit packing nếu decode tái hiện đúng tensor; nếu codec thay giá trị/scale/BN thì phải kiểm noninferiority mới.

### 3.2. Trọng số KD tăng khi CE local giảm: giả thuyết, chưa phải nguyên nhân

Source hiện dùng:

```text
w = 1 / stopgrad(max(CE_private + CE_proxy, 1e-6))
L_private = CE_private + w KL(p_proxy || p_private)
L_proxy   = CE_proxy   + w KL(p_private || p_proxy)
```

`w` có trần số học 10^6, nhưng không có trần thực dụng nhỏ. CE giảm từ 1 xuống 0,1 khiến hệ số tăng 10×, dù teacher có thể đang học thuộc receiver. Chưa có log gradient KD/CE đủ để kết luận đây là lỗi gây plateau. Đây là lý do chọn phép kiểm giới hạn hệ số tại §4.2, thay vì thêm một kênh hidden KD nữa.

## 4. Cải tiến phương pháp còn tiềm năng

### 4.1. Nền đối chứng huấn luyện: m4b + proxy dense

**Cơ chế giữ nguyên:** private `f_i` và proxy `g_i` học ở xe; RSU chỉ gộp proxy đồng dạng. Private giao tiếp với proxy qua 16 logits tại dữ liệu **ở ngay xe**. Cấu trúc này có tiền lệ [Federated Mutual Learning](https://arxiv.org/abs/2006.16765); lợi trên VeReMi phải lấy từ báo cáo nội bộ.

m4b áp đồng nhất train/synthetic/eval cho private và proxy. Mask đúng 9 feature noise/profile receiver và 4 vị trí tuyệt đối theo [AntiMem source](../../methods/anti-memorization/builds/antimem-pilot/source/proj/antimem.py). Không xóa cột, không đổi thứ tự 66 feature; 0 sau z-score không phải giá trị vật lý 0.

Nền nghiên cứu giai đoạn A: `fd_dml_m4b` dense cho tách cơ chế; trao đổi logic đầy đủ, cùng một lần tổng hợp/epoch với control hiện có. Đây là lịch tối ưu đối chứng, **không phải giới hạn contact vật lý đã chốt**. q4 cùng recipe chỉ là giai đoạn B. Đặt baseline riêng **no-peer/mutual-local m4b**, cùng hai model và compute, để tách lợi FL khỏi regularization local.

Không thay tần suất tổng hợp cùng lúc với loss rồi quy toàn bộ lợi cho loss. `sub4/sub16` có thể làm đối chứng chẩn đoán riêng cùng compute, ghi số lần tổng hợp; mô phỏng contact để sau. Cũng không so m4b mới với DML chưa mask rồi quy toàn bộ lợi cho kỹ thuật mới.

### 4.2. DML: KD có giới hạn và teacher đóng băng giữa hai lần tổng hợp

**Vấn đề:** private học teacher trên x local; teacher trainable cũng drift theo local. NTD/HKD không giải khoảng cách; tăng tần suất nhận teacher có ích nhưng chưa cô lập được cách private hấp thụ tri thức ở cùng lịch tổng hợp. Vì vậy kiểm cách tạo target và trọng số KD trước.

**Sửa A — giới hạn trọng số, không đổi teacher:**

```text
w_cap = min(w_current, 1.0)
L = CE_private + CE_proxy
    + w_cap * [KL(stopgrad(p_proxy) || p_private)
               + KL(stopgrad(p_private) || p_proxy)]
```

`1.0` là cấu hình sàng lọc đề nghị, không giá trị tối ưu đã biết. Giữ T=1, không đổi prior/gp, optimizer hoặc input. Log median/p95/max hệ số, tỷ lệ chạm trần, KL, CE và tỷ số norm gradient KD/CE trên batch probe cố định từ train. Nếu hiếm chạm trần, giả thuyết này yếu; không kỳ vọng tăng lớn.

**Sửa B — teacher snapshot:** giữ `T_i` là bản proxy gộp đã nhận ở đầu round, `eval()`, stop-gradient, không cập nhật BN/Dropout/weights đến snapshot gộp tiếp theo. Giai đoạn A nhận đầy đủ mỗi vòng logic; giai đoạn B mới xét contact commit. `g_i` vẫn train và gửi update như cũ. Chỉ thay nguồn target cho private:

```text
L_private = CE(y, f_i(M(x)))
            + w_cap * KL(stopgrad(T_i(M(x))) || f_i(M(x)))
L_proxy   = CE(y, g_i(M(x)))
            + w_cap * KL(stopgrad(f_i(M(x))) || g_i(M(x)))
```

`M` là m4b. Để phép kiểm B chỉ thay nguồn teacher, hệ số vẫn tính từ CE của private/trainable proxy như arm A. Sau này thay luật hệ số là một arm khác. Teacher chưa có snapshot quần thể: KD từ teacher đóng băng tắt; private/proxy vẫn CE. So kèm control cùng warm-up để không quy lợi cho việc bỏ KD đầu run.

**Ablation tối thiểu:** nền m4b; A; A+B. Nếu B có tín hiệu, thêm teacher local-own đóng băng với cùng lịch refresh; không upload private. Teacher frozen là adaptation của dự án, không gán cho FML như kết quả có sẵn.

**Chi phí:** không thêm payload/contact; thêm một bản proxy và một forward teacher/batch cho B. Chỉ riêng 398.319 phần tử params+buffers FP32 khoảng 1,59 MB bản teacher; peak activation/optimizer RAM phải profile riêng. Cần cả so cùng real-data exposure lẫn cùng compute/walltime; không gọi lợi thêm forward là miễn phí.

**Kỳ vọng định tính:** vừa; hấp thụ tri thức ổn định hơn có thể giúp private giữa hai lần tổng hợp. **Bác bỏ:** chỉ giảm KL, teacher stale làm private kém; Δprivate <+0,010 hoặc benign giảm >0,005. Nếu teacher frozen không tốt hơn trainable trên group và private không được lợi, dừng B. Không đổi tần suất tổng hợp để cứu cùng arm sau khi thấy kết quả.

### 4.3. Gaussian: m4b + EMA, tận dụng một snapshot thống kê nhiều round

**Vấn đề:** Gaussian có tri thức covariance quần thể nhưng local memorization làm suy giảm. m4b và EMA có tín hiệu riêng; chưa đủ bằng chứng rằng kết hợp luôn cộng được.

Giữ Gaussian full covariance đã audit, không thêm MALA/ratio/generator:

```text
(n_c, s_c, Q_c) = sum_i (n_ic, sum x_ic, sum x_ic x_ic^T)
mu_c = s_c / n_c
Sigma_c = Q_c / n_c - mu_c mu_c^T   # giữ regularization PSD của recipe gốc
x_syn ~ N(mu_c, Sigma_c)
L_i = CE(f_i(M(x_real)), y) + rho * CE(f_i(M(x_syn)), c)
```

Giữ `rho`, aux-batch và prior của recipe đối chứng. Không đưa dữ liệu E/group vào moment. Dataset tĩnh: chỉ cần bootstrap thống kê đủ/coherent một lần; **không gửi lại cùng moment mỗi round**. Mẫu synthetic vẫn được sinh bằng seed mới tại xe nên thay seed không tốn contact và không đồng nghĩa có tri thức quần thể mới.

Kiểm 2×2: Gaussian/m4b × non-EMA/EMA, thêm own-Gaussian + m4b + EMA. EMA dùng recipe đã kiểm (half-life hai epoch), BN tính lại bằng subset train local cố định. Đánh giá đúng head/BN đi với backbone; không tự áp EMA đã FAIL của DML lên nhánh này. Nếu triển khai EMA-backbone + fit head riêng, tách thành arm khác và có compute-matched refit control.

**Cải tiến payload có căn cứ:** m4b làm 13 tọa độ luôn bằng 0; chỉ cần truyền moment của **53 tọa độ còn lại**, rồi tái dựng vector66 bằng mask. Đây là Gaussian marginal chính xác trên phần model nhìn thấy, không phải low-rank/Nyström xấp xỉ covariance. Kiểm phân phối/PSD và predictions; không yêu cầu bit-exact chuỗi sample với Cholesky66 vì RNG/factorization có thể khác.

Ước lượng codec FP32 tam giác đối xứng cho một chiều của 16 lớp, count uint64:

```text
B(d) = 16 * [8 + 4*d + 4*d*(d+1)/2]
B(66) = 145856 B; B(53) = 95104 B
hai chiều d=53: 190208 B, chưa header/scaler/cohort/ACK
```

Moment raw sum có thể cần FP64 vì triệt tiêu số học; nếu FP64, phần float nhân đôi. Không được dùng số FP32 trên để kết luận B2 PASS trước kiểm sai số/counter overflow/serializer. Mask53 là hạng mục codec riêng; đo quality parity trước ghép EMA.

**Lợi kỳ vọng:** chất lượng vừa; kiểm tăng F1 trước, lợi byte tái dùng snapshot đánh giá sau. Có thể vẫn thua DML về F1; báo trung thực khoảng cách, không dùng ít byte thay gate chất lượng hiện tại. **Dừng:** EMA không hơn non-EMA +0,010; pooled không hơn own +0,010; lợi chỉ đến từ mask/compute local. Không mô tả học tiếp từ moment tĩnh là FL tiếp nhận tri thức mới.

### 4.4. Hybrid: chỉ mở nhánh sửa cơ chế, không nối Hybrid cũ

Nguồn cũ có pilot tích cực nhưng full phẳng; thêm Gaussian vào DML không tự đem synergy. Nhánh có lý do kiểm lại là **m4b trước, rồi synthetic dạy private**, vì bỏ shortcut đã thay động học full.

Đề nghị arm tối thiểu: giữ proxy và lịch tổng hợp của DML-m4b; thêm CE trên Gaussian pooled **chỉ cho private**. Proxy không học synthetic trong phép kiểm đầu. Giữ nhãn hard của lớp sinh, chưa thêm KD trên Gaussian.

Controls: DML-m4b; DML-m4b + own-Gaussian; DML-m4b + pooled-Gaussian; DML-m4b + replay local cân bằng lớp cùng số forward/steps. Phải hơn DML-m4b và own/replay ≥0,010, benign guard như §7. Nếu đã có kết quả Gaussian-m4b+EMA mạnh, so thêm với thành phần mạnh nhất cùng ngân sách. Báo interaction khi có đủ factorial; không suy synergy từ tổng điểm.

**Giá:** cần bootstrap moment ngoài proxy, thêm inference/training private. Giai đoạn A truyền đủ moment và proxy dense để kiểm chất lượng; không loại Hybrid vì chưa vừa contact. Giai đoạn B nếu dùng q4, logical bootstrap hai chiều moment53 FP32 khoảng190 kB cộng proxy khoảng417 kB; phải kiểm lại cold start/cap. Xếp Hybrid sau hai nhánh độc lập để biết nguồn tri thức nào tạo lợi, tránh ghép nhiều thay đổi cùng lúc.

Chỉ cân nhắc soft target Gaussian từ frozen proxy nếu arm hard-label thắng và teacher hơn control hard-label trên dữ liệu thật giữ lại. Gaussian-logit distillation cũ đã thua nhãn miễn phí; confidence cao trên synthetic không đủ mở lại FedGen-like adaptation.

### 4.5. Response consistency phía client: mở rộng swappos có kiểm soát

Đây là hướng **cải tiến recipe local**, chưa phải họ FL mới. Không mở shared-response scalar channel đã thiếu thông tin ranh giới.

Giữ 4 vị trí bị mask như swappos; tạo hai view bằng giữ/swap **nguyên khối 9 feature receiver** từ receiver khác trong cùng client. Giữ joint profile, không trộn từng tọa độ tùy tiện. Không biến đổi tốc độ, khoảng cách, timestamp hoặc residual chứa tín hiệu attack.

```text
p_a = f_i(T_a(x)); p_b = f_i(T_b(x))
L_local = 0.5 * [CE(p_a,y) + CE(p_b,y)] + lambda * JS(p_a,p_b)
```

Đề nghị `lambda=1` cho một phép sàng lọc, control `lambda=0` **cùng hai forward**. KL DML còn lại giữ nguyên, dùng cùng quy tắc average qua hai view cho cả hai arm. So thêm m4b mạnh nhất; nếu chỉ hơn unmasked nhưng thua m4b thì không ưu tiên. Receiver label được dùng để chọn donor local, không đưa vào predictor; thiếu donor hợp lệ thì skip và log tỷ lệ.

Nguồn ý tưởng: [Right for the Right Reasons](https://www.ijcai.org/proceedings/2017/371) regularize dependence vào input; [R-Drop](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf) dùng consistency giữa hai forward dropout. Phép swap+JS trên IDS ở đây là **adaptation riêng**, không kết quả được hai paper chứng minh.

Giả định cần kiểm: thay khối receiver không đổi nhãn nhiệm vụ. Không gọi đây là counterfactual vật lý đã đúng. Kỳ vọng vừa/thấp so m4b; 0 byte thêm nhưng tăng compute. Dừng nếu consistency tốt mà private group/benign không tăng; không thêm trao đổi risk weights để cứu arm.

## 5. Phương pháp mới đáng kiểm

### 5.1. FedDyn trên proxy: sửa drift bằng lịch sử, không thêm control-vector payload

**Lý do chọn:** sub4 cho thấy drift đáng xử lý. FedProx chỉ kéo về model chung và đã không tăng utility. FedDyn thêm số hạng tuyến tính theo lịch sử để chỉnh mục tiêu local; không đơn thuần đổi μ của FedProx. Kiểm liệu proxy và private học tốt hơn ở **cùng tần suất tổng hợp**, độc lập bài toán contact. Nguồn: [FedDyn, ICLR 2021, Algorithm 1](https://arxiv.org/pdf/2111.04263).

Ký hiệu `w_t` proxy server, `a_i` trạng thái lịch sử local, `h_t` trạng thái server, N tổng client, S cohort. Dạng nguyên bản:

```text
w_i = approximate_argmin_w [F_i(w) - <a_i,w> + alpha/2 * ||w-w_t||^2]
a_i <- a_i - alpha * (w_i-w_t)
h   <- h   - alpha/N * sum_{i in S}(w_i-w_t)
w_next = mean_{i in S}(w_i) - h/alpha
```

Chỉ áp trên **tham số trainable proxy đồng dạng**; không regularize private theo khoảng cách weights. BN buffers/counters theo baseline riêng, không đưa vào vector gradient history. Private giữ loss DML-m4b; các biến `a_i,h` khởi tạo0, lưu qua round và checkpoint. Proxy delta/model là payload; `a_i` local và `h` server không gửi thêm. Đây là thuận lợi R1 so SCAFFOLD.

**Phép kiểm:** baseline DML-m4b dense và FedDyn-proxy-m4b, cùng một lần trao đổi/round, cùng số batch và lịch LR. Đề nghị alpha=0,01 cho screening; không gọi giá trị này tối ưu. Local objective DML có teacher/private thay đổi, solver AdamW hữu hạn: **variant dự án, không thừa hưởng trực tiếp định lý FedDyn**. Nếu proxy không tốt lên, không cần mở thêm private KD. FedAvg-m4b/CE-only proxy có thể làm diagnostic nếu cần tách tác động mutual learning.

**Cổng:** private gp ≥ baseline+0,010, benign ≥baseline−0,005; norm `a_i,h`, proxy finite; proxy gain riêng không đủ. Nếu chỉ giảm drift norm mà không tăng private, dừng. FedDyn có nguy cơ history khuếch đại sai số như bài học EF; checkpoint/resume phải chứa history và không cập nhật hai lần khi ACK lặp.

**q4 là bước sau:** phải dùng update đã decode nhất quán giữa lịch sử local và server; original FedDyn có thể lệch nếu local dùng delta FP32 còn server nhận delta q4. Trước phối hợp, đặc tả trạng thái commit/rollback, kiểm đại số cập nhật và lossless limit; sau đó kiểm noninferiority. Không hứa giữ utility từ dense sang q4. Không ghép đồng thời FedDyn + async + KD mới trong lần đầu.

Mức ưu tiên: **sau KD cap/teacher và Gaussian**, cùng nhóm sửa drift với SCAFFOLD. FedDyn dễ giữ optimizer đối chứng hơn nhưng objective DML không tĩnh; SCAFFOLD cần matched optimizer control. Chọn theo khả năng cô lập cơ chế và utility, chưa theo số byte/contact.

### 5.2. Proxy buffered aggregation — để giai đoạn B, sau kiểm chất lượng

Phần dưới giữ thiết kế tham khảo cho kịch bản hệ thống tương lai; **không thuộc batch ưu tiên cải thiện huấn luyện hiện tại**. Một contact/client/round chỉ là một scenario có thể kiểm sau, không yêu cầu owner vừa chốt.

**Lý do chọn:** cohort20 đồng bộ có thể đợi client chậm, trong khi mỗi xe gặp RSU ngắn và khác thời điểm. Một server có thể nhận nhiều xe khác nhau dù từng xe chỉ có một contact/round. Đây **không tương đương sub4** của từng client.

[FedBuff, AISTATS 2022](https://proceedings.mlr.press/v151/nguyen22b/nguyen22b.pdf) tổng hợp khi buffer đủ K update và phân tích staleness. Paper không chứng minh giao thức short-contact/q4 của dự án; dùng làm căn cứ thiết kế mới.

**Trạng thái client:** private, proxy trainable, snapshot teacher/base version, pending upload, pending download, ACK/transaction ID. Trong lần contact duy nhất, upload kết quả đã train trước đó và nhận snapshot **hiện có**; không giả định RSU kịp chờ cohort rồi trả global mới ngay trong contact ấy. Tính toán diễn ra ngoài contact, vẫn bị ràng buộc throughput thiết bị.

**Trạng thái RSU:** proxy global, buffer cập nhật, danh sách transaction đã commit, phiên bản, metadata cohort. Một client tối đa một update/cohort. Nếu đã upload nhưng chưa ACK, retry cùng transaction không tạo contribution mới. Chỉ aggregate các payload hoàn chỉnh; private luôn dùng snapshot đầy đủ mới nhất của chính nó.

Candidate đầu: cohort K=5 hoặc20 như **hai kịch bản định trước**, không chọn K sau nhìn test. Update stale mang base version; buffer dùng delta theo base đúng. Dạng server adaptation:

```text
Delta_i = decoded_proxy_after_local - decoded_proxy_base_i
tau_i = server_version - base_version_i
w_next = w_current + eta_server * mean_{i in cohort}(s(tau_i) * Delta_i)
```

Kiểm `s=1` trước; sau đó mới phép so riêng `s(tau)=(1+tau)^(-1/2)` lấy ý tưởng staleness scaling trong FedBuff. Giữ denominator K để attenuation stale không bị triệt khi mọi update đều cũ. Tất cả rule clipping/TTL/eta phải khóa trước chạy; không thêm momentum β0,9 đã phân kỳ.

**Ràng buộc sum-only:** RSU không được xem từng delta rõ chỉ để tính weight. Trong mô phỏng trusted sum, log đúng boundary; nếu staleness weighting cần thông tin client riêng, dùng metadata công khai và client-side preweight theo cohort contract hoặc cohort version cố định. Chưa có cơ chế ấy thì chỉ kiểm unweighted cohort sum. Buffer≥5 không tự cung cấp secure aggregation mật mã; dropout/setup/recovery phải tính phí.

**Codec q4 đồng bộ hiện tại không thể bê nguyên sang async:** client có base khác nhau. Chọn snapshot đầy đủ hoặc delta từ base mà client xác nhận; báo resync bytes. Scale int4 riêng từng client không tự cộng trực tiếp trong secure integer sum; cần common-scale codec hoặc boundary mô phỏng được công bố. Đo lại compression/utility sau thay đổi.

Controls: đồng bộ cohort20; buffered cùng contact trace; cache/no-peer. Mỗi xe vẫn cùng trần một contact/round và cùng real-data steps. Chấm tại thời điểm mô phỏng cố định, ghi số update/kiến thức thực nhận, staleness, fairness theo tần suất contact. **Dừng:** server update nhiều hơn nhưng private kém, xe ít contact chịu suy giảm, hoặc overhead/resync làm vượt cap. Thành công hệ thống có thể là cùng F1 với ít thời gian chờ; không bắt buộc tăng F1 khi không có giới hạn thời gian.

### 5.3. SCAFFOLD-proxy: thêm thông tin hướng cập nhật để giảm drift

**SCAFFOLD-proxy.** [Paper gốc, Eq.3–5](https://proceedings.mlr.press/v119/karimireddy20a/karimireddy20a.pdf) sửa gradient bằng `g_i+c−c_i`. Đây là candidate huấn luyện độc lập, **chưa loại bởi payload lớn**: kiểm liệu hướng cập nhật chung hữu ích hơn quadratic tether của FedProx. Private vẫn tự do kiến trúc; không áp control cùng shape vào private khác nhau.

```text
w_i <- w_i - eta * (g_i(w_i) + c - c_i)
c_i_new = c_i - c + (w_base - w_i)/(K_steps*eta)  # Option II, SGD constant eta
```

Phải thêm `c` down và `delta_c_i` up. FP32 hai vector 395.024 phần tử thêm **3.160.192 B/lượt hai chiều**; giai đoạn A cho truyền đủ và ghi lại, nén control là giai đoạn B. Option II không áp thẳng AdamW hiện tại; muốn giữ nguyên cập nhật proxy dùng SGD và matched SGD control. Không quy thay optimizer là lợi SCAFFOLD.

Server cập nhật `w_next = w_base + eta_server * mean_i(delta_w_i)` và `c_next = c + (1/N)*sum_{i in S}(delta_c_i)` theo cohort; client giữ `c_i` qua round, khởi tạo0. Không reset `c_i` cùng optimizer. BN buffers không nằm trong control vector; private vẫn dùng optimizer/loss đối chứng. Đếm **bước thực đã áp dụng**, không lấy batch bị AMP skip vào `K_steps` rồi gọi công thức chính xác. Trong fixture xác minh công thức dùng FP32/SGD không momentum, không adaptive preconditioner; gradient clipping và DML objective không tĩnh phải ghi rõ nếu production variant vẫn giữ.

Ablation: (i) DML-m4b AdamW baseline; (ii) DML-m4b với proxy SGD, control `c=0`; (iii) cùng (ii) + SCAFFOLD. Giữ private optimizer và lịch dữ liệu. Phép so (iii)−(ii) cô lập correction, còn (iii)−(i) quyết định có hơn baseline thực dụng không. LR proxy SGD cần cấu hình riêng khóa bằng calibration train/internal split, không bê LR AdamW rồi kết luận cả họ FAIL. Nếu chưa đủ ngân sách khóa LR công bằng, ghi chưa đủ điều kiện so sánh, không gán thất bại cho correction.

Gate như FedDyn: private tăng ≥0,010, benign guard; log norm/cosine của gradient local và correction. Chỉ proxy tốt hơn hoặc drift nhỏ hơn chưa đủ. Chi phí vector lớn được báo, **không chặn phép kiểm utility giai đoạn A**.

### 5.4. FedBN: probe chẩn đoán phụ

**FedBN-proxy.** [FedBN](https://arxiv.org/abs/2102.07623), [code tác giả](https://github.com/med-air/FedBN/blob/master/federated/fed_digits.py) giữ cả BN affine và running state local. Private đã có BN riêng: áp lại ở private là no-op. Chỉ probe proxy current vs local full-BN; không đổi normalization cùng lúc. Teacher proxy tại xe phải dùng BN xác định; khi mỗi xe có BN riêng, không còn một proxy global duy nhất để chấm nếu chưa định nghĩa calibration.

Rủi ro: BN local có thể giữ receiver shortcut, tăng local accuracy nhưng giảm private global. Vì vậy xếp thấp hơn m4b, chỉ mở khi log BN/feature shift cho thấy vấn đề còn tồn tại. Không đổi mục tiêu A sang local để gọi PASS.

## 6. Những paper không nên bê nguyên vào bài toán

| Nguồn gốc | Điều có thể học | Vì sao không chọn nguyên bản |
|---|---|---|
| [FedGen, ICML 2021](https://proceedings.mlr.press/v139/zhu21b/zhu21b.pdf) | Sinh tri thức để regularize client | Generator latent/prediction rules cần giao diện tương thích; không bảo đảm latent tùy ý align. Adaptation Gaussian-logit cũ đã yếu; ưu tiên hard-label covariance trước |
| [FedMD](https://arxiv.org/pdf/1910.03581) | Consensus logits cho model khác kiến trúc | Cần public data; dự án không có giả định này. Thay bằng Gaussian phải kiểm support và teacher từ đầu |
| [FedDF, NeurIPS 2020](https://proceedings.neurips.cc/paper_files/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Paper.pdf) | Distill ensemble model trên unlabeled/generated inputs | Cần models và distillation support phía server; không tự đáp ứng sum-only và contact cap |
| [FedGKT, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/a1d4c20b182ad7137ab3606f0e3fc8a4-Paper.pdf) | Chuyển tri thức giữa model nhỏ/lớn | Gửi feature map từng mẫu tới server, trái contract hiện tại |
| [MixStyle, ICLR 2021](https://arxiv.org/abs/2104.02008) | Giảm phụ thuộc style/domain | Trộn mean/std feature map ảnh không tương đương tráo 66 feature tabular khác nghĩa; chỉ lấy động cơ chống nuisance |

Không paper nào trong danh sách chứng minh trực tiếp utility DAGSNet/VeReMi với cả R1 và R2. Các hướng đề xuất là adaptation/kiểm nghiệm, chưa căn cứ tuyên bố novelty chỉ từ việc ghép thành phần.

## 7. Thiết kế kiểm nghiệm và cổng quyết định

### 7.1. Dữ liệu, metric, horizon

1. Dùng full-data internal holdout của FullDrift: cùng split/scaler/manifest, group và old mỗi65.536 dòng; không dùng pilot2,5% thay kiểm full dynamics.
2. Screening seed42 ep6, T50 giữ nguyên; xác nhận seed43 bằng **đối chứng cùng seed**. Gaussian-m4b seed43 chưa có Gaussian không-mask cùng seed trong vòng7: không dùng so s42 để nhận paired PASS.
3. Candidate thắng mới kiểm ep15 và cuối cùng full50 protocol công bố. Không lấy best epoch. So cùng horizon; lịch LR theo tiến độ local-data/epoch, không tăng tốc cosine khi server buffer update nhiều hơn.
4. Official test chỉ báo cáo ở checkpoint đã khóa. Không dùng số centralized 0,8532 như trần bất khả vượt: checkpoint ấy từng được chọn theo test, không phải upper bound sạch. Giữ caveat trong [architecture](../../knowledge/architecture.md).
5. Scaler fit toàn train80% trước tách internal holdout: có preprocessing leakage vào internal probes; công bố, không refit âm thầm rồi so với lịch sử. Receiver có thể xuất hiện ở nhiều client; disjoint group phải theo key scenario/receiver đúng fixture, không coi client-disjoint là receiver-disjoint.

Primary `A = mean_i F1_macro(CM_i_group_gp)`. Không lấy F1 của confusion pooled thay mean-client. Báo đủ 10 metric, per-class, benign, private head/gp, proxy phụ; worst-client và p10. Góc local dùng head và mô tả rõ prior-reweighted confusion chỉ mô phỏng label shift, không phải holdout receiver thực của từng client.

**Gate đề nghị cho cơ chế tăng chất lượng:** ΔA≥+0,010, benign≥−0,005, worst-client global macro-F1 không giảm quá0,010; mỗi seed xác nhận riêng. **Gate codec:** ΔA≥−0,010, benign≥−0,010, byte giảm thật. Báo head/gp cùng lúc; giữ gp lịch sử cho gate, không đổi prior hậu nghiệm để cứu một arm. Nếu thay tỷ lệ loss synthetic/KD có thể làm gp sai prior hiệu dụng, mở thiết kế calibration riêng với split riêng và headline khóa trước.

Hai seed là kiểm lặp, không đủ tuyên bố significance rộng. Nếu cần CI, bootstrap theo nhóm receiver/session, không coi hàng triệu dòng hoặc 20 model cùng dữ liệu là mẫu độc lập.

### 7.2. Tách lợi FL khỏi lợi recipe

| Câu hỏi | Control cần có |
|---|---|
| m4b/EMA/consistency có chỉ regularize local? | local/no-peer và own cùng recipe |
| Teacher quần thể có thêm tri thức? | pooled teacher vs own teacher, cùng freeze/refresh/compute |
| Gaussian có chở quan hệ ngoài prior? | pooled covariance vs own và replay local cân bằng lớp |
| Một loss mới có thật giúp private? | cùng m4b, cùng optimizer, cùng input exposure và lịch tổng hợp logic |
| Nhiều compute có gây lợi giả? | báo equal-data và equal-compute; walltime/FLOPs forward/backward riêng |
| Async tăng tốc hay đổi cohort bias? | cùng trace, participation, cap và thời điểm chấm; log exposure theo xe |

### 7.3. R1 để giai đoạn B: số contact, hai chiều và cold start

**Không phải điều kiện phải giải quyết trước kiểm thuật toán ở giai đoạn A.** Bảng dưới chỉ dự toán thông tin cần truyền; không dùng để hạ ưu tiên SCAFFOLD/Hybrid khi chưa đo utility. Scenario một contact/round là stress tham khảo; số contact chính thức sẽ bổ sung sau theo yêu cầu owner.

Chưa có bộ tham số RSU cuối cùng. Giữ B0≈4.500 B, B2≈231.000 B từ [protocol](protocol.md) làm **mốc stress tham chiếu**; không tuyên bố owner đã chọn chúng làm cap triển khai mới. Phân tích theo tốc độ/phủ sóng ở [contact-budget](contact-budget-2026-10-16.md) là sensitivity giả định, chưa thay thế trace cohort20.

Trong mỗi scenario phải ghi rõ cap là tổng up+down hay riêng từng chiều. Mặc định bảo thủ của bảng dưới: **tổng hai chiều/contact**, chưa overhead. Không cộng tốc độ PHY thành application throughput đo được.

| Payload | Logical byte tối thiểu/client | B0: lower bound contact | B2: lower bound contact |
|---|---:|---:|---:|
| Proxy q4 một lượt up+down | 416.520 | 93 | 2 |
| Gaussian53 FP32 bootstrap up+down | 190.208 | 43 | 1 |
| Gaussian53 FP64 bootstrap up+down | 380.160 | 85 | 2 |
| Proxy dense một lượt up+down | 3.186.880 | 709 | 14 |

Lower bound `ceil(total/cap)` chỉ từ byte, giả định cho gộp phần dư hai chiều. Causal ordering, đợi cohort và overhead có thể tăng số contact; bảng không phải cam kết số round đủ. Một contact/round ⇒ payload chưa đủ có thể trải qua **nhiều round**, không nhiều contact cùng round. Nếu không cho carry-over fragments thì payload vượt cap phải bị loại, không miễn phí nâng cap.

Mỗi trace ghi `available/successful contacts`, bytes attempted/ACKed/wasted, bootstrap bytes, cohort wait, snapshot age, upload age, số commit và giờ mô phỏng. Client không contact vẫn local; không nhận model mới miễn phí qua object trong RAM server. Với Gaussian tĩnh, báo cả cold-start và sau bootstrap; không chỉ báo steady-state 0 byte rồi bỏ phí khởi tạo. Với nhiều RSU, phải mô tả đồng bộ backhaul/global state; chưa mô phỏng thì giới hạn kết luận một RSU logic.

### 7.4. R2: kiểm thật sau sàng lọc DAGSNet

Thiết kế giai đoạn sau: private DAGSNet, MLP và CNN1D, cùng66feature/16class semantics; proxy DAGSNet chung nếu dùng DML/FedDyn/FedBuff. Chọn architecture assignment theo seed độc lập nhãn/client size, rồi xoay assignment để tránh model yếu rơi đúng client khó. Chưa thay model trong công việc tài liệu hiện tại.

So local-only/no-peer, Gaussian và candidate thắng **trên cùng architecture assignment**. Báo gain từng kiến trúc và worst client, không chỉ trung bình cohort. Private không tải weights/head/BN/projector chung bắt buộc; chỉ KD logits local hoặc synthetic input66. Chi phí proxy cộng private vẫn phải vừa thiết bị; private nhỏ không tự làm toàn hệ thống nhẹ.

PASS R2 khi giao thức chạy mixed cohort, không cần shape alignment của private, và lợi FL giữ theo từng nhóm đã khóa. Không gọi một cohort DAGSNet có seed khác nhau là heterogeneous architecture. Probe MLP lịch sử của Gaussian không thay phép kiểm mixed cohort này.

## 8. Thứ tự thực hiện đề nghị

| Thứ tự | Việc kiểm | Quyết định sau kiểm |
|---|---|---|
| 0 | Khi có tác vụ collect được yêu cầu: audit seed43 vòng6–7; khóa baseline và payload semantics theo artifact | Không dùng trạng thái remote để chốt thắng; chưa cần contact simulator |
| 1 | DML-m4b: cap KD rồi frozen teacher. Gaussian-m4b + EMA | Utility dense/full-information, control local/own để chứng minh FL |
| 2 | FedDyn-proxy; SCAFFOLD-proxy với matched SGD control; consistency là nhánh local phụ | Mỗi lần một cơ chế; thắng seed42 rồi mới seed43/horizon |
| 3 | Hybrid private-only synthetic khi DML/Gaussian riêng đã được phân tích | Phải hơn thành phần mạnh nhất, không chỉ hơn local |
| 4 | Mixed architectures cho candidate thắng, vẫn truyền đủ thông tin | Xác nhận private independence và gain theo kiến trúc |
| 5 | Giai đoạn B: marginal53/codec q4 thật, cold start, contact scenarios và buffered aggregation | Đo phần utility giữ lại khi áp R1; số contact bổ sung sau |

Không mở toàn bộ ma trận cùng lúc. Screening full ep6 DML cũ khoảng205–217 phút/arm trên cấu hình Kaggle đã đo; **không** dùng con số ấy làm dự toán chắc chắn cho frozen teacher/FedDyn/consistency. Profile smoke và thời gian một epoch trước khi đề nghị quota. Không query account, launch, sửa training code, commit/push hoặc backup trong nhiệm vụ này.

**Ưu tiên nếu chỉ chọn hai nhánh:** DML-m4b với KD có giới hạn/frozen teacher; Gaussian-m4b với EMA. **Họ thuật toán mới ưu tiên:** FedDyn-proxy và SCAFFOLD-proxy để kiểm thông tin/hướng cập nhật global. Buffered aggregation, compression và số contact kiểm sau; FedBN/generator là nhánh dự phòng có điều kiện.

## 9. Điều cần chốt trước triển khai, chưa cần thao tác tay hiện tại

- Giai đoạn B mới cần ngân sách tổng up+down/contact, deadline, giữ mảnh qua round, scenario RSU và slowdown thiết bị. Những điểm này **không chặn thiết kế/kiểm chất lượng huấn luyện giai đoạn A**.
- Mức private F1/benign/worst-client mong muốn nếu owner muốn gate tuyệt đối; hiện tài liệu chỉ đề nghị gate cải tiến tương đối.
- Có cho mở stage mixed architectures khi DAGSNet PASS; kiến trúc và giới hạn compute cụ thể.
- Cổng/cấu hình từng build và quota trước launch. Phản hồi mới nhất đã chốt **chất lượng huấn luyện trước, contact sau**; chưa phải ủy quyền chạy.

Nguồn paper được dẫn ngay tại cơ chế tương ứng; nguồn số dự án trỏ reports có scope. Kết quả mới phải cập nhật dự báo bằng PASS/FAIL có phạm vi, giữ lại lý do bị bác để tránh lặp vòng thử cũ.

## 10. References — bài báo gốc

1. Shen et al. **Federated Mutual Learning**. 2020. [arXiv:2006.16765](https://arxiv.org/abs/2006.16765). Tiền lệ private độc lập + model chung học tương hỗ.
2. Acar et al. **Federated Learning Based on Dynamic Regularization**. ICLR 2021. [Paper](https://arxiv.org/pdf/2111.04263). FedDyn, local history và server correction.
3. Karimireddy et al. **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning**. ICML 2020. [PMLR](https://proceedings.mlr.press/v119/karimireddy20a.html). Control variates sửa client drift.
4. Nguyen et al. **Federated Learning with Buffered Asynchronous Aggregation**. AISTATS 2022. [PMLR](https://proceedings.mlr.press/v151/nguyen22b.html). Nền tảng nhánh hệ thống giai đoạn B.
5. Li et al. **FedBN: Federated Learning on Non-IID Features via Local Batch Normalization**. ICLR 2021. [Paper](https://arxiv.org/abs/2102.07623). Probe local BN, không bằng chứng tự động cho R2.
6. Ross, Hughes, Doshi-Velez. **Right for the Right Reasons: Training Differentiable Models by Constraining their Explanations**. IJCAI 2017. [Proceedings](https://www.ijcai.org/proceedings/2017/371). Regularize phụ thuộc input; không chứng minh receiver-swap giữ nhãn.
7. **R-Drop: Regularized Dropout for Neural Networks**. NeurIPS 2021. [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/file/5a66b9200f29ac3fa0ae244cc2a51b39-Paper.pdf). Consistency hai forward; adaptation IDS dùng transform khác.
8. Zhu, Hong, Zhou. **Data-Free Knowledge Distillation for Heterogeneous Federated Learning**. ICML 2021. [PMLR](https://proceedings.mlr.press/v139/zhu21b.html). FedGen; tham khảo và giới hạn ở §6.
9. **FedMD: Heterogenous Federated Learning via Model Distillation**. 2019. [Paper](https://arxiv.org/pdf/1910.03581). Cần public data, không áp nguyên bản.
10. **Ensemble Distillation for Robust Model Fusion in Federated Learning**. NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/18df51b97ccd68128e994804f3eccc87-Paper.pdf). FedDF; kiểm giả định server/data trước áp dụng.
11. **Group Knowledge Transfer: Federated Learning of Large CNNs at the Edge**. NeurIPS 2020. [Paper](https://proceedings.neurips.cc/paper/2020/file/a1d4c20b182ad7137ab3606f0e3fc8a4-Paper.pdf). FedGKT; loại truyền feature từng record theo contract.
12. **Domain Generalization with MixStyle**. ICLR 2021. [Paper](https://arxiv.org/abs/2104.02008). Động cơ chống domain cues; không bê phép trộn feature ảnh sang tabular IDS.
