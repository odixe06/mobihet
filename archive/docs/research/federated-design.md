# Thiết kế federated từ bài toán — 31/10/2026

Owner cho phép xây lại phương pháp từ kiến thức đã có. Hai yêu cầu: private model có thể khác kiến trúc,
giao tiếp hiệu quả. Chưa có trần byte; ưu tiên F1, báo đường đánh đổi chi phí; giữ DAGSNet chung trước khi kiểm kiến trúc trộn.
Không mặc định proxy, copula, hay một thuật toán KD phải được giữ nếu bằng chứng bác bỏ.

## Hợp đồng tối thiểu

Client cùng định nghĩa 16 nhãn và 77 feature đã scale; tham số, hidden dimension, optimizer private không cần giống nhau.
Không gộp weights/hidden prototype trực tiếp giữa kiến trúc tùy ý: khác không gian tọa độ/kích thước.
Chọn giao diện tri thức theo **đầu ra lớp**, hoặc thống kê input có schema chung; private inference không cần nhãn/router/server.
Thống kê input không phải đảm bảo privacy: tổng hợp và nén không tự cung cấp DP/secure aggregation.

Mục tiêu thực nghiệm: private own-test F1, phân phối client (đặc biệt p10/ít dữ liệu), chi phí train,
bootstrap và packet hai chiều. Đánh đổi F1/byte phải dựa trên artifact, không chỉ kích thước một packet.
Bài toán hiện chưa mô hình hóa mất gói, client di động đến/rời hoặc contact deadline; không tuyên bố hiệu quả trong các điều kiện đó.

## Dữ kiện dẫn thiết kế

- Proxy DML đắt; proxy học ở server/giá rẻ ở xe đều FAIL screening. Bỏ proxy là lựa chọn có bằng chứng tốt hơn trong phạm vi đã thử.
- Chỉ Q[y] rất nhỏ nhưng làm mất cấu trúc trong lớp. Bank synthetic + q(x) bảo toàn cấu trúc nhưng J1 chỉ hơn đối chứng cùng bank0,00212,
  chưa bù mất phủ support; chi phí131.136B/round thêm chưa mua được gain. Không tiếp tục chỉ vì entropy/teacheragreement đẹp.
- J4 bỏ realKD tăng timeDelayF1~0,2148 nhưng giảm cả15 lớp còn lại; meanΔ ở xe<256 mẫu=-0,00837.
  Không thể coi pooled+0,00363 là mọi client được lợi. Cần KD có chọn lọc bằng nhãn train hoặc teacher ít gộp mất thông tin hơn.
- Synthetic vẫn hữu ích so off trong S/G; tăng liều64 giúp p10. Chưa biết phần lợi đến từ augmentation nhãn hay tri thức FL.
  Vì vậy phải có control CEreal+CEsynthetic, không nhận learned teacher, vẫn dùng thống kê/prior toàn mạng.
- Copula bootstrap3.311.744B/xe lớn hơn nhiều1.088B/round; Gaussian394.368B giảm bootstrap nhưng utility thấp hơn ở hai seed.
  Chưa có cơ sở gọi copula tối ưu tổng thể, hoặc giảm bit là miễn phí.

## Cấu trúc ứng viên để kiểm chứng

1. Client train private bằng CE+LA trên dữ liệu thật; thêm support tổng hợp từ thống kê input toàn mạng.
2. Trích tri thức **từ forward dòng thật đã có**, không train proxy/không cần forward toàn bank chia sẻ sau epoch.
3. Gửi mean probability có điều kiện theo lớp và một partition input chung: Q[y,b(x)], kèm count bootstrap.
   Server gộp theo count, trả bảng nhỏ; cùng bảng dùng được cho bất kỳ private classifier16 đầu ra.
4. Synthetic KD dùng Q[y,b(x)] để truyền một phần quyết định theo input. Adaptive weight dùng marginalQ[y] suy từ cùng bảng,
   tránh vô tình trộn cải tiến teacher với thay độ mạnh loss. Router3sign projection cố định không train, B8 là lựa chọn khóa trước.
5. Tách riêng can thiệp realKD: giữ, bỏ toàn bộ, hoặc chỉ nhận khi teacher argmax khớp nhãn train; không đổi HP theo test.

Đây là **giả thuyết**, chưa là phương pháp thắng. Router có thể chia sai chiều phân biệt; bảng từ train-mode predictions có thể khác
post-epoch eval; thống kê synthetic vẫn có thể ngoài manifold. Cổng utility, per-class và seed phải bác bỏ được cơ chế.
Gửi meanfp16 thay sum để tránh overflow; decodedwire phải thật sự dùng trong train, không tính byteint8 nhưng họcfloat32 chưa nén.

## Kiểm chứng và thứ tự quyết định

[Batch K](screening-k10.md): factorial liều×realKD hai seed, augmentation-only hai seed, conditional teacher hai seed,
label-agreement gate hai seed. Tái dùng các cell đã audit thay chạy trùng.
Không gọi hai seed là chứng minh ý nghĩa thống kê, không gọi best-r10 là final50.
Nếu K không qua cổng, ghi FAIL đúng phạm vi; không nới cổng hoặc quét router/bin/temperature trên test để tạo thắng.
Nếu có ứng viên đạt, bước sau mới chốt50round, lặp seed rộng hơn, kiểm kiến trúc trộn và contact/sparsity.
Các bước này chưa được xác nhận bởi thí nghiệm10round đồng kiến trúc.

Giảm bootstrap (diagonal/low-rank/quantized input statistics) là trục chi phí độc lập; chỉ mở khi cần so đường đánh đổi,
đối chứng cùng teacher để tránh gán thay đổi utility cho riêng codec. Không thêm generator neural trước khi có bằng chứng cần thiết.
Nguồn và phân biệt adaptation với bài gốc: [synthetic-KD](synthetic-kd-next.md), [validation design](validation-k-design.md).

## K đã kiểm chứng — 31/10

[Kresults](results-k10.md): conditionalQ8bucketkhôngthắngcombo. GatedKDvàaugmentation-only làhaihướngcòncơsở;
khôngcoi proposedconditionalframeworklàmethodđãthắng. Gateưu tiênpooled/benign,augmentationưu tiêntail/ítmẫu vàkhôngneedlearnedmessage saubootstrap.
Chưa50round/mixedarchitecture/contactevaluation. Đây làkếtluậnkhảthi cógiớihạn,khôngphươngpháptốiưutoàncục.

## L đã kiểm chứng — 31/10

[Lresults](results-l10.md) táchtarget/scale/realgate vớicodecmeanfp16thật vàcontrolcùngcodec.
SoftfixednoReal vàsoftfixedgate đềuPASSgain ở42/43; adaptivegiảmF1 sofixed trongcáccellđãkiểm.
NhưvậyclassQ vẫn cógiátrị, khôngloại kênhteacher vìKaugmentationthắngcombo cóconfoundscale.
Ưu tiênsofttarget+fixed1 làmnền, giữaugmentation/noRealsoftfixed làcontrol; realgateđónggóppoolednhỏcầnxácnhận.
Class13teacher sai vẫnđượcdùngtrênsynthetic; kiểmfallback/postepochteacher theođốichứng, khôngthêm dunglượngpacket tựđộng.
Chi phí1088B/round+3311744Bbootstrap, full10r3322624B; kiếntrúctrộn/contactdrop/50roundvẫnchưađo.
