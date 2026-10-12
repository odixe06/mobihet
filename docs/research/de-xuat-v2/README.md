# Đề xuất V2 — FL cho IDS V2X sau P1, P2 và S-a

Ngày 2026-10-11. Tài liệu thiết kế ban đầu. **Sau đó chủ dự án đã yêu cầu triển khai kiểm khả thi**;
[cấu hình/cổng U đã chốt](../pilot-v2.md), [ledger hiện tại](../../../papers/v2/rebuild.md).
Source đã triển khai và kiểm local; chưa có kết quả Kaggle V2. Những giá trị đề nghị bên dưới xem cùng manifest mới nhất.
Chủ dự án xác nhận trong phiên: **mọi mô hình học được đều DAGSNet**. Không đưa cây học được,
GMM học bằng EM, MLP phụ hoặc generator kiến trúc khác vào các phương án bên dưới.
Các moment Gaussian H2 giữ vai trò thống kê gộp đã được cho phép; không thêm mạng học được ngoài DAGSNet.

## Đọc theo thứ tự

| cần tìm | tài liệu |
|---|---|
| Bằng chứng cũ, yêu cầu, giao tiếp, đánh giá, cổng chung | [Hợp đồng thiết kế và kiểm nghiệm](thiet-ke-chung.md) |
| Giữ sức chứa proxy, truyền sketch lên và bản vá thưa xuống | [V2-A: proxy DAGSNet với sketch và bù sai số](a-proxy-sketch.md) |
| Truyền hệ số nhỏ, tái dựng toàn bộ proxy từ seed chung | [V2-B: DAGSNet trong không gian tham số sinh từ seed](b-proxy-seed.md) |
| Học phần Gaussian còn thiếu bằng DAGSNet, sinh mẫu mới liên tục | [V2-C: federated density-ratio weighting](c-density-ratio.md) |
| Paper gốc nói gì, điều gì chỉ là adaptation của dự án | [Đối chiếu nguồn nghiên cứu](nguon-nghien-cuu.md) |

## Ưu tiên và phạm vi đóng góp

| hướng | tri thức liên client | điểm mới so với các build | B0 / B2 | ưu tiên |
|---|---|---|---|---|
| **V2-A** | cập nhật proxy DAGSNet qua CountSketch tuyến tính, giải nén sau gộp | thay payload toàn mạng bằng cập nhật thưa có bù sai số; giải quyết cả upload, download và nợ phiên bản | B0 có rủi ro sketch quá nhỏ; B2 có dư địa hơn | **1: kiểm khả năng giữ chất lượng proxy** |
| **V2-C** | tham số DAGSNet học phân biệt dữ liệu local đã làm trơn với Gaussian, dùng làm trọng số auxiliary | học trực tiếp độ sai của mô hình sinh; không dùng teacher yếu đổi nhãn lớp hoặc bank cố định | bản dense chỉ là trần không kênh; muốn R1 phải ghép carrier đã qua cổng | **2: kiểm có tri thức mới ngoài H2** |
| **V2-B** | hệ số cộng được trong cơ sở tham số chung do seed tạo | giới hạn bậc tự do học được để gói trạng thái nhỏ; mọi tầng vẫn chịu cập nhật | phù hợp byte B0 về kích thước; nguy cơ sức chứa và tối ưu | **3: hướng rủi ro cao, cần cổng sức chứa trước** |

Thứ tự trên là **khuyến nghị nghiên cứu**, chưa phải lệnh mở pilot. V2-A/B thuộc hướng proxy FL;
không đổi tên chúng thành họ FL hoàn toàn mới. V2-C đổi đối tượng học từ phân loại/khớp moment sang
ước lượng tỷ số mật độ có làm trơn. Cả ba mới ở phạm vi thí nghiệm MobiHet, **chưa khẳng định novelty công bố**.

V2-A ưu tiên vì bằng chứng cũ cho thấy model chia sẻ đủ sức chứa mang được tri thức đa receiver.
V2-C đáng nghiên cứu nếu cần cải thiện chất lượng ngoài Gaussian; không tự giải quyết được byte.
V2-B đánh đổi sức chứa và compute lấy trạng thái nhỏ, không lấy kết quả fine-tune LLM làm dự báo DAGSNet từ đầu.

## Giữ lại và loại khỏi đợt này

- Giữ H2 diag/full, H1 DAGSNet, local-only làm mốc; chạy lại mốc phù hợp trong cùng driver khi thực nghiệm.
  P2 có hạn chế prior/codec, nên số cũ chỉ giải thích lựa chọn, không thay đối chứng mới.
- S1 Nyström và H1+H2 là backlog đã có, không trình bày như phát hiện mới của V2.
- Không mở lại KME/DM tĩnh bằng tăng bank/steps/round để cứu cổng đã trượt.
- Tree/histogram classifier, mixture học bằng EM, generator khác DAGSNet: loại theo câu trả lời mới nhất.
- JEM/SGLD, dynamic condensation và functional boosting chỉ là hướng tham khảo trong ghi chú nguồn nếu có;
  chưa ưu tiên do thêm giả định hoặc chi phí chưa được bằng chứng hiện tại hỗ trợ.
- Không dùng prototype/head-only làm phương án chính; không chia sẻ latent hoặc gradient từng record.

## Trạng thái bàn giao

Các file này phân biệt **đã đo**, **suy luận**, **thiết kế đề nghị**, **chưa biết**. Cổng và HP bên trong là
đề nghị để review trước triển khai, không phải quyết định đã được chủ dự án duyệt hay kết quả đã đạt.
R2 vẫn chỉ ở thiết kế: private model tương lai có thể khác kiến trúc, giao diện tri thức không cần cùng tham số;
phiên hiện tại mọi model học được đều DAGSNet.

Không có thao tác tay bắt buộc cho đợt viết tài liệu. Khi yêu cầu triển khai sau này: chọn một hướng,
chốt cổng/HP còn mở, cập nhật quota lúc chạy; notebook mới phải W&B online trước train.
Giới hạn local chỉ kiểm tra nhỏ; không tự Git/DVC hoặc push.

Kiểm tài liệu ngày2026-10-11: link nội bộ các file mới/cập nhật resolve; phép tính payload, frame và bộ nhớ
trong A/B/C đã đối chiếu bằng số học; [index toàn dự án](../../file-index.md) khớp danh mục workspace không bị ignore.
Đây là kiểm tài liệu, không phải smoke thuật toán hoặc bằng chứng hiệu quả các phương án.

Nguồn trạng thái cũ: [bàn giao pilot](../../operations/handoff-2026-10-11.md),
[P1](../probes/pilot-p1.md), [P2](../probes/pilot-p2.md), [S-a](../probes/pilot-new-fl.md).
