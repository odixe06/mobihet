# Cơ chế đang chẩn đoán

Mục tiêu: từng xe học tốt trên own-test, nhất là ít dữ liệu/non-IID; giữ schema trao đổi chung cho model khác nhau.
Đây là họ đối chứng, chưa chọn phương pháp tổng. Cấu hình chung tại [protocol](protocol.md).

## Luồng thông tin

1. Bootstrap thống kê gộp theo lớp để dựng Gaussian copula/prior; không truyền mẫu local.
2. Client học model riêng bằng CE trên dữ liệu thật và64synthetic/bước, sinh tại client.
3. Client gửi mean probability theo true class (16×16); server gộp theo n_kc hoặc sqrt(n_kc).
4. Server gửi Q chung cho mọi xe. Q giữ cố định trong round, cập nhật cho round kế tiếp.
5. Client distill từ Q trên synthetic, tùy đối chứng thêm realKD có gate. CEreal có logit adjustment;
   KD dùng logit chưa adjustment. Không tự gọi hai hệ xác suất khác nhau là bug.

## Các khâu M kiểm riêng

- Upload: xác suất online so với full real pass sau epoch, eval BN/dropoutoff.
- Server: n_kc so sqrt(n_kc), cùng packet/schema.
- Client: augmentation, softtarget, realgate, fallback, uniform/local support mixture; loss outer coefficient1.
  Localρ=64/(64+n_kc), lớp ít mẫu nhận Q nhiều hơn. Đây là giả thuyết chẩn đoán, không bảo đảm có ích.
- Chi tiết thiết kế đã khóa: [screening-m20](screening-m20.md); kết quả ở [results-m20](results-m20.md).

## Chi phí và giới hạn

Q codec meanfp16:544B/chiều,1088B/xe/round; copula bootstrap3311744B/xe hai chiều.
20round:3333504B/xe; augmentation3311744B và0learnedB/round. Bootstrap vẫn chi phối chi phí.
Số byte chưa gồm transport/ACK/secureaggregation end-to-end. Aggregate statistics chưa có DP guarantee.
Client model khác kiến trúc và contactdrop chưa được kiểm; batch hiện cùng DAGSNetW1.
Snapshot nhỏ không chứa fullprivatebank, không dùng làm checkpoint resume.
