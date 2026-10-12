# Quyết định chủ dự án — 2026-10-12

## Tổ chức dự án

Mỗi phương pháp có thư mục `proposals/<tên-phương-pháp>/`: giải thích, kết quả, bài học, đường dẫn code/artifact.
Giữ đường dẫn source/notebook/manifest đã chạy để bảo toàn provenance/hash/DVC. Index toàn dự án bắt buộc.

## Collect và khôi phục

Thu output kernel đã hoàn tất còn thiếu. Được pull DVC chọn lọc khi cần; không Git/DVC add/push, không cleanup mới.
Artifact đã backup nhưng xóa local không đồng nghĩa chưa từng collect. Phân biệt khôi phục proof cũ với audit mới.

## K5 geometry primitive moments

- Kiểm cả G51 và G53; hoãn session/rate. Tên tạm, báo số chiều thực tế khi giữ thêm primitive.
- Thử cả bốn nhóm mask marginal và moment theo joint32 = first×receiverprofile4×senderprofile4; 13 atom thống kê có điều kiện từng ô. Không joint131.088.
- Ô có ≥5client và đủ mẫu cho covariance mới phát moment; ô hiếm fallback moment lớp gộp + mask ô, báo byte/tỷ lệ fallback, không bỏ record.
- Moment theo joint mask để đo tác động đến huấn luyện.
- Nhánh missing/nonfinite không xác định từ 66 feature đã impute: giữ feature suy diễn bị ảnh hưởng làm primitive.
- Hai cổng khóa riêng: quality ≥+0,015 so full66; carrier giữ F1 trong −0,005 và byte ≤65% control.
- Fresh full66-plain/full66-support, own, shuffle; cùng BN/sampler/compute.
- Toy kiểm missing/coincident trước pilot; fidelity không qua ⇒ dừng, chưa train.
- Mọi learned model DAGSNet; input 66; raw/perturbed record không rời xe.

Chủ dự án yêu cầu rút bài học, tổng hợp tài liệu, tiếp tục đề xuất/cải tiến và kiểm nghiệm.
Collect/audit xong trước dùng headline mới làm bằng chứng. K5 chưa triển khai/push ở thời điểm ghi quyết định.

## K5 saufidelitytoy

NominalG51/G53 bị bác bỏ bởi missingtoy actualextractor; giữ8derived⇒cùngG59.
Chủ dự án chọn kiểmG59 quality với marginal/joint32; khôngchạyhaiarmtrùng.
Full59covbyte≈80%full66trướcmask, khôngđạt65%; ghi carrierFAIL trước train, khônghạcổng.

Sau cảnhbáo bearing divisionnear0, chốt **cảG59/G61**. G61giữbearings2; G59dùnginverseđúngextractor,
invalidsupport diagnostic/nonfiniteFAIL. Cổngcarrierkhônghạ. V6pilot cầnlocalproof vàmanifest trướcpush.
