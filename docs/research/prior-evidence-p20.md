# Đối chiếu bằng chứng trước P20 — 01/11/2026

Owner yêu cầu tra v1 và repo features; chỉ đọc nguồn, không sửa hai repo đó hoặc nạp archive local.
Provenance/SHA và audit dữ liệu: `runs/p-study/prior-evidence.json`, tái tạo `scripts/audit_prior_evidence.py`.

## Model và phạm vi so sánh

`../fl_dagsnet_veremi_v2`: `core/dagsnet.py`, `core/grouped.py`, `docs/research/protocol.md` byte-identical repo này.
Giữ DAGSNetW1/395408parameter/77input/16class, optimizer/batch/epoch/LR chung. P vẫn là screening20round,
không gọi là benchmark thắng baseline50round. Không chuyển client sang model khác; schema kiến trúc độc lập vẫn là yêu cầu phương pháp.

[Tham chiếu r20](benchmark-r20-reference.md) tính lại17CMview trên cùng test/support, đủ10metric;
giữ khác biệt global/client/prototype. Hai baseline có pruning/head bổ sung tách khỏi bảng đồng nhất parameter.
Không suy chênh phương pháp là causal effect riêng teacher hoặc dùng baseline50 so screening20.

## V1: teacher tốt chưa đủ và không mở lại kỹ thuật FAIL thiếu giả thuyết mới

Nguồn pinned [v1 HEAD5205990](https://github.com/odixe06/mobihet-archive-veremi-v1/tree/5205990d359201287b15aee7354f7bc1b4ef791b).
V1 dùng66input/20–100client/partition và đánh giá khác; số F1 không chuyển thành gain dự kiến cho v4/5343xe/own-test.

- [P23–28](https://github.com/odixe06/mobihet-archive-veremi-v1/blob/5205990d359201287b15aee7354f7bc1b4ef791b/docs/research/methods/synthetic-learning-recovery-2026-10-22.md):
  hiệu chỉnh teacher tăng teacherF1 nhưng transfer không giữ worst-client; teacher agreement trên synthetic tăng chưa đủ utility.
  Support thật là oracle vi phạm giả định FL, không triển khai hoặc gửi record trong P.
- [P36](https://github.com/odixe06/mobihet-archive-veremi-v1/blob/5205990d359201287b15aee7354f7bc1b4ef791b/docs/research/methods/teacher-support-two-seed-2026-10-22.md):
  teacher×copula qua frozen-head hai seed nhưng online chỉ gain~.0003. Không nhận cải tiến từ head-only/offline teacherfit.
- [P37](https://github.com/odixe06/mobihet-archive-veremi-v1/blob/5205990d359201287b15aee7354f7bc1b4ef791b/docs/research/methods/synthetic-weight-dynamics-2026-10-23.md):
  tách BN/head-only dưới production ở cả hai checkpoint; BN-only không gây utilityharm theo khóa.
  Gradient CE xoay mạnh khi trộn synthetic chưa chứng minh BN là nguyên nhân. Không mở lại BNsplit/recalibration thành ứng viên P.
- [P33](https://github.com/odixe06/mobihet-archive-veremi-v1/blob/5205990d359201287b15aee7354f7bc1b4ef791b/docs/research/methods/client-learning-diagnosis-2026-10-22.md):
  tri thức chỉ chuyển ở support có input phù hợp, quan hệ liên feature quan trọng. Mean lớp/scalar và tăng liều KD không tự thêm ranh giới.

Suy luận cho P: kiểm teacher phụ thuộc input và proxy server trên support được refresh mỗi round, với đối chứng class cùng support,
hard cùng support và streaming. Khác nhánh anchorJ cũ: không dùng một bank cố định cả run, fixedweight1, codec fp16 decode thực.
Đây là giả thuyết mới, không khẳng định neuralproxy vượt classQ hoặc lặp CCVR/BNsplit/GMM/MADE.

## V4: nhãn và cấu trúc feature

Repo `../mobihet-features` HEAD62c5a24; source local có thể chưa commit, fileSHA trong receipt mới là code đã đọc.
Feature report ở `../mobihet-features/docs/research/feature-processing-v4.md`;
nguồn thực là file local được fingerprint, không suy HEAD bao phủ các file uncommitted.

- Scaler fit originalfoldtrain,71feature+6flags. NaN→0, flags0/1, nhiều feature dùng chung flag.
  `src/v2/scale.py` định nghĩa mapping `nan_indicator`, không suy từ hậu tố tên feature.
- Audit419840row (first1024/trainfile, gồm originaltrain+val) có0nonbinaryflag và0vi phạm flag1→feature0;
  không fullcontentfingerprint hoặc full-data proof. Không đọc officialtest để fit rule.
- Copula hiện nội suy mọi cột, kể cả6flag, nên có thể sinh cờ liên tục và feature khác0 khi báo thiếu.
  P đo vi phạm rồi kiểm projection: thresholdflag.5, zero đúng feature theo mapping; không đổi scaler/input/model.
  Hardaugmentation×softQ đối chứng2×2 nhằm tách sửa support khỏi teacher.
- TimeDelay dời cảsend/receive và giữ độ trễ2–4s/phiên trong audit cũ: `rel_delay` không tự lộ attack.
  Không kết luận bất khả phân biệt trên77feature; feature report vẫn thấy tương tác/bối cảnh có ích.
- Feature selection v4 và val xe chưa thấy là pooledmodel, không phủ định shortcut khi từng model chỉ học một xe.
  Benign/worldcontext và Sybilprior shift là caveat; không đưa world/receiver/time/label vào77input.

## Những nút thắt cần phân biệt

1. Nội dung teacher: classQ mất inputdependence, nhưng offdiagonal có utility O; không bỏ cấu trúc trước bằng chứng mới.
2. Serverfit/carrier: ensemble không bảo đảm teacher mạnh; proxy có thể làm mất tri thức hoặc ngoại suy sai.
   Đo fitKL và teacher/student trên cùng diagnosticrows, kiểm cả logits carrier và fp16modelcarrier trên cùngbank.
3. Clientreception: gate theo lớp và scale; có controlstrength theo từng client, không tăng KD chỉ vì client ít data.
4. Support: missingness hợp lệ chưa đồng nghĩa jointmanifold đúng; projection chỉ kiểm một ràng buộc rõ ràng của v4.
5. Localobjective/optimization: LA, lượng bước, train–testgap; realonly có/không LA là lossdiagnostic,
   giữ1epoch/resetAdamW. Chưa tăng bước hoặc thayprotocol riêng một method.

Không hứa tăng F1 mỗi round. Đọc own-test/mean/p10/xe<256/rare/absent/perclass và byte+time;
teacherfit/confidence hoặc pooledgain riêng lẻ chưa chốt phương pháp.
