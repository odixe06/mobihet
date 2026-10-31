# P3 — Chặn synthetic làm lệch BatchNorm private

**Trạng thái (17/10 ~17:10Z): backlog** — BN tính lại chỉ trên dòng thật làm gp −0,032/−0,043 ([kết quả probe](../../methods/full-drift/results.md)); không có tín hiệu. Trước đó: probe trước (kernel vòng 13): `calibrate()` tính lại BN trên 65.536 dòng thật của chính client
cho weights ep6 đã audit. Mở arm train nếu gp Δ ≥ +0,005 trên cả hai seed gpool; không thì backlog
([design §Vòng 13](../../methods/full-drift/design.md)). Đề xuất owner 17/10 (ưu tiên 3).

## Vấn đề (đã xác minh trong code, chưa xác minh tác hại)
Private forward nối 512 real + 256 Gaussian; DAGSNet có nhiều `BatchNorm1d` ([core/dagsnet.py](../../../core/dagsnet.py)) ⇒ batch
và running statistics nhận cả hai miền. CE real và CE synthetic lấy mean riêng rồi cộng trọng số 1 ⇒ synthetic chiếm 1/3 batch
nhưng 1/2 loss CE.

## Bản train tối thiểu (chỉ khi probe có tín hiệu)
Forward real trước, cập nhật BN bình thường; forward synthetic riêng với BN dùng running moments real nhưng vẫn cho gradient qua
affine/backbone. Chỉ đổi chế độ BN (không `model.eval()` toàn private làm tắt dropout); một optimizer step như baseline; không đổi
proxy BN. Controls: batch nối gốc; hai forward cùng cập nhật BN; hai forward chỉ cập nhật BN từ real. BN riêng cho synthetic là arm sau.

R1: 0 byte. R2: model không có BN không cần biến thể này.

## Tiền lệ
- Tính lại BN trên dữ liệu local cho model EMA: offline +0,011/+0,010, Kaggle +0,006/+0,006, lợi giảm dần
  ([kế thừa](../../archive/legacy/kien-thuc-ke-thua.md)); FullDrift `private_ema` (EMA + BN tính lại) gpool 0,677 < 0,717.
- Build12 N4/N5: không đóng băng BN khi chưa có đối chứng.
- AdvProp (CVPR 2020) tách BN cho hai miền ảnh; thay bằng Gaussian tabular là suy luận thiết kế. Khác FedBN (localize BN proxy).

## Dấu hiệu sai
Recalibration không giúp; full probe không qua cổng. Không viện dẫn BN là nguyên nhân duy nhất của plateau.
