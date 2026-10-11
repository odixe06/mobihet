# Pilot S — hai kênh tri thức mới

Khóa trước chạy 2026-10-11. [Đề xuất + nguồn](de-xuat-bo-sung-2026-10-11.md).
User xác nhận: mọi private/encoder dùng DAGSNet; R2 chỉ ở thiết kế; local chỉ kiểm tra nhỏ.
Ngoại lệ Q4 mới: RSU được đọc bộ mẫu **tổng hợp**, không đọc mẫu thật. Không dùng raw initialization.

## S-a: sàng khả năng học

Code `mh/new_methods.py`, `mh/new_fl.py`. Cùng fixture P1: 20 client, 2,5% train,
66 feature; 20 round × 1 epoch, batch512, cosine1e-3→1e-5, AdamW wd0 reset mỗi client/round,
clip1, AMPfp16. Không mở test20%. Hai seed42/43, đánh giá `old/group` r5/10/15/20.
Chỉ số chính `group gp`, F1 trung bình20private; báo cả head, 10metric, per-class.
Không giới hạn kênh ở S-a; **không** dùng kết quả này làm bằng chứng R1.

| arm | tri thức / đối chứng |
|---|---|
| local | local-only; histogram nhãn cho gp được trả byte |
| diag | H2 diag, thống kê fp32 lên / tham số fp16 xuống thực sự lượng tử hóa |
| kme_gauss | như kme nhưng target Fourier là kỳ vọng giải tích của Gaussian diag; đối chứng optimizer/finite bank |
| kme | cộng histogram + moment1/2 +128 feature Fourier theo lớp, tổng ≥5client; client tối ưu bank128điểm/lớp |
| dm_gauss | mỗi xe sinh4điểm/lớp từ Gaussian của chính xe bằng noise; gửi bộ fp16 tới RSU, chuyển tiếp19client còn lại |
| dm | cùng init/bộ/byte với dm_gauss; tối ưu bank khớp mean nonlinear embedding của DAGSNet cố định |

Mọi arm phụ dùng batch256, CE tổng hợp trọng số1, forward trộn batch local như P1.
Phiên bản triển khai là **adaptation**, không reproduce DP-MERF/FedDM, không có DP.

### Quyết định cài đặt khóa

- KME:64tần số Gaussian,64cos+64sin, seed1701; bốn bandwidth
  sqrt(66)×{0,25;0,5;1;2},16tần số/mức. Không fit bandwidth/HP bằng holdout/test.
- KME gửi cả input mean/variance để khởi tạo tổng hợp; khác H2 ở embedding phi tuyến và tối ưu phân phối.
  Bank128điểm/lớp: Adam lr0,05,200bước; loss MMD feature +0,1phạt mean/second moment chuẩn hóa.
  KME-gauss cùng optimizer, init, byte; chỉ thay target bằng E[phi(x)] giải tích của Gaussian diag.
- DM: encoder DAGSNet bằng khởi tạo seed của run, frozen/eval; match mean latent256chiều,
  chia residual cho std latent local (floor variance1e-4). Adam lr0,03,200bước,
  phạt0,01mean chuẩn hóa. Không cập nhật encoder; không truyền trọng số private.
  Đây là condensation tĩnh với một encoder; không phải thuật toán FedDM iterative/global-model nguyên bản.
- Init tất cả synthetic bằng noise độc lập; DM chỉ dùng mean/std **của chính xe** để scale noise.
  Không lấy mẫu thật làm init/nearest neighbor/coreset. Receiver không replay book của chính mình.
- Stats fp32 trước cộng; dữ liệu xuống/synthetic fp16 thực sự encode/decode số. Header object32B.
  S-a ghi payload (kể cả histogram); chưa tính retransmission/header từng frame vì chưa có contact.
- Hiệu chỉnh gp giữ theo P1; báo head để thấy prior correction có hợp với synthetic learning hay không.

### Cổng khóa (cả hai seed, r20)

1. Arm−local ≥+0,03 `group gp`, >0 `old gp`; benign F1 không giảm quá0,03.
2. **KME**−max(diag,kme_gauss) ≥+0,015 `group gp`.
3. **DM**−dm_gauss ≥+0,01 `group gp`.
4. Trượt: ghi loss synthesis, F1 theo lớp, head/gp và byte; không kéo dài/tune để cứu.
5. Qua: sang S-b dưới đây. Không gọi hai seed là kiểm định ý nghĩa thống kê.

## S-b: kênh hữu hạn cho hướng qua S-a

Giữ1contact/xe/round, B0+B2, simulator `mh/channel.py`; giữ cổng trước khi mở run.
Baseline local và đối chứng cơ chế chạy cùng driver/codec/prior có byte, không tái dùng mốc P1 miễn phí prior.
Chỉ model/thống kê/synthetic đã nhận đủ object/version mới dùng; histogram gp cũng cần nhận qua contact.
Codebook tĩnh chia theo (producer,class) coherent; cursor từng object, luân phiên nguồn/lớp theo cid để tránh bias.
KME chỉ decode tổng ≥5contributor; không dùng tensor quần thể trước khi tải xong phiên bản.
S-b giữ ≥70% lợi S-a so local; benign guard như trên. Báo cả byte up/down/ACK/retry,
coverage nguồn/lớp, round kích hoạt, Pareto F1-byte. Nếu không qua, không lên full-scale.

## Kiểm chứng và vận hành

- Local: math/payload checks bằng dữ liệu nhân tạo; smoke512row/client,512row/holdout,
  3round; test resume sau round1 và strict reconstruction từ weights thực ghi. Không rút kết luận F1.
- Kaggle:6arm×2seed, notebook riêng mỗi seed chạy đồng thời, private, image pin, assert2×T4.
  Ưu tiên quota thấp đủbudget: đọc lại trước push. Dự kiến ≤2h/notebook, cần đo thực tế.
- Giữ config+source+fixture hashes, weights từng round, predictions+confusion từng client tại round đánh giá,
  10metric JSON/CSV. Chỉ xác nhận hoàn tất sau pull và verifier đủ20round.
- Chưa mở protocol đầy đủ; bước hiện hành là S-a rồi S-b có điều kiện.
