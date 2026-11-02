# Phương pháp đề xuất — MobiHet cho IDS V2X băng thông thấp, mô hình khác nhau

Trạng thái 02/11/2026: **lõi đã có bằng chứng; ba thành phần đang kiểm ở W20.** Không phải phương pháp đã chốt.
Bằng chứng: [R20/V20](results-r20-v20.md), [Q20](results-q20.md), [P20](results-p20.md), [LESSONS](../../LESSONS.md).

## Ràng buộc thiết kế

1. **Băng thông hạn chế:** không trao đổi trọng số; mọi gói tính đủ byte hai chiều, gồm bootstrap.
2. **Mỗi xe một mô hình bất kỳ:** chỉ trao đổi thứ không phụ thuộc kiến trúc — thống kê dữ liệu gộp và xác suất đầu ra
   16 lớp. Yêu cầu duy nhất với mô hình xe: nhận 77 đầu vào, trả logit 16 lớp. Thí nghiệm hiện giữ DAGSNet W1 cho mọi xe
   để cô lập cơ chế; trộn kiến trúc chưa chạy (owner 02/11: chỉ cần bảo đảm cách truyền dùng được cho nhiều model).
3. Không gửi dòng dữ liệu, nhãn từng dòng, trọng số riêng, BN hay gradient. Chưa có DP/secure aggregation.

## Luồng thông tin

```text
Bootstrap (1 lần)   xe → server: số mẫu/lớp, histogram biên 77×256 theo lớp (mã thưa), tổng hiệp phương sai normal-score
                    server → xe: prior toàn cục, CDF biên, mean/Cholesky copula        → mỗi xe tự sinh synthetic
Mỗi round           xe → server: Q_k = trung bình xác suất dự đoán theo lớp thật trên dữ liệu local (16×16 fp16)
                    server → xe: Q = trung bình theo số mẫu n_kc (16×16 fp16)
Tầng tùy chọn       + xác suất trên 1024 truy vấn synthetic chung (seed) → target theo input cho 32/64 dòng synthetic
```

| thành phần | byte/xe | trạng thái |
|---|---:|---|
| bootstrap hiện tại (fp32 dày) | 3.311.744 (1 lần, 2 chiều) | đã chạy mọi batch |
| bootstrap uplink thưa, không mất mát | 271.031 | đo offline, tổng server giữ nguyên → không cần chạy lại |
| bootstrap downlink nén (CDF uint16, fp16 Cholesky) | 729.408 | **W20 boot_compact** |
| class Q mỗi round | 544 + 544 | đã chạy |
| tầng input-bank mỗi round | +32.800 mỗi chiều | đã chạy (Q/V) |
| **tổng 20 round, lõi nén** | **≈1,02 MB** | phụ thuộc W20 |
| tổng 20 round, lõi + input-bank | ≈2,33 MB | phụ thuộc W20 |
| tham chiếu: FedAvg/APFed trao trọng số | ≈63,8 MB | benchmark |

## Học tại xe

1. **CE + logit adjustment** trên đúng 1 epoch dữ liệu local mỗi round (như repo benchmark).
2. **64 dòng synthetic copula mỗi bước**, target từ Q (round 1: nhãn cứng). Đây là nguồn lợi lớn nhất:
   val sạch +.0226 pooled, p10 +.11–.13, lớp vắng 0 → .76; test +.033 pooled.
3. Target synthetic: hiện là KL(Q[y]‖p). **W20 synth_nckd** thử nhãn cứng cho phần đúng/sai + giữ quan hệ lớp sai
   (non-target KD) để hàng Q nhầm (Q[13] ≈ .74 timeDelay/.25 benign) không dạy synthetic timeDelay thành benign.
4. **Không realKD** (KD trên dòng thật) ở mặc định: replay cùng teacher vẫn −.05 timeDelay, −.05 lớp vắng; val sạch không còn
   lợi pooled. **W20 replay/live_real_nckd** kiểm dạng chỉ phần non-target — nếu bỏ được tác hại mà giữ +.005 other15 thì đưa lại.
5. **Step floor synthetic** (W20): xe có < 16 bước thật học thêm bước chỉ synthetic. 0 byte, +3,5% bước client.
   Xe < 256 dòng hiện chỉ 2–4 bước/round và còn tăng mạnh r15→20; đây là ứng viên chính cho khoảng cách p10 với APFed.

## Server

Gộp Q theo n_kc (đã thử sqrt: không ổn định). Không proxy model, không model downlink (P20: không lợi, tốn byte).
Tầng input-bank gộp xác suất trên truy vấn chung theo n_k,y; chỉ dùng khi ngân sách cho phép (lợi chủ yếu ở xe yếu).

## Bằng chứng theo thành phần

| thành phần | kết luận | nguồn |
|---|---|---|
| copula augmentation | giữ — lợi lớn, tổng quát hóa trên val sạch | V20, P20 C1−C0 |
| class Q trên synthetic | giữ — +.0016–.0019 pooled so augmentation, > độ ồn phiên .0005 | P20 C2−C1 |
| quan hệ off-diagonal của Q | giữ — fullQ hơn uniform/permutation | O20 (archive) |
| input-target bank | tầng tùy chọn — val: pooled +.0025, p10 +.009–.014 so augmentation | V20, Q20 |
| realKD binary/margin/strength | loại khỏi mặc định — hại timeDelay/lớp vắng cả khi teacher cố định | R20, V20 |
| sqrt input weight, proxy server, model downlink | loại | Q20, P20 |
| non-target KD, step floor, bootstrap nén | đang kiểm | [W20](screening-w20.md) |

## Giới hạn còn mở

- **timeDelay** là giới hạn của dữ liệu/đặc trưng: CE thuần trên val vẫn recall train .886 → val .267 ở xe lớn.
  Cải thiện cần đặc trưng thời gian/ngữ cảnh mới ở repo features, không chỉ cơ chế FL.
- **Xe yếu:** p10 .69–.71 so APFed .761 (chia trọng số, ~19× byte). Step floor là phép thử đầu; nếu không đủ, kế tiếp là
  bộ nhớ truy vấn/bank tích lũy và reliability từ validation local (đã thiết kế, chưa chạy).
- 20 round chưa bão hòa; kết luận cuối phải chạy lại combo đã chọn ở 50 round theo [protocol](protocol.md) cùng baseline.
- Trộn kiến trúc chưa kiểm: cần một ô có ≥2 kiến trúc (vd DAGSNet W1 + W0.5) trước khi tuyên bố heterogeneous.

## Sau W20 — quy tắc chọn

Đọc mọi hiệu so theo độ ồn phiên ([R20/V20 §1](results-r20-v20.md#1-độ-ồn-giữa-các-phiên-cùng-công-thức-cùng-seed)).
Giữ thành phần qua gate của nó ở cả hai seed; rồi chạy **một ô combo** (lõi + các thành phần qua gate, tầng có/không input-bank)
2 seed × 20 round, sau đó 50 round cho bảng benchmark. Không cộng gain từng ô để dự đoán combo.
