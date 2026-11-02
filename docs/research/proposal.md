# Phương pháp đề xuất — MobiHet cho IDS V2X băng thông thấp, mô hình khác nhau

Trạng thái 02/11/2026 sau W20: **lõi đã chốt thành phần; combo và realKD đang kiểm ở [X20](screening-x20.md).** Chưa có 50 round.
Bằng chứng: [W20](results-w20.md), [R20/V20](results-r20-v20.md), [Q20](results-q20.md), [P20](results-p20.md), [LESSONS](../../LESSONS.md).

## Ràng buộc thiết kế

1. **Không chia sẻ dữ liệu local** (chuẩn FL): không gửi dòng dữ liệu, nhãn từng dòng, đặc trưng từng mẫu, trọng số, BN,
   gradient. Chỉ gửi thống kê gộp và xác suất trung bình; server chỉ dùng **tổng** trên mọi xe (§Quyền riêng tư).
2. **Băng thông hạn chế:** không trao đổi trọng số; mọi gói tính đủ byte hai chiều, gồm bootstrap.
3. **Mỗi xe một mô hình bất kỳ:** chỉ trao đổi thứ không phụ thuộc kiến trúc. Yêu cầu duy nhất với mô hình xe: nhận 77 đầu vào,
   trả logit 16 lớp. Thí nghiệm giữ DAGSNet W1 cho mọi xe để cô lập cơ chế; trộn kiến trúc chưa chạy.

## Luồng thông tin

```text
Bootstrap (1 lần, 2 pha)  xe → server: số mẫu/lớp, histogram biên 77×256 theo lớp
                          server → xe: prior + CDF biên toàn mạng (uint16)
                          xe → server: tổng normal-score Σz, Σzzᵀ theo lớp (dùng CDF toàn mạng)
                          server → xe: mean/Cholesky copula (fp16)              → mỗi xe tự sinh synthetic
Mỗi round                 xe → server: Σ_rows p(x) theo lớp thật (16×16) + số mẫu/lớp
                          server → xe: Q = trung bình toàn mạng (16×16 fp16)
Tầng tùy chọn             + xác suất trên 1024 truy vấn synthetic chung (seed) → target theo input cho 32/64 dòng synthetic
```

## Học tại xe

1. **CE + logit adjustment** trên đúng 1 epoch dữ liệu local mỗi round (như repo benchmark).
2. **64 dòng synthetic copula mỗi bước**, target KL(Q[y]‖p) (round 1: nhãn cứng). Nguồn lợi lớn nhất:
   val sạch +.0226 pooled, p10 +.11–.13, lớp vắng 0 → .76.
3. **Step floor synthetic 16 bước/round** (W20 PASS): xe có <16 bước thật học thêm bước chỉ synthetic; dữ liệu local vẫn đúng
   1 epoch. 0 byte, +3,5% bước, +2–4% thời gian. Xe <256 dòng: F1 +.018–.027, lớp vắng +.12–.14, p10 +.008–.018.
4. **Không realKD** ở mặc định. KD đầy đủ hại timeDelay/lớp vắng và tạo feedback; phần non-target (NCKD) cắt feedback và cho
   pooled cao nhất (+.0022–.0025) nhưng còn hại lớp vắng −.03. X20 kiểm `nckd_local` (chỉ lớp xe có) — chỉ đưa vào nếu qua gate.
5. Loại: synthetic hard+NCKD (W20: chỉ dời ngưỡng timeDelay/benign, benign F1 −.018), margin/strength, sqrt input, proxy,
   model downlink.

## Byte/xe

| thành phần | byte | trạng thái |
|---|---:|---|
| bootstrap uplink thưa, không mất mát | 271.031 (1 lần) | W20 boot_compact |
| bootstrap downlink CDF uint16 + fp16 mean/Cholesky | 729.408 (1 lần) | W20 boot_compact, utility trong spread |
| class Q mỗi round | 544 + 544 | mọi batch |
| tầng input-bank mỗi round | +32.800 mỗi chiều | X20 bank |
| **tổng 20 round, lõi** | **1.022.199** | W20; combo X20 core |
| tổng 20 round, lõi + input-bank | 2.334.199 | X20 bank |
| lõi khi có secure aggregation (uplink dày, số nguyên 32/64 bit) | ≈2.418.000 | ước tính, chưa chạy |
| tham chiếu FedAvg/APFed trao trọng số | ≈63,8 MB | benchmark |

## Quyền riêng tư

- Server chỉ dùng **tổng toàn mạng**: histogram và Σz/Σzzᵀ gộp bằng allreduce, Q = Σ n_kc·Q_kc / Σ n_kc, bank gộp theo n_k,y
  (đã kiểm code `methods/mobihet.py`, `mobihet_next.py`). Không bước nào cần thống kê riêng của một xe → tương thích
  secure aggregation (server chỉ thấy tổng). Downlink chỉ là thống kê toàn mạng.
- **Chưa có secure aggregation/DP trong thí nghiệm.** Không có SecAgg, histogram 77×256 của xe nhỏ gần như lộ giá trị biên
  từng mẫu (không lộ liên kết giữa các cột). Tuyên bố bảo mật phải ghi rõ: tương thích SecAgg, chưa có DP.
- SecAgg cần vector dày cố định độ rộng: uplink thưa 271 KB không dùng được; uplink dày = 1.655.872 B (đúng bằng uplink
  fp32 đã chạy ở mọi batch trước W boot_compact) → lõi ≈2,42 MB/xe/20 round, vẫn ≈1/26 FedAvg. Không đổi utility.

## Bằng chứng theo thành phần

| thành phần | kết luận | nguồn |
|---|---|---|
| copula augmentation | giữ — lợi lớn, tổng quát hóa trên val sạch | V20, P20 |
| class Q trên synthetic | giữ — +.0016–.0019 pooled so augmentation | P20 |
| quan hệ off-diagonal của Q | giữ — fullQ hơn uniform/permutation | O20 (archive) |
| step floor synthetic | giữ — xe nhỏ/lớp vắng tăng mạnh, pooled giữ | W20 |
| bootstrap nén | giữ — −69% byte, utility trong spread | W20 |
| input-target bank | tầng tùy chọn — chờ X20 bank (đã có floor) | V20, Q20, X20 |
| realKD non-target | chờ X20 nckd_local — pooled tốt nhất nhưng hại lớp vắng | W20 |
| realKD binary/margin/strength, synthetic hard+NCKD | loại | R20, V20, W20 |
| sqrt input weight, proxy server, model downlink | loại | Q20, P20 |

## Giới hạn còn mở

- **timeDelay** bị giới hạn bởi đặc trưng: CE thuần recall train .886 → val .267 ở xe lớn; mọi biến thể loss chỉ đổi điểm
  làm việc timeDelay/benign. Cần đặc trưng thời gian/ngữ cảnh ở repo features.
- **Xe yếu:** p10 ~.708 (floor) so APFed .761 (chia trọng số, ~60× byte lõi). Low256 r15→20 còn tăng: 50 round sẽ cho biết
  khoảng cách thật.
- Kết luận cuối phải chạy combo đã chọn ở 50 round theo [protocol](protocol.md) cùng baseline.
- Trộn kiến trúc chưa kiểm: cần một ô ≥2 kiến trúc (vd DAGSNet W1 + W0.5) trước khi tuyên bố heterogeneous.

## Sau X20 — quy tắc chọn

Đọc theo [screening-x20](screening-x20.md). Phương pháp = core (+ bank nếu qua gate) (+ nckd_local nếu nckd_local và full qua gate).
Combo thắng chạy 2 seed × 50 round cùng baseline benchmark; kèm một ô trộn kiến trúc nếu owner duyệt.
