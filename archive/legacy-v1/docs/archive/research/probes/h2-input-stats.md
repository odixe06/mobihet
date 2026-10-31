# H2-T0-lite — thống kê lớp gộp ở không gian input (2026-10-10)

Script: [probe_h2_input_stats.py](../../../../archive/methods/gaussian-class-moments/builds/input-statistics-probe/scripts/probe_h2_input_stats.py). Số gốc: [results.json](../../../../archive/methods/gaussian-class-moments/builds/input-statistics-probe/h2_input_stats/results.json).
Chạy local (RTX 3050) qua `run_local_guarded.py`: quét dữ liệu 37 s, toàn bộ 230 s, peak RSS 3,8 GB.
Đề xuất: [H2](../de-xuat-huong-moi.md#h2--thống-kê-lớp-trong-không-gian-input--dữ-liệu-tổng-hợp-cho-head-triển-khai-ưu-tiên-2).

## Câu hỏi

Mỗi client chỉ gửi **tổng cộng dồn theo lớp** `n_c, Σx, Σxxᵀ` của 66 feature, không gửi dòng nào. Tổng gộp của các client khác có đem lại thông tin
mà client không tự có không? Không client nào thấy dữ liệu của client khác, kể cả trong đối chứng (quy tắc FL, chủ dự án nhắc 2026-10-10).

## Thiết kế

- Partition 20 client α=0,5. Client đích 0/5/10/15. Seed 42/43.
- Local = 300k dòng ngẫu nhiên của client. Thống kê gộp từ **toàn bộ dòng của 19 client còn lại**; script cộng các tổng như RSU/secure aggregation.
- MLP 66-256-256-16 (không BN), AdamW 1e-3, 3.000 bước, batch 512. Các arm trộn 256 local + 256 phụ.
- Test: mẫu ngẫu nhiên 3% (322.149 dòng) của test toàn cục, áp scaler một lần. Chỉ mô tả, không chọn HP.
- Arm:

| arm | phần phụ của batch |
|---|---|
| a0 | không (512 local) |
| a1 | replay cân bằng lớp từ chính client — **đối chứng** prior/cân bằng |
| diag | mẫu từ Gaussian đường chéo `N(μ_c, diag σ²_c)` gộp |
| full | mẫu từ Gaussian hiệp phương sai đầy đủ `N(μ_c, Σ_c)` gộp |
| mean | đúng điểm `μ_c` gộp — đối chứng kiểu prototype |

Cổng đề xuất (đặt trước khi chạy, [đề xuất H2](../de-xuat-huong-moi.md)): tổng hợp − a1 ≥ +0,01 và tổng hợp − mean > 0, cả hai seed.

## Kết quả (macro-F1)

| client | seed | a0 | a1 | diag | full | mean | diag−a1 | full−a1 | mean−a1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 42 | 0,5612 | 0,5646 | 0,5693 | 0,6115 | 0,5490 | +0,0046 | +0,0469 | −0,0156 |
| 0 | 43 | 0,5486 | 0,5569 | 0,5682 | 0,6226 | 0,5392 | +0,0113 | +0,0658 | −0,0177 |
| 5 | 42 | 0,5182 | 0,5254 | 0,6058 | 0,6529 | 0,5154 | +0,0804 | +0,1276 | −0,0100 |
| 5 | 43 | 0,5213 | 0,5311 | 0,6179 | 0,6601 | 0,4883 | +0,0869 | +0,1290 | −0,0428 |
| 10 | 42 | 0,5923 | 0,5991 | 0,6180 | 0,6504 | 0,5747 | +0,0190 | +0,0513 | −0,0244 |
| 10 | 43 | 0,5939 | 0,5975 | 0,6255 | 0,6511 | 0,5788 | +0,0279 | +0,0536 | −0,0188 |
| 15 | 42 | 0,4001 | 0,4144 | 0,5116 | 0,5962 | 0,4138 | +0,0973 | +0,1818 | −0,0006 |
| 15 | 43 | 0,4280 | 0,4160 | 0,5233 | 0,5902 | 0,4062 | +0,1074 | +0,1742 | −0,0097 |

| so sánh | TB seed 42 | TB seed 43 | dương |
|---|---:|---:|---:|
| full − a1 | **+0,102** | **+0,106** | 8/8 |
| diag − a1 | +0,050 | +0,058 | 8/8 |
| mean − a1 | −0,013 | −0,022 | 0/8 |

**Cổng PASS cho cả diag và full**, hai seed. Full gần gấp đôi diag ⇒ **thông tin bậc hai** (tương quan giữa feature) mang phần lớn lợi ích.
Điểm trung bình lớp **làm kém đi**, khớp bài học cũ (prototype trung bình lớp chỉ ±0,006 ở Build3–6). Client yếu nhất (15) lợi nhiều nhất: +0,17…+0,18.

F1 theo lớp (TB 4 client × 2 seed):

| lớp | a1 | diag | full | full − a1 |
|---|---:|---:|---:|---:|
| benign | 0,434 | 0,460 | 0,478 | +0,043 |
| constantPositionOffset | 0,554 | 0,773 | 0,786 | +0,232 |
| zeroSpeedReport | 0,597 | 0,802 | 0,839 | +0,241 |
| trafficCongestionSybil | 0,714 | 0,745 | 0,870 | +0,156 |
| suddenConstantSpeed | 0,255 | 0,373 | 0,409 | +0,154 |
| constantSpeedOffset | 0,278 | 0,300 | 0,422 | +0,144 |
| randomSpeedOffset | 0,677 | 0,789 | 0,813 | +0,136 |
| feignedBraking | 0,758 | 0,824 | 0,874 | +0,116 |
| suddenStop | 0,200 | 0,201 | 0,307 | +0,107 |
| randomPositionOffset | 0,817 | 0,884 | 0,916 | +0,100 |
| dataReplay | 0,339 | 0,348 | 0,419 | +0,080 |
| accelerationMultiplication | 0,840 | 0,814 | 0,897 | +0,057 |
| reversedHeading | 0,735 | 0,745 | 0,784 | +0,049 |
| dosAttack | 0,957 | 0,993 | 0,993 | +0,036 |
| positionMirroring | 0,140 | 0,126 | 0,153 | +0,013 |
| timeDelayAttack | 0,115 | 0,103 | 0,112 | −0,003 |

timeDelay/positionMirroring không đổi: đúng trần thông tin của bộ sinh dữ liệu ([kế thừa §2](../../legacy/kien-thuc-ke-thua.md#2-dữ-liệu--sự-thật-đã-đo)).

## Giới hạn — đọc trước khi dùng số

- **MLP, không phải DAGSNet**; 3.000 bước, không phải 50 round × 1 epoch. Mốc local yếu (0,40–0,59), nên lợi tuyệt đối có thể co lại khi local mạnh hơn.
  Pilot → full scale thường chỉ còn khoảng nửa lợi (LESSONS §3).
- Thống kê gộp dùng **đủ dữ liệu** của 19 client, một lần, không tính ngân sách contact, không có nhiễu DP.
- **Payload hiệp phương sai đầy đủ:** 16 lớp × (2.211 + 66) giá trị ≈ 73 kB fp16 mỗi chiều. Vượt B0 (4,5 kB/contact) nhưng chỉ cần gửi khi dữ liệu local đổi:
  với dữ liệu tĩnh, mỗi client upload một lần (≈ 16 contact ở B0, 1–2 ở B1). Diag 4,3 kB vừa một contact B0. Hạng thấp là điểm giữa, chưa đo.
- **Riêng tư:** tổng `Σxxᵀ` của một client trong không gian input lộ nhiều hơn tổng trong latent (ví dụ trung bình các feature vân tay receiver).
  Muốn RSU chỉ thấy tổng nhiều client thì cần secure aggregation; muốn bảo đảm định lượng thì cần DP. Chưa đo ảnh hưởng của nhiễu DP.
- Test là mẫu con 3%, mô tả; chưa có holdout nội bộ. Một lần chạy cho mỗi (client, seed).

## Bước tiếp đề xuất (chưa chạy, chờ chủ dự án)

1. Hạng thấp r ∈ {4, 8, 16} + diag dư, để biết payload nhỏ nhất còn giữ lợi.
2. DAGSNet: replay tổng hợp trong lúc train local và/hoặc head triển khai trên backbone EMA đóng băng, theo protocol `fl_dagsnet_veremi`, 20c.
3. Độ nhạy với nhiễu DP trên các tổng (ε vài mức) và với số client gộp (≥ k).
4. Mixture theo cụm trong lớp (lớp đa mode: benign, Sybil) nếu Gaussian một mode bão hòa.
