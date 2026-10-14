# Pilot P1 — sàng lọc mọi hướng H1–H5 (không giới hạn kênh)

Lập 2026-10-10, **khóa trước khi chạy**. Chủ dự án yêu cầu (2026-10-10): thử tất cả hướng đề xuất để tìm hướng khả thi nhất;
hướng nào không khả thi thì tìm nguyên nhân và ghi bài học.
Driver: [`mh/pilot.py`](../../../core/pilot_runtime.py). Fixture: [`scripts/build_pilot_fixture.py`](../../../scripts/build_pilot_fixture.py).

## 1. Vì sao có pilot riêng

Protocol đầy đủ (50 round trên 43 M dòng) tốn khoảng 12 h GPU/arm trên 2×T4 (NILM-FL 20c: 12,2 h).
Sàng 15 arm × 2 seed theo cách đó thì không khả thi. Pilot dùng dữ liệu nhỏ, holdout nội bộ, không đụng test20%.
Chỉ hướng qua cổng mới lên các bước sau (kênh, khác kiến trúc, protocol đầy đủ).

## 2. Thiết lập chung

| mục | giá trị |
|---|---|
| Client | 20, Dirichlet α=0,5 (`fl_client/alpha05/20_client`), mọi client tham gia mọi round |
| Train | 2,5% số dòng non-holdout của mỗi client, lấy ngẫu nhiên (seed 1729), giữ ≥ 64 dòng/lớp. Tổng 1.000.770 dòng, mỗi client 19.882–136.358 dòng (giữ tỉ lệ cỡ client) |
| Holdout | `old`: 2% cuối theo thời gian của từng (lớp × scenario), lấy từ toàn partition. `group`: receiver bị giữ lại (hash mod 20 = 0), không có trong train. Mỗi tập 65.536 dòng |
| Model private | DAGSNet 395.024 tham số, cùng một init theo seed |
| Lịch | 20 round × 1 epoch local, batch 512, AdamW wd 0 tạo lại mỗi client/round, LR cosine theo round 1e-3 → 1e-5, clip 1,0, AMP fp16, loss fp32 |
| Đánh giá | round 5/10/15/20, mean trên 20 client của model suy luận. `head` = argmax. `gp` = argmax(logit − log π_i + log π_g), với π_g từ histogram nhãn gộp (64 B/client, mọi arm đều trả) |
| Seed | 42, 43 |

Khác protocol đầy đủ: ít dữ liệu, ít round, holdout thay test. Bài học archive: lợi ở pilot khi lên full scale thường còn khoảng một nửa.

## 3. Các arm (mỗi arm đổi một yếu tố so với `local`)

Mọi client dùng DAGSNet (chủ dự án 2026-10-11: chưa chạy đa mô hình; so với các phương pháp đã chạy kịch bản full DAGSNet).
Cột cuối chỉ ghi phương án có **khả thi về thiết kế** khi mỗi client dùng model khác nhau (R2) hay không.

| arm | hướng | đi qua kênh | byte (ước, mỗi client) | khác kiến trúc (R2) |
|---|---|---|---|---|
| `local` | mốc dưới | histogram nhãn | 64 B một lần | có |
| `fedavg` | đối chứng trên chia sẻ tham số | toàn bộ DAGSNet | ≈ 790 kB lên + 790 kB xuống / round | **không** (trung bình tham số cần cùng kiến trúc) |
| `dml_mlp64`, `dml_mlp256`, `dml_dagsnet` | H1 proxy + học tương hỗ (NILM-FL) | tham số proxy | 19 kB / 174 kB / 790 kB mỗi chiều / round | có: chỉ proxy chung kiến trúc, private tùy ý (KL trên logit 16 lớp) |
| `h2_full`, `h2_lr4`, `h2_diag` | H2 Gaussian lớp ở input, tổng cộng dồn | tổng `n, Σx, Σxxᵀ` | lên 146 kB (full/lr) hoặc 8,5 kB (diag) một lần. Xuống 73 / 12,7 / 4,2 kB một lần | có: mẫu ở không gian input 66 chiều, mọi model đọc được |
| `h2_mean` | đối chứng prototype | tổng `n, Σx` | lên 4,3 kB, xuống 2,1 kB một lần | có |
| `h2_bal` | đối chứng prior (replay cân bằng của chính xe) | không | 0 | có |
| `h2_own` | đối chứng làm mượt (thêm 2026-10-11 sau khi `h2_full` qua G1, **trước khi** có số của nó): Gaussian full-cov fit trên dữ liệu **của chính xe** | không | 0 | có |
| `lg` | H3a LG-FedAvg: lớp cuối chung | Linear 256→16 | 8,2 kB mỗi chiều / round | có điều kiện: mọi model phải có latent cùng số chiều (thêm projector) |
| `fedgh` | H3b FedGH: RSU train head trên prototype | prototype lên, head xuống | 8,3 / 8,2 kB / round | có điều kiện: như `lg` |
| `fedproto` | H4 FedProto (λ=1) | prototype | 8,3 kB mỗi chiều / round | có điều kiện: latent cùng số chiều; ý nghĩa latent khác nhau giữa kiến trúc |
| `dsfl` | H5a DS-FL, tập mở = mẫu Gaussian H2 với seed chung | thống kê H2 một lần + xác suất int8 trên 1.024 điểm/round | 16 kB mỗi chiều / round | có: chỉ trao đổi xác suất 16 lớp |
| `fedgen` | H5b FedGen trên giao diện LG | head lên; head + generator xuống | 8,2 kB lên, 165 kB xuống / round | có điều kiện: như `lg` (generator sinh latent chung) |

Batch phụ (H2, DS-FL, FedGen): 256 điểm/bước, trọng số loss 1, cùng forward với batch local (BN thấy batch trộn).

## 4. Cổng (khóa 2026-10-10, trước khi có số)

Chỉ số chính: `group`, quy tắc `gp`, f1_macro trung bình 20 client, round 20. Phụ: `old` gp. Mọi Δ là cặp cùng seed.

- **G1 — tri thức có ích:** arm − `local` ≥ **+0,03** ở `group` gp cả hai seed, và `old` gp > 0 cả hai seed.
- **G2 — benign:** benign F1 (`group` gp) không thấp hơn `local` quá **0,03** ở seed nào.
- **G3 — H2 mang tri thức ngoài prior/tâm lớp:** `h2_*` − `h2_bal` ≥ +0,01 và `h2_*` − `h2_mean` > 0, cả hai seed.
  Thêm (2026-10-11, khóa trước khi chạy): `h2_full` − `h2_own` ≥ +0,01 cả hai seed. Không đạt ⇒ lợi của H2 là làm mượt/chống học thuộc, không phải tri thức từ client khác.
- **G4 — H5 thêm được gì so với nền của nó:** `dsfl` − `h2_full` ≥ +0,01 và `fedgen` − `lg` ≥ +0,01, cả hai seed.
  Không đạt thì hướng đó không có đóng góp riêng.
- `fedavg` và `proxy` toàn cục của H1 chỉ là số tham chiếu, không chọn hướng theo chúng.

**Qua cổng G1 + G2** (và G3/G4 với nhánh tương ứng) ⇒ lên P2.
**Trượt** ⇒ phân tích nguyên nhân (F1 theo lớp, đường cong round, prior, loss local, byte), ghi vào [kết quả](probes/pilot-p1.md) và LESSONS.
Không chỉnh siêu tham số để cứu arm trượt, trừ khi nguyên nhân chỉ rõ một lỗi cài đặt.

## 5. Các bước sau

- **P2 — kênh** ([mh/p2.py](../../../mh/p2.py), khóa 2026-10-11 sau P1, trước khi chạy P2). **Chưa chạy:** đã khởi động local rồi dừng sau ~6 phút
  khi chủ dự án tắt máy và chốt chạy job nặng trên Kaggle; chưa có số nào. Hàng đợi `scripts/p2_queue.txt`. cùng fixture/lịch/đánh giá như P1, mỗi xe mỗi round một contact,
  ngân sách từ simulator archive ([mh/channel.py](../../../core/rsu_contact.py)). Hai chính sách:
  - `naive`: không cursor, xuống trước lên sau; object chỉ tới nếu vừa trong một contact.
  - `h6`: cursor tiếp tục giữa các contact, làm xong version đã bắt đầu, ưu tiên transfer dở dang, RSU giảm trọng số update cũ 1/(1+staleness).
  - Arm (sửa 2026-10-11 theo chủ dự án: **proxy H1 là DAGSNet**, đồng nhất cấu hình mọi client; proxy MLP của P1 chỉ còn là số tham khảo):
    - H2: B0 `h2_full`, `h2_diag` (naive + h6), `h2_lr4` (h6, đường lên giống full); B2 `h2_full` (naive + h6).
    - H1: `dml_dagsnet` int8 (395 kB mỗi chiều). B2 naive + h6; B0 chỉ h6 (naive ở B0 không bao giờ gửi xong một bản vì trần 32 KiB < 395 kB).
      Thêm `B3 naive` int8 làm mốc "int8, gần không giới hạn", để tách tác dụng của lượng tử hóa khỏi tác dụng của kênh.
  - **Cổng P2-a (giữ lợi):** lợi so với `local` của P1 (group gp, cùng seed) còn ≥ 70% lợi của chính arm đó ở P1 không ràng buộc, cả hai seed.
    Với `h2_diag`/`h2_lr4`, mốc 100% là P1 của chính dạng đó.
  - **Cổng P2-b (H6):** `h6` − `naive` ≥ +0,01 cả hai seed, ở tier mà `naive` mất > 30% lợi.
  - DML báo thêm `head` (P1 cho thấy `gp` sửa quá tay khi private bám proxy).
- ~~P3 — khác kiến trúc~~: **không chạy ở giai đoạn này** (chủ dự án 2026-10-11). Code hỗ trợ sẵn (`--private dagsnet,cnn1d,gru,mlp256`), chỉ đánh giá khả thi về thiết kế ở bảng §3.
- **P4 — protocol đầy đủ** (`fl_dagsnet_veremi`, 50 round, test toàn bộ): chạy trên Kaggle, phải hỏi chủ dự án trước vì là run dài.
