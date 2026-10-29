# Kết quả pilot P1 — sàng lọc H1–H5

**XONG 2026-10-11 08:50 UTC+7**, 32 run (16 arm × 2 seed), không lỗi. Kế hoạch + cổng khóa trước: [pilot-plan.md](../pilot-plan.md).
Bảng đầy đủ: `python scripts/report_pilot.py` (đọc `papers/pilot/runs/*/result.json`).

## 1. Số (round 20, `group` gp, mean 20 client; Δ là cặp cùng seed so với `local`)

| arm | s42 | s43 | Δ group | Δ old | Δ benign F1 | G1 | G2 | byte/client/round (lên/xuống) | khác KT (R2) |
|---|---:|---:|---|---|---|---|---|---|---|
| `local` | 0,4646 | 0,4633 | — | — | — | | | 64 B một lần | có |
| `fedavg` (tham chiếu) | 0,8006 | 0,7969 | +0,336 / +0,334 | +0,325 / +0,325 | +0,389 / +0,385 | PASS | PASS | 797 / 797 kB | **không** |
| `dml_dagsnet` (H1, NILM-FL) | 0,6212 | 0,6256 | **+0,157 / +0,162** | +0,156 / +0,165 | +0,199 / +0,204 | PASS | PASS | 797 / 797 kB | có |
| `dml_mlp256` (H1) | 0,5865 | 0,5885 | **+0,122 / +0,125** | +0,124 / +0,128 | +0,126 / +0,130 | PASS | PASS | 174 / 174 kB | có |
| `h2_full` (H2) | 0,5697 | 0,5727 | **+0,105 / +0,109** | +0,104 / +0,111 | +0,035 / +0,036 | PASS | PASS | 146 / 73 kB **một lần** | có |
| `dml_mlp64` (H1) | 0,5141 | 0,5143 | +0,050 / +0,051 | +0,052 / +0,056 | +0,030 / +0,031 | PASS | PASS | 19 / 19 kB | có |
| `h2_lr4` (H2) | 0,5550 | 0,5557 | +0,091 / +0,092 | +0,086 / +0,092 | +0,027 / +0,030 | PASS | PASS | 146 / **12,7 kB** một lần | có |
| `h2_diag` (H2) | 0,5425 | 0,5443 | +0,078 / +0,081 | +0,079 / +0,082 | +0,022 / +0,028 | PASS | PASS | **8,6 / 4,3 kB** một lần | có |
| `dsfl` (H5a) | 0,5393 | 0,5385 | +0,075 / +0,075 | +0,070 / +0,073 | +0,016 / +0,018 | PASS | PASS | 146 / 73 kB một lần + 16 / 16 kB | có |
| `fedproto` (H4) | 0,5044 | 0,5063 | +0,040 / +0,043 | +0,041 / +0,046 | +0,039 / +0,032 | PASS | PASS | 8,3 / 7,8 kB | có điều kiện |
| `h2_own` (đối chứng làm mượt) | 0,4875 | 0,4887 | +0,023 / +0,025 | +0,022 / +0,027 | +0,025 / +0,029 | fail | PASS | 0 | có |
| `fedgen` (H5b) | 0,4823 | 0,4807 | +0,018 / +0,017 | +0,017 / +0,020 | +0,007 / +0,008 | fail | PASS | 8,2 / 159 kB | có điều kiện |
| `lg` (H3a) | 0,4838 | 0,4839 | +0,019 / +0,021 | +0,020 / +0,023 | +0,005 / +0,005 | fail | PASS | 8,2 / 8,2 kB | có điều kiện |
| `h2_mean` (đối chứng) | 0,4617 | 0,4614 | −0,003 / −0,002 | | | fail | | 4,4 / 2,2 kB một lần | có |
| `h2_bal` (đối chứng) | 0,4574 | 0,4584 | −0,007 / −0,005 | | | fail | | 0 | có |
| `fedgh` (H3b) | 0,4453 | 0,4507 | −0,019 / −0,013 | −0,017 / −0,010 | −0,008 / −0,023 | fail | PASS | 8,3 / 7,8 kB | có điều kiện |

Model phụ (không dùng để chọn hướng):

| model | s42 | s43 |
|---|---:|---:|
| proxy toàn cục `dml_dagsnet` | 0,7740 | 0,7701 |
| proxy toàn cục `dml_mlp256` | 0,6752 | 0,6776 |
| proxy toàn cục `dml_mlp64` | 0,5668 | 0,5653 |
| head toàn cục `lg` | 0,4840 | 0,4840 |
| head toàn cục `fedgh` | 0,4394 | 0,4482 |

Cổng riêng:
- **G3 (H2) PASS** cho cả ba dạng:
  - − `h2_bal`: full +0,112 / +0,114, lr4 +0,098 / +0,097, diag +0,085 / +0,086;
  - − `h2_mean` đều > +0,08;
  - **`h2_full` − `h2_own` +0,082 / +0,084**.
- **G4 (H5) fail cả hai:** `dsfl` − `h2_full` −0,030 / −0,034; `fedgen` − `lg` −0,002 / −0,003.

## 2. Phân tích

### 2.1 H2 — thống kê bậc hai ở input

- Tâm lớp (`h2_mean`) và cân bằng lớp (`h2_bal`) đều ≈ 0. Lợi tăng theo lượng cấu trúc bậc hai được giữ: diag +0,08, hạng 4 +0,09, full +0,11.
- **Phần lớn lợi là tri thức của client khác, không phải làm mượt.** Gaussian fit trên dữ liệu của chính xe (`h2_own`) chỉ +0,024.
  Gaussian gộp 20 client (`h2_full`) hơn nó +0,083. Phân phối có điều kiện theo lớp ở mức quần thể chứa đa dạng receiver mà một xe không có.
- Loss r20 đã lưu: `h2_full` 0,63, `h2_own` 0,21, `local` 0,01. **Đính chính khi tổng hợp V2:** loss H2
  gồm CE real + CE synthetic; chưa chứng minh riêng CE real cao hoặc chặn học thuộc vân tay receiver.
  Lợi của thống kê quần thể có bằng chứng từ contrast F1; cơ chế chống học thuộc cần loss tách và holdout riêng.
- Lợi rải trên 14/16 lớp (+0,04…+0,18). Hai lớp không được lợi:
  - positionMirroring +0,01;
  - timeDelay +0,02.
  FedAvg lại học được positionMirroring (+0,26). ⇒ lớp này **không** chạm trần thông tin: phân phối Gaussian lớp không mô tả được
  quan hệ phi tuyến của nó (vị trí lật theo trục đường, heading giữ nguyên). Chỉ timeDelay có trần đã biết (centralized 0,23).
- Byte: lên 146 kB, xuống 73 kB, **một lần duy nhất** cho cả run (dữ liệu tĩnh). Tính trên 20 round thì 7,3 + 3,6 kB/round, rẻ nhất trong các hướng có lợi lớn.

### 2.2 H1 (DML) — lợi tăng theo sức chứa proxy

- Private: mlp64 +0,05, mlp256 +0,12, DAGSNet +0,16. Proxy toàn cục: 0,57 / 0,68 / 0,77. Trên dữ liệu này, chia sẻ tham số mang tri thức đa receiver tốt nhất.
- Chi phí tăng tương ứng: 19 / 174 / 797 kB mỗi chiều mỗi round. Ở B0 (trung vị 4,7 kB/contact) chỉ mlp64 int8 (9,5 kB) là gần khả thi.
- **`gp` sửa quá tay khi private bị chưng cất về mô hình toàn cục.** Ví dụ `dml_mlp64`:
  - `head` tăng đều 0,415 → 0,558 (r5 → r20);
  - `gp` đạt đỉnh 0,570 ở r15 rồi rơi về 0,514, benign F1 0,55 → 0,38. Hai seed giống nhau; mlp256 cũng vậy (đỉnh r15 0,611 → 0,587).
  - Cơ chế: trọng số KL kiểu NILM là 1/sg(ce_s + ce_r), lớn dần khi CE giảm, nên cuối run private bám sát proxy.
    Prior hiệu dụng của private dịch về π_g, và trừ log π_i thành sửa quá tay.
  - ⇒ Bài học: `gp` chỉ đúng với model học CE thuần trên phân bố local. Phương pháp kéo model về tri thức toàn cục phải ước prior hiệu dụng, hoặc báo cả `head`.
- Loss train tăng ở cuối run (cả `fedavg`) là do cách đo: LR cosine về 1e-5 nên model trong một round hầu như không khớp thêm vào dữ liệu local.
  Loss trung bình khi đó ≈ loss của model được tải xuống. Đây không phải dấu hiệu trôi.

### 2.3 H3 (head chung) — trượt: chia sẻ head không sửa backbone

- `lg` +0,02: chỉ 4k tham số lớp cuối được trung bình; backbone vẫn học thuộc (loss local r20 0,009, bằng `local`). Head chung trên latent đã lệch thì không mang ranh giới đa receiver.
  Khớp C3/M2 của archive (chỉnh head không cứu được backbone).
- `fedgh` −0,02:
  - Head toàn cục được RSU fit trên 20×16 **tâm lớp** latent, nên bị chặn bởi C1 (trung bình lớp không chở ranh giới).
  - Mỗi round head local bị thay bằng head lệch với backbone, nên loss train kẹt 0,45–0,49 suốt run.
  - Cuối run LR nhỏ, backbone không kịp thích nghi, loss lên 0,68–0,86 và điểm giảm.
- ⇒ Bài học: tri thức chỉ đặt ở lớp cuối (dù trung bình tham số hay fit từ prototype) không đủ khi lỗi nằm ở representation học thuộc theo receiver.

### 2.4 H4 (FedProto) — qua G1 nhưng nhỏ, đúng như archive

- +0,04, lặp lại số archive (prototype so với không trao đổi +0,045). MSE kéo latent về tâm lớp toàn cục, làm giảm học thuộc (loss local 0,04 so với 0,01).
- Lợi đều nhưng nhỏ trên mọi lớp (+0,01…+0,08), không lớp nào nhảy vọt. Tâm lớp không chở ranh giới.
- Vai trò: baseline HtFL rẻ (8 kB/round). Không đầu tư biến thể (archive: mọi biến thể ±0,006).

### 2.5 H5 — chưng cất bằng logit và generator: trượt G4, không thêm gì so với nền

- **DS-FL** (+0,075) **thua** chính thống kê H2 mà nó dùng làm tập mở (`h2_full` +0,107).
  - Cùng các điểm Gaussian, nhãn lớp sinh ra chúng là chính xác theo định nghĩa.
  - Soft label DS-FL là trung bình (đã làm sắc, T = 0,1) của 20 model local, mỗi model chỉ ~0,46 F1.
  - Teacher kém hơn nhãn miễn phí, nên chưng cất thay nhãn cứng bằng nhãn nhiễu.
  - Thêm nữa, mỗi round chỉ 1.024 điểm mở, nên model học thuộc được (loss local 0,015, so với `h2_full` 0,63).
  - Khớp bài học archive: teacher không qua cổng chất lượng chỉ ngang làm mượt (Build7, S13).
- **FedGen** (+0,018) ≈ `lg` (+0,020):
  - Generator sinh latent để head đúng. Gradient của nó chỉ đi vào lớp cuối, backbone vẫn học thuộc (loss local 0,009).
  - Cùng nguyên nhân với H3: tri thức không chạm tới representation.
- ⇒ Bài học:
  - Có sẵn mô hình sinh theo lớp thì nhãn của nó tốt hơn ensemble của các model yếu. Logit/KD chỉ đáng dùng khi teacher mạnh hơn hẳn (ví dụ proxy toàn cục H1).
  - Mọi kênh chỉ tác động lên head đều bị chặn bởi C3.

## 3. Kết luận P1

| hướng | kết quả (group gp, Δ local) | byte | R2 | lên P2 |
|---|---|---|---|---|
| H1 DML (proxy DAGSNet / MLP256 / MLP64) | +0,16 / +0,12 / +0,05; proxy toàn cục 0,77 / 0,68 / 0,57 | 797 / 174 / 19 kB mỗi chiều mỗi round | có | mlp256 ở B2, mlp64 int8 ở B0 |
| H2 Gaussian input (full / lr4 / diag) | +0,11 / +0,09 / +0,08; −own +0,08 | một lần: 146+73 / 146+13 / 8,6+4,3 kB | có | full ở B0+B2, diag và lr4 ở B0 |
| H4 FedProto | +0,04 | 8 kB/round | có điều kiện | baseline |
| H3 LG / FedGH | +0,02 / −0,02 | 8 kB/round | có điều kiện | dừng |
| H5 DS-FL / FedGen | +0,075 (< H2) / +0,02 (≈ LG) | 16 kB/round / 167 kB/round | có / có điều kiện | dừng |

- Mốc trên tham chiếu FedAvg DAGSNet +0,33 (797 kB mỗi chiều mỗi round, không khác kiến trúc được). Khoảng cách tới mốc này còn lớn.
- Hai hướng khả thi: **H2** (rẻ nhất, một lần, hợp B0) và **H1** (lợi lớn nhất nhưng tốn byte mỗi round). Chúng tác động theo hai cách khác nhau
  (H2: dữ liệu tổng hợp; H1: chưng cất từ model toàn cục), nên có thể **kết hợp H1 + H2** — đề xuất cho bước sau.
