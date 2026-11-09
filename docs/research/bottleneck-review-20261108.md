# Nút thắt Core so oracle/APFed/centralized — 08/11/2026

> Snapshot phân tích 08/11 (trước KM50 r50). §4 "bước tiếp" đã thực hiện qua sim b1–b11; số gap sau KM ở [catalog đề xuất](improvement-proposals.md).
> §1–§3, §5, §7 (cấu trúc dataset, ràng buộc compute/không trao model) vẫn là căn cứ hiện hành.

Chẩn đoán hậu kiểm, chưa launch. Nguồn: CM own-test r50 đã audit (Core F2/F3, oracle Y0, nofloor Y6, FedAvg/APFed/centralized
repo benchmark, seed42 một seed) và train CM Core r20. Tái tạo `scripts/analyse_core_bottleneck.py` → `runs/transport-probe/bottleneck.txt`.
Probe generator chỉ train→val: `scripts/probe_transport_generator.py` → `runs/transport-probe/judge.json`. Official test đã đọc lặp.

## 1. Gap không nằm chủ yếu ở benign↔timeDelay

| r50 seed42 | Core | oracle | FedAvg | APFed | central |
|---|---:|---:|---:|---:|---:|
| accuracy | .9094 | .9141 | .9228 | .9278 | .9580 |
| precision_macro / recall_macro | .898/.902 | .911/.924 | .935/.896 | .938/.903 | .968/.953 |
| lỗi benign↔timeDelay | 500k | 575k | 457k | 437k | 303k |
| lỗi 14 lớp còn lại | 347k | 229k | 264k | 239k | 90k |
| F1 benign + F1 timeDelay | 1.183 | 1.212 | 1.046 | 1.149 | 1.537 |

- Core→APFed: 172k lỗi = 63k b/td + **109k lớp khác**. Tổng F1 b/td của Core còn **cao hơn** APFed; Core dự đoán timeDelay đúng tỷ lệ prior
  (5,2%), APFed gần như bỏ timeDelay (dự đoán 0,96%). Gap macro tới APFed gần như toàn bộ là **precision lớp tấn công**:
  lớp1 P .835/.976, lớp7 .813/.900, lớp6 .919/.969, lớp3 .950/.975, lớp10 .971/.993. FP chủ yếu từ dòng benign.
- Oracle sửa phần 14 lớp (229k, tốt hơn APFed) nhưng làm b/td xấu hơn (+75k). Dữ liệu thật đúng hơn không sửa b/td.
- FP lớp c xảy ra ở xe **có lớp c cả train lẫn test** (Core 439.630 vs 6.887 ở xe không có c): không phải tác dụng phụ của bảo vệ lớp vắng/prior toàn cục.

## 2. Đây là lỗi tổng quát hóa, không phải fit

Train CM r20 Core: accuracy .978, precision lớp1/7/11 .991/.995/.987; own-test .839/.811/.820 (xe ≥16.384: .893/.837/.833).
Model riêng học thuộc dữ liệu local (một epoch × số round) và chỉ nhận tri thức toàn mạng qua synthetic cân bằng 64 dòng/bước.
FedAvg/APFed mỗi round khởi đầu từ hàm học trên mọi xe, nên ít FP hơn trên benign tương lai. Nofloor43 r50 pooled ≈ Core43, nhưng
xe<256 client mean −.028: floor quan trọng cho xe nhỏ, không ảnh hưởng nút thắt pooled.

## 3. Vì sao synthetic copula thua oracle — synthetic lệch ranh giới thật

Judge học trên dòng thật fold train (60 file; **thước đo offline**, không thuộc phương pháp, không triển khai):
tỷ lệ dòng được judge gán đúng nhãn. `runs/transport-probe/{judge,lattice}.json`.

| lớp | val thật (MLP) | copula (MLP) | copula (HGB) | copula snap lưới (HGB) |
|---|---:|---:|---:|---:|
| 1 accelerationMult. | .960 | .855 | .780 | .779 |
| 7 positionMirroring | .947 | .791 | .800 | .791 |
| 4 dataReplay | .968 | .883 | .863 | .854 |
| 10 reversedHeading | .998 | .957 | .946 | .946 |
| 3 constantSpeedOffset | .983 | .950 | .963 | .949 |
| 13 timeDelay | .709 | .421 | .480 | .489 |
| 11 / 12 / 15 | .907/.998/.922 | .812/.996/.942 | .454/.411/.467 | .779/.997/.902 |
| trung bình 16 lớp | .934 | **.872** | .781 | .863 |

- **Sửa kết luận tạm:** lớp11/12/15 "sai nhãn" theo HGB là artifact của judge cây (ngưỡng sát atom; feature rời rạc như `ses_dt`
  63 giá trị/atom 86%, `nbr_n_close` 7, cờ nhị phân). Snap về lưới giá trị hợp lệ sửa HGB nhưng judge MLP (mượt, gần DAGSNet) không đổi:
  **bỏ lattice-snap**.
- Theo judge MLP, thiếu hụt còn lại ở lớp 1/7/4/10/3/13 — trùng các lớp mất precision trong FL (mục 1). Dòng thật local che phần này
  trên train (precision train .99); vùng chưa thấy bị synthetic lệch manifold "chiếm" → FP trên benign tương lai.
  Các lớp này định nghĩa bằng quan hệ hàm giữa trường khai và chuyển động suy ra (resid = khai − suy ra); một Gaussian latent/lớp không giữ quan hệ đó.

**Transport benign→lớp c** (OT Gaussian latent từ dòng benign thật, 0 byte): judge .798 (+.017), lớp1 .851, lớp7 .842,
nhưng lớp5/11/13 xấu hơn và TV16 vs val .068 > .062. Bằng chứng yếu → **không pilot**.

## 4. Hệ quả cho bước tiếp

1. Sửa sinh dữ liệu chỉ thắng nếu giảm synthetic lệch nhãn (judge MLP) ở lớp 1/7/4/10/3 trên train→val;
   pair/forest/standard/transport đều chưa đạt điều này.
2. Cơ chế nhắm trực tiếp nút thắt: synthetic phải được gán nhãn/lọc bởi tri thức học từ dòng thật, theo input (không phải Q[y] theo lớp).
   Input-bank (ensemble private trên 1024 dòng) đã mất lợi r50; cần nguồn tri thức tổng quát hóa tốt hơn private đã overfit.
3. b/td là giới hạn của quyết định từng dòng ở model riêng; centralized cho thấy thông tin tồn tại. Hướng sender-fusion cần đổi protocol.

## 5. Cấu trúc dataset liên quan bộ sinh (train fold, `scripts/probe_attack_twins.py`)

highway_2, 400 xe, chỉ fold train → `runs/transport-probe/twins-highway_2.json`. Nguồn dựng: repo `mobihet-features` (LESSONS, feature-processing-v4).

- 15 world = mô phỏng lại cùng giao thông, mỗi world một loại tấn công; benign mỗi lần nhận lấy từ một world còn nguyên (benign/xe trải median 15 world).
- Mỗi tấn công sửa **một trường**; dòng tấn công có **twin benign** (cùng sender/message_id/rcv_time) trong cùng xe: lớp1/2/3/6/7/10/11/12/15 = 100%,
  4/5/8/9 ≈ 33–43%, timeDelay/Sybil = 0. Twin luôn ở world khác.
- So với twin, chỉ một họ feature đổi: lớp1/6 (acl) 6/77 feature, 3/9/11/15 (spd) 15–22, 10 (hed) 16, 2/7/8 (pos/map) 20–34, 12 suddenStop 53.
  Feature còn lại trùng tuyệt đối. Copula mỗi lớp sinh lại cả 77 chiều, nên phá đúng quan hệ tất định này.
- Phần không đổi vẫn mang ngữ cảnh **sender tấn công**, khác benign gộp: TV>.2 chọn 43 feature cho lớp1 (thật 6).

Hai thử nghiệm 0-byte đã **FAIL** trên judge (không pilot): sparse conditional từ benign local, tập S theo TV>.2 (đúng-nhãn .741 so copula .780)
và theo tập đổi thật từ twin (.667, lớp1 .508, TV16 .121). Lấy ngữ cảnh từ benign ngẫu nhiên sai phân phối ngữ cảnh sender tấn công.
Không mở lại transport/sparse-conditional thiếu cơ chế chọn đúng ngữ cảnh.

## 6. Proxy đã từng chạy

MobiHet-proxy gốc trên v4 (proxy DAGSNet trao delta 3-bit, mutual KD trên dòng thật, hệ số adaptive) đủ r50:
private/proxy macro .8953/.8995 (repo benchmark, notion-results), ≈ Core và dưới APFed .9119. Proxy mới phải khác cơ chế,
không chỉ đổi tên: lessons cho thấy KD dòng thật hại timeDelay/lớp vắng, adaptive weight kém fixed, Q8/3-bit làm giảm.

## 7. Ràng buộc owner 08/11

Không thêm compute tại xe: Core ≈6F/dòng thật (64 thật + 64 synthetic/bước, + floor, + sample copula), ngang APFed 5,97F, gấp đôi FedAvg 2,99F
(bảng cost repo benchmark). Proxy DAGSNet (P1 ensemble/P2 gán nhãn synthetic) **bị owner loại vì compute ×2**, chưa code/push.
Ứng viên sau phải ≤ compute Core, ưu tiên giảm.
