# Codec study — giảm băng thông KM16 (luật khóa 09/11/2026, trước push)

Owner 09/11: giảm băng thông (mỗi round và bootstrap) trước khi chốt phương pháp chính; thử hết các mức; codec mất mát lên thẳng
full-scale (sim 200 xe không thấy đuôi xe nhỏ/lớp vắng); luật xét **F1 toàn hệ thống**, guard lớp nhỏ được nới.
Mục tiêu owner (Proposal 2): hiệu năng top 3; băng thông **không tính bootstrap** vẫn hơn nhóm prototype; compute không tụt hạng.

## Băng thông hiện tại — KM16 w0 .5 int8 codec gốc (KM50, đo trong run)

| gói / xe | byte | ghi chú |
|---|---:|---|
| lên: counts | 64 | |
| lên: histogram biên (sparse, lossless) | 73.847 | Core |
| lên: class moments fp32 **dense** 16 lớp | 197.120 | Core; xe trung vị có 8/16 lớp |
| lên: 4 lượt Lloyd (count + mean int8, dense 256 ô) | 87.040 | |
| lên: lượt cuối (count + mean + cov int8, dense 256 ô) | 791.552 | xe trung bình chỉ có ~128 ô |
| **bootstrap lên** | **1.149.623** | |
| xuống: counts | 64 | |
| xuống: CDF fp16 | 630.784 | lossless zlib → 267.411 (L50) |
| xuống: mean + Cholesky lớp fp16 | 98.560 | **không dùng** khi bật KM (client chỉ sinh từ mixture) |
| xuống: tâm cụm 4 lượt fp16 | 157.696 | **đếm thiếu 1 lượt** (lượt cuối cũng cần tâm): đúng là 197.120 |
| xuống: mixture (p fp32, mean fp16, Cholesky fp16) | 1.577.984 | 256/256 cụm giữ |
| **bootstrap xuống** | **2.465.088** | |
| **mỗi round: Q lên + xuống** | **544 + 544** | header 32 B + 16×16 fp16; 50 round = 54.400 |
| **tổng 50 round** | **≈3,67 MB** | không tính bootstrap: **54 KB** (pFedES 0,14 MB, TinyProto 0,17 MB, FedProto 1,64 MB) |

KM50 mô phỏng tâm cụm chính xác (fp64) nhưng tính byte fp16; preflight 40 xe: làm tròn fp16 đổi cụm của **0,006 %** dòng → cách tính chấp nhận được.

## Preflight (local, không train)

`scripts/check_km_lite.py --n 40` PASS: knob tắt và knob tắt + `km_bytes` == source KM50 frozen bit-for-bit (`km_p/km_mu/km_L`);
mọi arm hữu hạn; CH8 uplink == KM50. Smoke `sim_fold_val.py` 200 xe 2 round (`q_bits=4 km_chol_bits=8 km_bytes`): compile PASS, skip 0,
Q **≈108 B lên / 100–111 B xuống mỗi round**. Bootstrap 40 xe (sim, chỉ để so thứ tự; số full-scale lấy từ run):

| cấu hình | lên | xuống unicast | tổng | so KM50 dense 3,61 MB |
|---|---:|---:|---:|---:|
| KM50 lossless (sparse + zlib, bỏ class downlink, CDF lossless) | 458 K | 1.794 K | 2,25 MB | −38 % **không mất mát** |
| CH8 (Cholesky int8) | 458 K | 1.093 K | 1,55 MB | −57 % |
| CH6 | 458 K | 950 K | 1,41 MB | −61 % |
| CM8 (class moments int8) | 315 K | 1.794 K | 2,11 MB | |
| CT8 (tâm int8) | 458 K | 1.713 K | 2,17 MB | |
| LY2 / LY0 (Lloyd 2 / 0 lượt) | 441 / 415 K | 1.725 / 1.655 K | 2,17 / 2,07 MB | đổi generator |
| lite8 (CH8+CM8+CT8) | 315 K | 1.012 K | 1,33 MB | |

Broadcast: gói xuống giống nhau ở mọi xe (trừ tâm theo lớp có mặt); một bản broadcast ≈ 1,0–1,8 MB cho cả vùng phủ RSU.

## Arm (full-scale 5.343 xe, cosine50 dừng r20, official own-test, `km_bytes=true` mọi arm)

`scripts/run_codec_pilot20.py` → `runs/codec-pilot20/batch.json`. Hp = đúng KM50 cùng seed + knob (assert trong launcher).

| arm | knob | đối chứng |
|---|---|---|
| CH8 / CH6 | `km_chol_bits=8` / `6` | KM50 Q42 r16–r20 |
| CM8 | `km_cls_q=int8` | KM50 Q42 |
| CT8 | `km_cen_q=int8` | KM50 Q42 |
| QQ8 / QQ4 | `q_bits=8` / `4` (byte **mỗi round**) | KM50 Q42 |
| LY2 / LY0 | `cop_kmeans_iters=2` / `0` | KM50 Q42 |
| LT8 | `cop_kmeans_lite=8` (combo) | KM50 Q42 |
| LT6S43 | `cop_kmeans_lite=6`, seed 43 (kiểm lại lite6 bằng run mới, luật mới) | KM50 Q43 |

## Luật (khóa trước push; `scripts/analyse_codec_pilot20.py`)

Δ = arm − KM50 cùng seed, ghép cặp; **trung bình r16–r20** (lý do: Δ ghép cặp một round dao động low256 sd≈.0035, absent sd≈.009 —
[km-lite](screening-km-lite.md)).

- **Chính:** Δf1_macro TB ≥ −.0015 **và** Δf1_macro tại r20 ≥ −.003.
- **Guard nới (owner 09/11):** Δp10 ≥ −.010, Δlow256 ≥ −.010, Δabsent ≥ −.03, ΔtimeDelay ≥ −.005, Δbenign ≥ −.001 (TB r16–r20).
- Audit trước luật: đủ round, hữu hạn, skip ≤ 1 %, CM support == đối chứng, hp == đối chứng + knob, overlay sha, có `KM_CODEC_BYTES`.

**Chọn:** mỗi khe chọn mức PASS rẻ nhất — Q: QQ4 > QQ8 > fp16; Cholesky: CH6 > CH8 > fp16; CM8, CT8, LY0 > LY2 > 4 lượt nếu PASS.
Không cộng gain. Bước kế (tự động theo thang; **owner 09/11 duyệt combo**): **một pilot combo** gồm mọi mức đã chọn, cùng luật r20; PASS → r50 seed 42 + 43
(luật như trên tại r46–r50 vs KM50 r50). Mọi codec lossless (sparse, zlib, bỏ class downlink, CDF lossless) áp ngay, không cần train.

## Kết quả — 09/11/2026 21:18 (10/10 kernel audit PASS, `runs/codec-pilot20/{audit,verdict-r20}.json`)

9/10 arm **PASS** luật khóa r20; chỉ **QQ4 FAIL** (trượt macro_mean và low256).

### Bảng delta so KM50 cùng seed (ghép cặp, trung bình r16–r20) và phán quyết

| arm | kết quả | Δmacro TB | Δmacro r20 | Δp10 | Δlow256 | ΔtimeDelay | Δbenign | Δabsent | lỗi gate |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|---|
| **CH8** | **PASS** | +0.0000 | +0.0000 | +0.0011 | +0.0002 | +0.0011 | +0.0000 | −0.0032 | qua |
| **CH6** | **PASS** | −0.0000 | −0.0001 | +0.0006 | −0.0014 | +0.0011 | −0.0000 | +0.0014 | qua |
| **CM8** | **PASS** | +0.0001 | −0.0003 | −0.0001 | −0.0024 | +0.0007 | +0.0000 | −0.0062 | qua |
| **CT8** | **PASS** | −0.0001 | −0.0002 | +0.0006 | −0.0012 | −0.0006 | −0.0001 | −0.0107 | qua |
| **QQ8** | **PASS** | −0.0006 | −0.0007 | −0.0019 | −0.0029 | +0.0002 | −0.0002 | −0.0180 | qua |
| **QQ4** | **FAIL** | **−0.0019** | −0.0018 | −0.0082 | **−0.0104** | −0.0050 | −0.0000 | −0.0175 | macro_mean, low256 |
| **LY2** | **PASS** | −0.0003 | −0.0004 | +0.0012 | −0.0015 | −0.0001 | −0.0004 | +0.0113 | qua |
| **LY0** | **PASS** | −0.0011 | −0.0014 | −0.0008 | −0.0011 | −0.0011 | −0.0006 | +0.0037 | qua |
| **LT8** | **PASS** | −0.0001 | −0.0002 | −0.0004 | −0.0007 | −0.0005 | +0.0001 | +0.0039 | qua |
| **LT6S43** | **PASS** | −0.0001 | −0.0001 | −0.0008 | −0.0011 | +0.0003 | −0.0000 | −0.0080 | qua |

### Bảng đo byte thực tế full-scale (`KM_CODEC_BYTES`)

| arm | uplink TB (B) | downlink unicast TB (B) | tổng unicast TB | broadcast (B) | p50 (B) | p90 (B) | round up+down (B) | flip fp16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CH8 | 434,531 | 1,075,295 | **1,509,826** | 1,111,024 | 1,478,656 | 1,835,008 | 544 + 544 | 0.0060% |
| CH6 | 434,531 | 928,798 | **1,363,329** | 964,527 | 1,335,296 | 1,687,552 | 544 + 544 | 0.0060% |
| CM8 | 299,127 | 1,841,640 | **2,140,768** | 1,877,369 | 2,097,152 | 2,437,120 | 544 + 544 | 0.0060% |
| CT8 | 434,540 | 1,764,048 | **2,198,587** | 1,782,840 | 2,162,688 | 2,506,752 | 544 + 544 | - |
| QQ8 | 434,531 | 1,840,800 | **2,275,331** | 1,876,529 | 2,244,608 | 2,600,960 | **155 + 164** | 0.0060% |
| QQ4 | 434,531 | 1,840,800 | **2,275,331** | 1,876,529 | 2,244,608 | 2,600,960 | **87 + 88** | 0.0060% |
| LY2 | 421,008 | 1,775,256 | **2,196,264** | 1,796,693 | 2,162,688 | 2,498,560 | 544 + 544 | 0.0057% |
| LY0 | 395,813 | 1,709,403 | **2,105,216** | 1,716,549 | 2,068,480 | 2,387,968 | 544 + 544 | 0.0055% |
| LT8 | 299,145 | 998,637 | **1,297,782** | 1,017,429 | 1,245,184 | 1,576,960 | 544 + 544 | - |
| LT6S43 | 299,145 | 852,133 | **1,151,278** | 870,925 | 1,097,728 | 1,429,504 | 544 + 544 | - |

### 10 metric tại r20 (5.343 xe, own-test, official macro)

| arm | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CH8 | 0.9154 | 0.9057 | 0.9154 | 0.9072 | 0.9012 | 0.9154 | 0.9154 | 0.9008 | 0.9154 | 0.9088 |
| CH6 | 0.9153 | 0.9055 | 0.9153 | 0.9071 | 0.9010 | 0.9153 | 0.9153 | 0.9006 | 0.9153 | 0.9088 |
| CM8 | 0.9152 | 0.9052 | 0.9152 | 0.9070 | 0.9011 | 0.9152 | 0.9152 | 0.9004 | 0.9152 | 0.9086 |
| CT8 | 0.9152 | 0.9052 | 0.9152 | 0.9070 | 0.9012 | 0.9152 | 0.9152 | 0.9005 | 0.9152 | 0.9086 |
| QQ8 | 0.9151 | 0.9048 | 0.9151 | 0.9070 | 0.9007 | 0.9151 | 0.9151 | 0.9000 | 0.9151 | 0.9085 |
| QQ4 | 0.9152 | 0.9049 | 0.9152 | 0.9068 | 0.8993 | 0.9152 | 0.9152 | 0.8990 | 0.9152 | 0.9081 |
| LY2 | 0.9148 | 0.9058 | 0.9148 | 0.9067 | 0.8999 | 0.9148 | 0.9148 | 0.9003 | 0.9148 | 0.9084 |
| LY0 | 0.9144 | 0.9052 | 0.9144 | 0.9060 | 0.8989 | 0.9144 | 0.9144 | 0.8993 | 0.9144 | 0.9077 |
| LT8 | 0.9153 | 0.9054 | 0.9153 | 0.9071 | 0.9011 | 0.9153 | 0.9153 | 0.9005 | 0.9153 | 0.9087 |
| LT6S43 | 0.9154 | 0.9059 | 0.9154 | 0.9072 | 0.9013 | 0.9154 | 0.9154 | 0.9009 | 0.9154 | 0.9089 |

### Đề xuất mức rẻ nhất cho combo kế tiếp
- **Cholesky:** CH6 (`km_chol_bits=6`, PASS, unicast 1,36 MB, broadcast 964 KB).
- **Q mỗi round:** QQ8 (`q_bits=8`, PASS, 155 B lên + 164 B xuống = 319 B/round so với 1.088 B gốc, giảm −71%). QQ4 FAIL trượt.
- **Class moments:** CM8 (`km_cls_q=int8`, PASS, uplink giảm 135 KB/xe).
- **Tâm cụm:** CT8 (`km_cen_q=int8`, PASS).
- **Lloyd iterations:** Giữ nguyên (4 lượt hoặc LY2).
- **Combo lite6 (LT6):** Kiểm lại trên seed 43 (LT6S43) PASS hoàn hảo (Δmacro −0.00005, mọi guard an toàn), bootstrap unicast xuống chỉ còn **1,15 MB** (so gốc 3,61 MB, giảm **−68%**). Kết hợp thêm QQ8 đưa chi phí mỗi round xuống **319 B/round** (so gốc 1.088 B/round, giảm **−71%**).

## Combo r20 — luật khóa 10/11/2026 trước push

Owner 10/11 chọn: **hai combo song song × seed 42/43** + **Core + QQ8** để so Core/KM cùng băng thông giảm.
`scripts/run_codec_combo20.py` → `runs/codec-combo20/batch.json`; analyser `scripts/analyse_codec_combo20.py`.
Full-scale 5.343 xe, cosine50 dừng r20, lưu state để nối r50; hp = control + knob (assert trong launcher).

| arm | knob | đối chứng |
|---|---|---|
| CBA42 / CBA43 | `cop_kmeans_lite=6` (CH6+CM8+CT8) + `q_bits=8` + `cop_kmeans_iters=0` + `km_bytes` — mức PASS rẻ nhất mọi khe | KM50 Q42 / Q43 |
| CBB42 / CBB43 | `cop_kmeans_lite=6` + `q_bits=8` + `km_bytes` (Lloyd 4 lượt; dự phòng nếu LY0 + QQ8 cộng hại) | KM50 Q42 / Q43 |
| C0 | Core (`mobihet_sim`, mọi knob tắt), seed 42 | F2 core s42 — **chỉ báo** (sanity đường sim = Core) |
| CQ8 | Core + `q_bits=8`, seed 42 | C0 (cùng batch) |

**Luật mỗi arm-seed** = luật codec r20 ở trên (Δ ghép cặp TB r16–r20: macro ≥ −.0015, macro r20 ≥ −.003; guard p10/low256 ≥ −.010,
absent ≥ −.03, timeDelay ≥ −.005, benign ≥ −.001). **Combo PASS = cả hai seed PASS.** Audit như codec study; arm Core không có `KM_CODEC_BYTES`.

**Bước kế (tự động):** CBA PASS → nối CBA42/43 lên r50 từ state r20 (luật như trên tại r46–r50 vs KM50 r50); CBB dừng (CBA rẻ hơn).
CBA FAIL, CBB PASS → nối CBB. Cả hai FAIL → dừng, báo owner (giữ codec từng khe đã PASS, không tune).
CQ8 PASS → nối CQ8 s42 lên r50 + push CQ8 s43 r50, so F2/F3 r50 nếu C0 ≈ F2 (|Δmacro TB| ≤ .0015); nếu không, nối cả C0 lên r50 làm đối chứng.
