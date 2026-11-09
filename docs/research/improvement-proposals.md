# Đề xuất cải tiến ngoài generator — catalog 09/11/2026

Nguồn trạng thái duy nhất cho đề xuất **không thuộc generator** (generator: [catalog](generators.md), đã đóng băng KM16 w0 .5, codec gốc).
Mỗi dòng: bằng chứng → cơ chế → lợi kỳ vọng → dấu hiệu bác bỏ. Ràng buộc owner: compute tại xe ≤ Core, không trao model/proxy,
chỉ thống kê cộng được + xác suất; không chọn HP/round qua official test. Status job: [continue](../operations/continue.md). **09/11: sim b11 xong — PH, EM5, EM8, QK, QT2 đều FAIL (mục cuối).**

## Nút thắt cần nhắm (đọc lại CM r50 seed42 đã audit; `scripts/analyse_core_bottleneck.py`, LESSONS 09/11)

| r50 seed42 | Core | KM50 | APFed | oracle | central |
|---|---:|---:|---:|---:|---:|
| f1_macro | .8992 | .9077 | .9119 | .9176 | .9593 |
| benign P / R | .823/.859 | .828/.850 | .807/.956 | .844/.828 | .888/.956 |
| FP benign→tấn công (ngoài timeDelay) | 124 k | 101 k | 45 k | 83 k | 20 k |
| P posMirroring / suddenConstSpeed / feignedBraking / accMult | .813/.847/.919/.835 | .819/.867/.905/.929 | .900/.923/.969/.976 | .892/.846/.909/.926 | .972/.976/.987/.994 |
| timeDelay P / R | .415/.291 | .387/.318 | .616/.176 | .374/.378 | .758/.519 |

Đọc: KM đã lấy phần lớn lợi generator (accMult P .835→.929); gap còn lại tới APFed là **benign recall + precision lớp tấn công trên benign tương lai**
(train precision .99 vs own-test .84: model riêng học thuộc local). KM hơn APFed ở timeDelay. Sim b9/b10: mọi biến thể generator/synthetic ≤ 0.

## Đề xuất theo module

| id | module | kỹ thuật | bằng chứng → cơ chế | lợi kỳ vọng / bác bỏ | compute · byte | trạng thái |
|---|---|---|---|---|---|---|
| **PH** | suy luận | hiệu chỉnh prior local sau train: `z + α·la_shift` (la_shift = log π_local − log π_global, Laplace +1, đã có tại xe), α ∈ {0, .25, .5, .75, 1} chấm cùng một run | CE train dùng `z + la_shift` nên `z` mang posterior theo **prior toàn mạng** (benign 26%); dòng tương lai của xe theo prior local (non-IID mạnh; prior test ≈ prior train từng xe trừ Sybil 8,8→17,7%) → cộng lại shift = quyết định Bayes theo prior local | benign R ↑, P lớp 1/6/7/11 ↑ → macro ≥ +.002 ở α tốt nhất (fold val sim). Bác bỏ: không α nào ≥ +.002, hoặc absent/timeDelay trượt guard (rủi ro như Y50: lớp vắng nhận shift âm lớn) | 0 train · 0 byte; suy luận +1 phép cộng | FAIL b11 |
| **EM5 / EM8** | trạng thái model | trung bình động trọng số theo round tại xe `w̄ ← β·w̄ + (1−β)·w` (kể cả BN buffer), chấm own-test bằng `w̄`, train tiếp từ `w`; β .5 / .8 | train P .99 vs test .84; Core absent dao động .617→.662 giữa r19/r20; FedAvg/APFed mỗi round khởi đầu từ hàm toàn mạng nên ít FP → trung bình iterate làm phẳng nghiệm, giảm học thuộc lô cuối (SWA/Mean Teacher) | P lớp tấn công + p10 ↑, macro ≥ +.002. Bác bỏ: macro < +.002 hoặc timeDelay/benign trượt | ≈0 (1 axpy/round) · 0 byte; RAM xe +1 model (1,6 MB). Full-scale sim +4,2 GB (fp16) → cần probe RAM/state trước Kaggle | FAIL b11 |
| **QK** | teacher | Q theo **(lớp, cụm KM)** đo trên dòng thật: xe gán dòng thật vào cụm gần nhất của lớp đó **một lần** sau bootstrap (tâm đã có), gửi tổng xác suất theo (y,k); synthetic từ cụm k nhận `Q[y,k]`; ô (y,k) thiếu đếm → `Q[y]` | bottleneck §4.2: cần tri thức **phụ thuộc input**, Q[y] hằng theo lớp; QS 09/11: Q phải đo trên dòng thật. Cụm KM là vùng input cả hai đầu cùng biết → synthetic ở cụm gần benign nhận target mềm hơn | P lớp 1/6/7/11 ↑, macro ≥ +.002. Bác bỏ: không tăng precision các lớp đó, hoặc absent/c11 giảm | gán cụm 1 lần (≪ 1 epoch) · Q 16×16×16 int8 ≈ 4 KB/round/chiều (+0,4 MB/xe/50r; tổng ≈1,6 MB vẫn < KM50 3,67 MB) | FAIL b11, họ đóng |
| **QT2** | teacher | làm mềm Q: `Q_T ∝ Q^{1/2}` trước khi dùng làm target | Q đo in-sample quá tự tin (Q[13,13] .94 vs recall .29); soft Q có utility (LESSONS) → target gần nhầm lẫn ngoài mẫu hơn | macro ≥ +.002. Bác bỏ: macro/absent giảm (Q mềm có thể làm yếu bảo vệ lớp vắng) | 0 · 0 | FAIL b11 |
| RS | lịch train | xe ≥1024 dòng chỉ dùng 50% dòng thật/round (đổi mỗi round), giữ synthetic + floor | giảm compute; nhưng **b2 phản bằng chứng**: tăng tỷ trọng synthetic ở xe lớn (L2big/L4big) −.0028/−.0036, timeDelay giảm | không khuyến nghị; chỉ chạy nếu owner ưu tiên giảm compute | −≈30% compute · 0 | — |
| — | generator / synthetic | K/EM/whitening/share/temp/gap-fill/Q synth/pop/part | sim b9/b10 FAIL hết; owner 09/11 đóng băng | không mở lại bằng tên mới | — | đóng |
| — | giao tiếp | lite6 (1,20 MB/xe/50r) | sim PASS 2 seed; **pilot full-scale r20 FAIL** low256 −.0063, absent −.0186 | int4/PPCA đã loại (probe LL) | — | FAIL |

Ngoài hợp đồng (cần owner quyết riêng, không đưa vào Core): hợp nhất lịch sử theo sender cho timeDelay (đổi protocol);
EM prior trên test chưa nhãn (test-driven adaptation, trái hợp đồng đánh giá).

## Sim b11 — luật khóa 09/11/2026 (owner: "chạy thử hết tất cả", trước push)

Nền = LT6 (KM16 w0 .5 + lite6), 200 xe fold val, r20 cosine20, cap 2,75 h, như b9/b10. **10 kernel = 5 arm × seed 42/43 chạy cùng lúc**
(quota dư; tránh chờ vòng seed43): C/CS43 (đối chứng, `ph_grid=true` xuất thêm view ph025…ph100 — chỉ đánh giá, không đổi train),
EM5, EM8 (`w_ema=0.5/0.8`), QK (`q_cluster=true`, min 8 dòng/ô), QT2 (`q_temp=2`). Launcher `SIM_BATCH=b11 scripts/run_sim_kaggle.py`;
overlay `methods/mobihet_sim.py` sha `54cb5fa4…` (knob tắt = LT6: `check_km_lite.py` PASS; smoke local 200 xe 3 round gộp 4 knob: compile PASS,
skip 0, view cùng support). Verdict: `scripts/analyse_b11.py` → `runs/sim-kaggle-b11/verdict-r20.json`.

PASS từng seed khi Δmacro ≥ +.002 vs C cùng seed + guard chung (p10 ≥ −.006, timeDelay ≥ −.0035, benign ≥ −.0005, absent ≥ −.01 khi ≥2.000 dòng).
**Ứng viên full-scale = PASS cả hai seed → tự push pilot full-scale r20 seed42** (thang tự động ở [CONTEXT](../../CONTEXT.md)). PH: α* = macro cao nhất trong các α qua guard ở seed42; PASS khi α* PASS cả seed42 và seed43.
EM5 và EM8 cùng qua → giữ β có Δmacro trung bình hai seed cao hơn. Không cộng gain các arm để dự báo combo; không tune β/α/T ngoài lưới đã khóa.

**Quan hệ lịch sử của QK:** K10 (31/10, [archive](../../archive/docs/research/screening-k10.md)) đã thử Q theo lớp × 8 bucket siêu phẳng
ngẫu nhiên (`kd_bins=8`) và **FAIL** ("không hơn combo, tăng byte vô ích"). QK khác ở chỗ ô = cụm KM mà generator lấy mẫu
(synthetic cụm k nhận Q[y,k] đo trên dòng thật cùng cụm), không phải partition ngẫu nhiên; nếu QK cũng FAIL thì **đóng họ "Q theo vùng input"**.

## Privacy — dữ liệu local có bị lộ không (kiểm 09/11; **owner hoãn, không xử lý lúc này**)

- **Không arm b11 nào gửi dòng/model/gradient.** PH, EM5/EM8, QT2: 0 byte mới, mọi tính toán tại xe (EMA giữ trên xe; PH dùng prior local đã có).
  QK gửi thêm trung bình xác suất model int8 + count theo ô (lớp, cụm) — **đầu ra model, không phải feature**; ô < 8 dòng không gửi riêng
  (chỉ vào Q lớp) để không ô nào là thống kê của một dòng. Smoke: ≈83 ô/xe, +1,7 KB lên, +4,1 KB xuống mỗi round.
- **Rủi ro có sẵn từ trước (không do b11): bootstrap Core/KM gửi `Σz`, `Σzzᵀ` theo lớp (và theo cụm với KM) từng xe.** Đếm train+val:
  **510 ô (xe, lớp) chỉ có 1 dòng** (442 xe), 2.161 ô 1–4 dòng, 4.165 ô 1–7 dòng / 69.993 ô có dữ liệu. Với ô 1 dòng, `Σz` chính là normal score
  của dòng đó → đảo qua CDF chung ra feature (sai số trong bin). Server chỉ không thấy được khi có **secure aggregation** (chỉ thấy tổng toàn mạng) —
  hệ thống hiện **chưa triển khai SecAgg/DP**, nên tuyên bố "không lộ dòng" chỉ đúng dưới giả định SecAgg. Ô cụm KM còn nhỏ hơn ô lớp → nhiều ô 1 dòng hơn.
- Hướng giảm (cần owner chọn, chưa chạy): (a) ghi rõ giả định SecAgg trong bài; (b) ngưỡng tối thiểu m dòng/ô cho moment bootstrap (ô nhỏ gộp vào
  thống kê lớp/không gửi) — đổi thống kê, phải kiểm lại utility ở sim; (c) DP noise trên moment — đổi utility, cần ε cụ thể.

Nguồn kỹ thuật (đã ghi [papers](../../papers/README.md)): logit adjustment (Menon et al. 2021), SWA (Izmailov et al. 2018),
Mean Teacher EMA (Tarvainen & Valpola 2017), FD class-output exchange (Jeong et al. 2018), KD temperature (Hinton et al. 2015).
Chỉ mượn ý tưởng; công thức áp dụng ở trên là của dự án, chưa đối chiếu toàn văn.

## Kết quả sim b11 (09/11, 10/10 kernel audit PASS, `runs/sim-kaggle-b11/{audit,verdict-r20}.json`)

**Cả 5 arm FAIL luật r20; không ứng viên nào lên full-scale** (verdict cùng dấu ở r16/r18/r20). Nhánh b11 dừng, không tune β/α/T.

| arm | Δmacro s42 / s43 | Δp10 | Δlow256 | ΔtimeDelay | Δbenign | Δabsent | lỗi gate |
|---|---:|---:|---:|---:|---:|---:|---|
| EM5 | −.0001 / −.0000 | −.002 / +.006 | +.013 / −.007 | +.001 / +.003 | +.0002 / +.0008 | −.021 / +.011 | macro |
| EM8 | +.0004 / **+.0018** | −.008 / −.001 | −.004 / **−.037** | +.003 / **+.017** | +.0025 / +.0041 | −.031 / −.021 | macro (+p10 s42) |
| QK | −.0007 / +.0000 | +.007 / −.003 | −.008 / −.016 | +.001 / +.014 | −.0009 / +.0001 | −.011 / +.021 | macro (+benign s42) |
| QT2 | **−.0041 / −.0042** | −.005 / −.005 | −.010 / −.023 | **−.014 / −.009** | +.0003 / −.0005 | −.079 / −.058 | macro, timeDelay |
| PH α .25 (α* s42) | +.0002 / +.0002 | +.000 / +.001 | −.007 / −.011 | +.002 / +.001 | +.0001 / −.0001 | −.094 / −.047 | macro |
| PH α 1 | −.0002 / −.0013 | −.022 / −.028 | −.012 / −.029 | +.006 / +.005 | −.0012 / −.0014 | **−.314 / −.340** | macro, p10, benign, absent |

Đọc (không đổi verdict):
- **PH:** prior local lúc suy luận gần như không đổi precision benign (≤ .0006) nhưng giết lớp vắng (absent −.31/−.34 ở α 1) → FP benign→tấn công
  **không phải lệch prior/ngưỡng**; đúng rủi ro đã ghi (như Y50 prior).
- **EM8:** tăng precision lớp tấn công (lớp1 +.017/+.010) và timeDelay (+.003/+.017) nhưng hại xe nhỏ (low256 −.004/−.037, p10) →
  trung bình iterate có ích ở xe lớn, làm chậm xe ít bước. Không tune β sau số.
- **QK ≈ 0** cả hai seed → cùng K10 bucket8: **đóng họ "Q theo vùng input"**.
- **QT2 −.004** cả hai seed, timeDelay −.009/−.014 → Q mềm hơn hại; độ sắc của Q đo trên dòng thật là đúng mức (cùng bài học QS).

10 metric r20:

| arm | accuracy | P macro | P micro | P weighted | R macro | R micro | R weighted | F1 macro | F1 micro | F1 weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 0.9236 | 0.9243 | 0.9236 | 0.9169 | 0.9103 | 0.9236 | 0.9236 | 0.9144 | 0.9236 | 0.9175 |
| EM5 | 0.9235 | 0.9242 | 0.9235 | 0.9169 | 0.9102 | 0.9235 | 0.9235 | 0.9143 | 0.9235 | 0.9175 |
| EM8 | 0.9246 | 0.9278 | 0.9246 | 0.9184 | 0.9087 | 0.9246 | 0.9246 | 0.9147 | 0.9246 | 0.9182 |
| QK | 0.9229 | 0.9232 | 0.9229 | 0.9161 | 0.9095 | 0.9229 | 0.9229 | 0.9136 | 0.9229 | 0.9170 |
| QT2 | 0.9226 | 0.9244 | 0.9226 | 0.9157 | 0.9036 | 0.9226 | 0.9226 | 0.9103 | 0.9226 | 0.9158 |
| CS43 | 0.9230 | 0.9243 | 0.9230 | 0.9160 | 0.9095 | 0.9230 | 0.9230 | 0.9139 | 0.9230 | 0.9166 |
| EM5S43 | 0.9233 | 0.9242 | 0.9233 | 0.9163 | 0.9096 | 0.9233 | 0.9233 | 0.9138 | 0.9233 | 0.9170 |
| EM8S43 | 0.9251 | 0.9281 | 0.9251 | 0.9189 | 0.9100 | 0.9251 | 0.9251 | 0.9156 | 0.9251 | 0.9189 |
| QKS43 | 0.9232 | 0.9241 | 0.9232 | 0.9167 | 0.9093 | 0.9232 | 0.9232 | 0.9139 | 0.9232 | 0.9173 |
| QT2S43 | 0.9216 | 0.9225 | 0.9216 | 0.9144 | 0.9036 | 0.9216 | 0.9216 | 0.9096 | 0.9216 | 0.9148 |

## Hướng a + b — sim b12, luật khóa 09/11/2026 (owner: thử a và b; a chỉ ở lớp suy luận; b quét đủ các mức)

**Hướng a — gộp quyết định theo phiên sender (chỉ suy luận, 0 byte, không đổi train/input/generator).** Mỗi message vẫn được chấm
bằng đúng nhãn của nó trên đúng tập test; chỉ đổi quy tắc ra quyết định: message t của phiên P lấy argmax tổng log-prob của w message
gần nhất của P tới t (chỉ message đã nhận, theo thời gian). Phiên = **(xe, world, pseudonym `sender_alias`)**; không dùng `sender_id` thật
(sẽ nối các pseudonym Sybil = lộ nhãn). Bằng chứng dataset (fold val, `scripts/build_sessions.py`): khóa (xe, pseudonym) chỉ 16,8% dòng thuộc phiên
thuần nhãn (twin benign từ world khác trộn với tấn công), thêm world → 99,1%, nhãn trùng message trước 99,7%; 62% dòng có lịch sử phiên.
Sidecar `cache/sessions/{train,test}_{sess,t,y,off}.npy` khớp 1:1 thứ tự loader (v3-raw ↔ v4 cùng 410 file, cùng thứ tự dòng, assert từng file);
sim dùng `cache/sessions/sim_val200.npz` (overlay, 1 MB; lúc chạy assert nhãn từng dòng khớp env.test).
**So baseline:** baseline hiện chấm từng message độc lập → kết quả fused chỉ so nội bộ (MobiHet fused vs per-message cùng model);
muốn so APFed/FedAvg phải áp cùng lớp gộp cho baseline trước khi tuyên bố.

**Hướng b — compute tối thiểu, quét liều.** Floor synthetic-only (xe nhỏ): 24/32/48/64 = +1,7/+3,7/+8,2/+13,1% compute so Core;
synthetic/bước 80/96/112/128 (cùng trọng số, cân bằng lớp) = +12,7/+25,5/+38,2/+51,0%.

Batch: 9 arm × seed 42/43 = 18 kernel (C = LT6 + view gộp; F24…F64; R80…R128), r20 cosine20, 200 xe fold val, cap 2,75 h, view gộp trên mọi arm.
Overlay `methods/mobihet_sim.py` sha `132b65d8…` + sidecar gọn `sim_val200.npz` sha `ac7b2500…` (sess int32, rank thời gian trong phiên, nhãn int8 để assert từng rank;
knob tắt = LT6, `check_km_lite.py` PASS), smoke 200 xe gộp F24+R80+fuse: compile PASS. **Push 09/11 ~11:20; xong 12:40 — FAIL hết theo luật (mục kết quả b12).**
Verdict `scripts/analyse_b12.py` → `runs/sim-kaggle-b12/verdict-r20.json`. Guard chung: ΔtimeDelay ≥ −.0035, Δbenign ≥ −.0005, Δabsent ≥ −.01 (≥2.000 dòng).

- **Gộp (a):** view fuse vs "private" cùng run: Δmacro ≥ +.002, Δp10 ≥ −.006 + guard; w* = cửa sổ macro cao nhất trong các cửa sổ qua guard ở C s42; PASS khi w* qua cả C và CS43.
- **Floor:** vs C cùng seed: Δp10 ≥ +.006 và Δmacro ≥ −.0015 + guard. **Synthetic/bước:** Δmacro ≥ +.002, Δp10 ≥ −.006 + guard.
- Ứng viên = PASS cả hai seed. Chọn mức **compute thấp nhất** PASS; mức cao hơn chỉ thay khi gain trung bình hai seed hơn ≥ +.006 p10 (floor) / ≥ +.001 macro (synthetic).
  Báo đủ đường liều: Δmacro, Δp10, Δlow256, Δabsent theo % compute. PASS → thang tự động (CONTEXT): pilot full-scale r20 (gộp cần sidecar test lên Kaggle).

## Kết quả sim b12 (09/11 12:40, 18/18 kernel audit PASS, `runs/sim-kaggle-b12/{audit,verdict-r20}.json`)

**Theo luật khóa: không ứng viên nào PASS cả hai seed → không lên full-scale** (floor, synthetic/bước, gộp phiên). Không đổi luật sau số.

Đường liều (trung bình hai seed, Δ so C cùng seed; PASS từng seed s42,s43):

| arm | compute | Δmacro | Δp10 | Δlow256 | ΔtimeDelay | Δabsent | PASS |
|---|---:|---:|---:|---:|---:|---:|---|
| F24 | +1,7% | +.0001 | −.0047 | −.0254 | +.0025 | +.029 | –,– |
| F32 | +3,7% | +.0004 | +.0031 | −.0156 | +.0006 | +.034 | –,✓ |
| **F48** | **+8,2%** | **+.0013** | **+.0078** | −.0097 | +.0023 | **+.065** | –,✓ |
| F64 | +13,1% | +.0016 | +.0055 | −.0010 | +.0089 | +.060 | ✓,– |
| R80 | +12,7% | +.0004 | −.0007 | −.0169 | +.0026 | +.010 | –,– |
| R96 | +25,5% | +.0010 | +.0096 | −.0068 | +.0072 | +.026 | –,– |
| R112 | +38,2% | +.0009 | +.0017 | −.0182 | +.0022 | +.026 | –,✓ |
| R128 | +51,0% | +.0018 | −.0017 | −.0129 | +.0059 | +.034 | –,✓ |

- **Floor:** macro và absent tăng gần đơn điệu theo liều; **điểm gãy ≈ F48** (+8,2% compute: macro +.0013, p10 +.0078, absent +.065);
  F64 chỉ thêm +.0003 macro cho +4,9% compute. Không mức nào qua cả hai seed vì p10 sim dao động ±.01–.03 giữa seed (vd R80 −.018/+.017).
- **Synthetic/bước:** hiệu quả thấp — R128 (+51%) ≈ +.0018 macro, bằng F48 ở 1/6 compute. Không phải đòn bẩy đáng giá.
- **Gộp phiên (view trên 18 run, cùng model):** fuse016 Δmacro **+.0018 trung bình, thấp nhất +.0008 — dương ở 18/18 run**; benign +.0023,
  positionMirroring +.025, nhưng **feignedBraking −.028** (tấn công gián đoạn bị pha loãng bởi message trông bình thường) và absent −.028.
  C s42 trượt guard p10/low256 → w* không có → FAIL. Lợi nhỏ hơn trần nhiều: lỗi model **tương quan trong phiên** (cùng sender, cùng kiểu sai).

10 metric r20 (view "private" = per-message; hai dòng cuối = gộp fuse016 của control):

| arm | accuracy | precision_macro | precision_micro | precision_weighted | recall_macro | recall_micro | recall_weighted | f1_macro | f1_micro | f1_weighted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 0.9229 | 0.9240 | 0.9229 | 0.9162 | 0.9100 | 0.9229 | 0.9229 | 0.9145 | 0.9229 | 0.9172 |
| F24 | 0.9228 | 0.9243 | 0.9228 | 0.9160 | 0.9101 | 0.9228 | 0.9228 | 0.9144 | 0.9228 | 0.9168 |
| F32 | 0.9232 | 0.9238 | 0.9232 | 0.9163 | 0.9108 | 0.9232 | 0.9232 | 0.9144 | 0.9232 | 0.9171 |
| F48 | 0.9237 | 0.9258 | 0.9237 | 0.9169 | 0.9109 | 0.9237 | 0.9237 | 0.9153 | 0.9237 | 0.9175 |
| F64 | 0.9244 | 0.9267 | 0.9244 | 0.9180 | 0.9117 | 0.9244 | 0.9244 | 0.9162 | 0.9244 | 0.9185 |
| R80 | 0.9228 | 0.9245 | 0.9228 | 0.9159 | 0.9099 | 0.9228 | 0.9228 | 0.9144 | 0.9228 | 0.9168 |
| R96 | 0.9233 | 0.9246 | 0.9233 | 0.9166 | 0.9110 | 0.9233 | 0.9233 | 0.9152 | 0.9233 | 0.9176 |
| R112 | 0.9224 | 0.9233 | 0.9224 | 0.9155 | 0.9098 | 0.9224 | 0.9224 | 0.9138 | 0.9224 | 0.9165 |
| R128 | 0.9233 | 0.9248 | 0.9233 | 0.9167 | 0.9115 | 0.9233 | 0.9233 | 0.9156 | 0.9233 | 0.9176 |
| CS43 | 0.9227 | 0.9243 | 0.9227 | 0.9156 | 0.9085 | 0.9227 | 0.9227 | 0.9134 | 0.9227 | 0.9164 |
| F24S43 | 0.9229 | 0.9240 | 0.9229 | 0.9161 | 0.9094 | 0.9229 | 0.9229 | 0.9138 | 0.9229 | 0.9168 |
| F32S43 | 0.9231 | 0.9244 | 0.9231 | 0.9161 | 0.9099 | 0.9231 | 0.9231 | 0.9142 | 0.9231 | 0.9169 |
| F48S43 | 0.9234 | 0.9257 | 0.9234 | 0.9165 | 0.9105 | 0.9234 | 0.9234 | 0.9151 | 0.9234 | 0.9173 |
| F64S43 | 0.9234 | 0.9252 | 0.9234 | 0.9166 | 0.9101 | 0.9234 | 0.9234 | 0.9148 | 0.9234 | 0.9174 |
| R80S43 | 0.9229 | 0.9239 | 0.9229 | 0.9161 | 0.9097 | 0.9229 | 0.9229 | 0.9142 | 0.9229 | 0.9170 |
| R96S43 | 0.9231 | 0.9247 | 0.9231 | 0.9164 | 0.9100 | 0.9231 | 0.9231 | 0.9147 | 0.9231 | 0.9173 |
| R112S43 | 0.9234 | 0.9253 | 0.9234 | 0.9166 | 0.9116 | 0.9234 | 0.9234 | 0.9159 | 0.9234 | 0.9176 |
| R128S43 | 0.9239 | 0.9258 | 0.9239 | 0.9171 | 0.9117 | 0.9239 | 0.9239 | 0.9159 | 0.9239 | 0.9179 |
| C:fuse016 | 0.9262 | 0.9308 | 0.9262 | 0.9198 | 0.9091 | 0.9262 | 0.9262 | 0.9165 | 0.9262 | 0.9199 |
| CS43:fuse016 | 0.9261 | 0.9313 | 0.9261 | 0.9198 | 0.9080 | 0.9261 | 0.9261 | 0.9157 | 0.9261 | 0.9195 |

## Đóng hướng (owner 09/11 ~14:00)

- **Hướng b (tăng compute: floor/synthetic rows):** đóng. Không lặp seed F48/F64, không đổi luật b12.
- **Hướng a (gộp quyết định theo phiên sender):** đóng. Không thử quy tắc gộp khác.

## Generator với thông tin mới (09/11 ~16:00, đề xuất — chưa chạy, chờ owner)

Thông tin mới: (1) codec lite6 lệch macro ≤ .0004 ở mọi round r10–r20 → **độ chính xác** mixture không phải nút thắt; muốn tăng hiệu năng
phải thêm **thông tin mới**, không phải bit; (2) gói xuống giống nhau ở mọi xe → nếu tính broadcast, mixture lớn hơn gần như miễn phí byte,
nhưng K32 (kể cả thống kê toàn mạng, b9 PK32) đã FAIL → không có bằng chứng tăng K giúp; (3) b12: thêm synthetic/bước lợi rất ít
(R80–R128: +.0004…+.0018 cho +13–51 % compute) → 64 dòng/bước có thể đã thừa.

| # | đề xuất | bằng chứng → cơ chế | dấu hiệu bác bỏ | chi phí |
|---|---|---|---|---|
| G1 | **giảm synthetic 64 → 48 / 32 dòng/bước** (giảm compute, mục tiêu hạng compute) | đường liều b12 phẳng quanh 64 → tín hiệu synthetic bão hòa | full-scale r20: Δmacro TB r16–r20 < −.0015 hoặc guard nới trượt | compute ≈ −12,7 % / −25,5 % (4,00 → ~3,5×F) |
| G2 (**bỏ**, owner 09/11) | K riêng cho benign (vd 32–48, lớp khác 16) | gap còn lại = benign recall + FP lớp 7/6/1 từ benign; benign đa dạng nhất | benign F1 và precision lớp 7/6/1 không tăng ở full-scale | byte xuống tăng (broadcast ≈ miễn phí); ưu tiên thấp vì K32 FAIL |

Khuyến nghị: G1 trước (khớp ghi chú Proposal 2 "cân nhắc sinh synthetic mỗi round" và mục tiêu compute không tụt hạng). G2 chỉ khi owner muốn
thêm thử generator. Không đề xuất thêm bit/độ chính xác cho generator.

### Owner 09/11 ~16:30: **G1 chạy thử, G2 bỏ**, combo codec được duyệt

**G1 — luật khóa trước push** (`scripts/run_g1_pilot20.py` → `runs/g1-pilot20/`; `scripts/analyse_codec_pilot20.py 20 g1-pilot20`).
Arm SR48 / SR32 = KM50 Q42 hp + `synth_rows_sim=48` / `32` (+ `km_bytes=true`, chỉ đếm byte), full-scale 5.343 xe, cosine50 dừng r20,
seed 42, so KM50 Q42 ghép cặp, **cùng luật codec** ([screening-codec](screening-codec.md)): Δf1_macro TB r16–r20 ≥ −.0015 và tại r20 ≥ −.003;
guard nới TB r16–r20 Δp10 ≥ −.010, Δlow256 ≥ −.010, Δabsent ≥ −.03, ΔtimeDelay ≥ −.005, Δbenign ≥ −.001.
Compute báo bằng `t_train` TB r2–r20 so KM50 (cùng phần cứng 2×T4) và tỷ lệ dòng/bước (real 64 + synthetic).
Chọn mức PASS rẻ nhất (SR32 > SR48). PASS → thang tự động: nối r50 + seed43 r50 (luật như trên tại r46–r50).
Kết hợp với codec PASS: chỉ sau khi cả hai nhánh có verdict, bằng run combo riêng.

### Kết quả G1 (09/11 21:18, 2/2 kernel audit PASS, `runs/g1-pilot20/{audit,verdict-r20}.json`)

**Cả 2 arm FAIL; nhánh G1 dừng.**

| arm | kết quả | Δmacro TB | Δmacro r20 | Δp10 | Δlow256 | ΔtimeDelay | Δbenign | Δabsent | compute (`t_train` so KM50) | lỗi gate |
|---|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| **SR48** | **FAIL** | **−0.0016** | −0.0019 | −0.0084 | **−0.0118** | −0.0019 | −0.0005 | −0.0111 | x0.923 (−7,7%) | macro_mean, low256 |
| **SR32** | **FAIL** | **−0.0041** | **−0.0041** | **−0.0269** | **−0.0304** | **−0.0071** | −0.0008 | **−0.0320** | x0.834 (−16,6%) | trượt 6/7 gate |

10 metric tại r20:
- SR48: accuracy 0.9145, precision_macro 0.9039, recall_macro 0.8994, f1_macro 0.8988, f1_weighted 0.9078
- SR32: accuracy 0.9138, precision_macro 0.9024, recall_macro 0.8969, f1_macro 0.8966, f1_weighted 0.9067

**Kết luận cơ chế G1:** Giảm dòng synthetic (48 hoặc 32) làm giảm lượng tín hiệu đại diện toàn mạng cho các xe ít mẫu và các lớp hiếm (`low256` sụt mạnh −.012 và −.030; `absent` −.032 ở SR32). Trực giác "synthetic bão hòa ở 64 dòng" bị bác bỏ ở quy mô full-scale. Không giảm dòng synthetic; giữ 64 dòng. Nhánh G1 dừng, không kéo dài hay tune.
