# Sổ tay xây dựng phương pháp — kiểm 13/11/2026

Một trang cho phiên muốn **cải thiện macro-F1** hoặc **thiết kế phương pháp mới**: đang ở đâu, trần đo được,
đường học nói gì, họ cơ chế nào đã đóng, hướng nào còn mở và cần owner quyết gì. Chi tiết từng batch theo link;
số liệu ở đây tái tạo bằng `python scripts/analyse_saturation.py` → [analysis.json](../../runs/saturation-review-20261112/analysis.json)
và [bottleneck.txt](../../runs/transport-probe/bottleneck.txt). Không chọn HP/round bằng official test.

## 1. Vị trí hiện tại (r50, pooled own-test, 16 lớp)

| phương pháp | F1 macro | ghi chú |
|---|---:|---|
| centralized DAGSNet | .9593 | một model trên mọi dữ liệu; ngoài FL |
| oracle (synthetic = dòng thật pool) | .9176 | trần **kênh synthetic** khi generator hoàn hảo (seed42) |
| APFed (trao model) | .9130 | baseline mạnh nhất (TB 3 seed) |
| CANPerFL | .9097 | có fine-tune local sau FL (TB 3 seed) |
| **CBB** (TB 41/42/43) | **.9078 ± .0004** | phương pháp chính |
| FedAvg global | .9048 | TB 3 seed, hạng 4 |
| CQ8 = Core+QQ8 | .8991 | đối chứng không KM |

Ghép lý thuyết “lớp nào tốt hơn lấy lớp đó” giữa CBB và APFed/FedAvg ≈ **.918**: model toàn cục cho precision lớp tấn công
(lớp1/3/6/7 P .97–.98 vs CBB .84–.95), model riêng cho timeDelay (.35 vs .18–.27). Đây là ước lượng trên bảng số, không phải run.

## 2. Quá trình học của CBB — đã bão hòa chưa?

| cửa sổ round | ΔF1 macro (s42/s43) | ΣLR | ΔF1 / ΣLR | train loss s42 |
|---|---|---:|---|---|
| 10→20 | +.0086 / +.0088 | 7,96e-3 | 1,08 / 1,10 | .294→.234 |
| 20→30 | +.0043 / +.0048 | 5,05e-3 | 0,85 / 0,96 | .234→.195 |
| 30→40 | +.0025 / +.0021 | 2,14e-3 | 1,15 / 0,98 | .195→.171 |
| 40→50 | +.0007 / +.0008 | 3,85e-4 | 1,75 / 2,05 | .171→.166 |

- **Phẳng ở r50 là do lịch cosine (LR r40–50 ≈ 1e-4→1e-5), không do model hết học.** Lợi trên mỗi đơn vị LR không giảm
  (còn tăng ở pha anneal); train loss vẫn giảm. Kết luận cho tối ưu: **chưa bão hòa**, bị giới hạn bởi ngân sách tối ưu
  của protocol (50 round × 1 epoch, LR cố định).
- CBB−CQ8 **tăng dần** .0061→.0071→.0080→.0087→.0087 (s42), .0053→.0086 (s43): KM16 là tri thức thật, không chỉ tăng tốc
  (ngược bank F50, lợi r20 mất ở r50).
- r30→r50 mọi lớp tăng **trừ benign −.008/−.007**; lớp11 +.015, lớp1 +.010, TD +.007/+.005; p10 .758→.777.
  Train dài hơn tiếp tục đổi benign lấy lớp tấn công/TD — dấu hiệu model riêng học thuộc benign local.
- Diễn giải có giới hạn: hai seed, test đã đọc lặp; ngoại suy “thêm LR ⇒ thêm F1” chưa được kiểm, baseline cũng sẽ tăng nếu
  được thêm ngân sách.

## 3. Ngân sách sai số macro — đẩy F1 lên phải nhắm đâu

Thiếu hụt macro CBB r50 = 1 − .9078 = .0922, chia theo lớp `(1−F1_c)/16`:

| lớp | F1 | phần thiếu | tỷ trọng |
|---|---:|---:|---:|
| **timeDelay (13)** | .351 | .0405 | **44%** |
| benign (0) | .840 | .0100 | 11% |
| suddenConstantSpeed (11) | .848 | .0095 | 10% |
| positionMirroring (7) | .864 | .0085 | 9% |
| 12 lớp còn lại | ≥.925 | .0237 | 26% |

- TD .35→.50 ⇒ ≈ +.009 macro; bằng mọi tinh chỉnh lớp khác cộng lại. Centralized TD .616 ⇒ thông tin có trong 77 feature
  khi một model thấy đủ dữ liệu; oracle chỉ .376, FedAvg .177 ⇒ **không kênh FL nào hiện có học được TD**.
- V20: TD recall train .886 → val tương lai .267 ở xe lớn ⇒ model riêng **học thuộc sender TD local**, không học chữ ký TD chung.
- Benign/lớp 7/11: FP từ benign tương lai (train precision .99 vs test .84) — lỗi tổng quát hóa của model riêng.
- 59% lỗi đếm dòng là benign↔TD; mọi knob prior/ngưỡng/tỷ lệ chỉ **dời** lỗi giữa hai lớp này ([LESSONS](../../LESSONS.md)).

## 4. Họ cơ chế đã đóng (theo cơ chế, không theo tên)

Danh sách từng study và link doc gốc: [index phương pháp đã thử](tried-methods-index.md).

| họ | đã thử | lợi macro tốt nhất quan sát | vì sao đóng |
|---|---|---|---|
| generator / synthetic | standard, forest, pair, transport, EM, K32, whitening, temp, share, gap-fill, pop/part (b1–b10), G1 48/32 dòng, G2 | KM16 +.0085 (đã nhận) | KM16 là trần cục bộ sim; giảm dòng hại; [generators](generators.md) |
| liều compute synthetic | floor 24–64, rows 80–128 (b12) | +.0018 (R128, +51% compute) | dưới +.002; owner đóng 09/11 |
| nội dung teacher Q | QT2, QK/K10 (Q theo vùng), Q từ synthetic, SQ sqrt-count, GP/LG uplink, H/E nguồn ngoài fit (B18) | +.0009 | Q đo trên dòng thật ở độ sắc hiện tại là đúng mức; ngoài fit làm Q[TD] nhầm benign |
| dạng loss KD | RK/SK/JS/LS (B13), MK/HK mean-KD (B16), NR real-NCKD (B17), realKD W20 | trên CBB +.0016 (NR s43, s42 −.0002); W20 +.0025 chỉ trên recipe bank cũ | không ổn định qua seed; đổi TD lấy benign |
| prior / ngưỡng | PH b11, P prior-coordinate (B17 trên CBB), Y50 trên Core | Y50 Core full +.0033/+.0029; CBB sim −.0007/+.0002 | gain Core không chuyển sang CBB |
| tối ưu / regularize | SAM B15, EMA b11, MX interpolation B14 | +.0018 (EM8 s43) | không ổn định; hại xe nhỏ |
| input-bank / proxy | F50 bank, P20, proxy MobiHet | bank r20 +.0002 → r50 −.0016 | lợi sớm mất ở r50; proxy ≈ Core |
| gộp quyết định theo phiên | b12 fuse016 (suy luận) | **+.0020 / +.0023** cả hai control | macro qua mốc; loại vì guard p10 + owner đóng 09/11 |

Trong hợp đồng hiện tại (quyết định từng message, protocol 50r×1epoch, không trao model, chỉ thống kê theo lớp),
**hơn 40 biến thể nằm trong ±.002 của control** và trần đo được còn nhỏ (oracle +.010, cần generator mới mà sim đã chặn).
**Kết luận: hướng CBB trong hợp đồng hiện tại đã gần hết khả năng cải thiện đáng kể.** Dư địa lớn chỉ còn khi nới một trong
ba giới hạn ở §5.

## 5. Quyết định owner 12/11 và hướng còn lại

Owner đã trả lời ba hướng nới hợp đồng:
- **O1 gộp phiên (suy luận): giữ đóng.** IDS chấm từng message độc lập.
- **O2 ngân sách tối ưu: giữ protocol** 50 round × 1 epoch, cosine LR; không fine-tune sau FL, không thêm epoch.
- **O3 trao tham số: không.** Bắt buộc mỗi xe có thể dùng model **hoàn toàn khác**; chỉ trao thông tin không phụ thuộc kiến trúc
  (thống kê data-space, count, xác suất đầu ra trên input chung hoặc theo lớp). Không prototype hidden-space, không head/adapter chung.

Ba kênh không phụ thuộc kiến trúc đều đã khai thác: thống kê data-space → generator (KM16, trần oracle .9176);
xác suất theo lớp → Q (họ §4 đã đóng); xác suất trên input synthetic chung → bank P20/Q20/F50 (lợi r20 mất ở r50).
Ứng viên chưa thử còn lại đều **lợi kỳ vọng nhỏ**, phải qua diagnostic local trước khi tốn quota:

| id | ứng viên | khác họ đã đóng ở đâu | diagnostic local trước | dấu hiệu bác bỏ |
|---|---|---|---|---|
| A1 | nhãn mềm theo posterior của chính generator `p_gen(y|x)` cho từng dòng synthetic (0 byte; xe tự tính từ mixture/CDF đã có) | Q[y] mềm theo lớp, QK theo cụm; A1 mềm **theo dòng**, chỉ ở chỗ các lớp generator chồng nhau (TD↔benign) | Bayes-classifier của generator trên dòng thật fold val: accuracy/TD so với judge MLP .872; tỷ lệ dòng synthetic có posterior không one-hot | posterior gần one-hot toàn bộ hoặc Bayes-classifier TD < copula judge .42 |
| A2' | **cụm biên theo nhầm lẫn của generator** (thay A2 “K riêng benign” cố định; owner 12/11 muốn chọn lớp theo dữ liệu, không định sẵn benign) — **FAIL gate local 12/11** (Δmacro +.0030, TD↓) [B19](screening-boundary-b19.md) | K32/EM/temp tăng cụm hoặc LL đều lớp; A2' chỉ thêm thành phần cho cặp (lớp thật y, lớp generator đoán nhầm c') đo trên dòng thật, đúng chỗ ranh giới hỏng | Bayes-classifier generator (mid-rank) trên fold val: macro ≥ .890 (từ .880) **và** recall TD/benign/7/11 tăng | gate local trượt, hoặc sim Δmacro < +.002 một seed |

**A1 — luật diagnostic khóa 12/11 trước khi chạy** (`scripts/diag_gen_posterior.py`, sim 200 xe seed panel 7, generator CBB y hệt
bootstrap sim, không train, không official test). Mật độ lớp c: `log f_c(x) = log Σ_k p_ck N(z_c; μ_ck, L_ck L_ckᵀ) + Σ_j [log(m_cj/Δ) − log φ(z_cj)]`,
`z_c` = PIT chính xác theo CDF tuyến tính từng bin của lớp c, Δ = 8/256, mass sàn 1e-7. Prior đều 1/16 (synthetic cân bằng).
Target dự kiến từ r2: `t(x) = Σ_c p_gen(c|x) Q[c]`; r1 giữ hard CE như CBB. **Đi tiếp lên sim hai seed khi cả hai đúng:**
(i) ≥ 2% dòng synthetic có `p_gen(y|x) < .9` (posterior không one-hot); (ii) Bayes-classifier của generator trên dòng thật fold val
có recall TD ≥ .42 (mức judge MLP trên copula). Trượt một điều kiện → dừng A1, ghi kết quả.

**Kết quả A1 (12/11, [diag.json](../../runs/a1-gen-posterior/diag.json)): FAIL cả hai điều kiện → dừng, không lên sim.**
(i) chỉ **0,86%** dòng synthetic có `p_gen(y|x) < .9` (benign 7,1%, TD 6,3%, 14 lớp còn lại ≤ .3%): mixture các lớp gần như không
chồng nhau trong 77 chiều, nên target theo dòng gần trùng nhãn cứng. (ii) Bayes-classifier generator trên 211.118 dòng thật fold val:
recall TD **.36** (prior đều) / .31 (prior toàn mạng) < .42.
Số phụ: Bayes-classifier generator, không train, đạt macro **.873/.880** — dưới CBB sim r20 (.9146/.9129) ở gần mọi lớp
(benign P .81 vs .83; lớp 1/2/7 P .78–.80 vs .85–.95; TD F1 .30 vs .38): không dùng được làm teacher toàn cục tốt hơn model riêng.
Recall thật thấp nhất của generator đúng các lớp yếu của CBB: TD .31, benign .76, lớp7 .78, lớp11 .80, lớp2 .86.
Chấm dòng thật phải dùng **mid-rank** như lúc fit thống kê; PIT chính xác đặt giá trị atom ở mép bin → acc .02 (lỗi đo, không phải generator).

**A2' (đã chạy, FAIL local — chi tiết [B19](screening-boundary-b19.md)):** sau bootstrap KM16 như CBB, mỗi xe chấm một lần dòng train thật bằng Bayes-classifier
generator (mid-rank, prior toàn mạng); với dòng nhãn y bị đoán c' ≠ y, gửi count/Σz/Σzzᵀ theo cặp (y, c') trong không gian
normal-score của lớp y, codec int8 như lite, ô < 8 dòng/xe không gửi riêng. Server giữ cặp ≥ 20 dòng toàn mạng, thêm thành phần
vào mixture lớp y với trọng số đúng tỷ lệ count thật (không phóng đại lớp; khác “ưu tiên synthetic lớp thiếu” đã hại macro).
Số thành phần tăng do dữ liệu quyết định; diagnostic A1 dự báo tập trung TD/benign/7/11/2. Chi phí: bootstrap thêm một lượt
chấm dòng train + moment theo cặp (ước tính +30–90 KB/xe lên, broadcast xuống); mỗi round không đổi byte/compute.
Rủi ro đã biết: EM/whitening tăng LL nhưng hại utility; fidelity không thay utility; tăng mass vùng biên có thể chỉ dời ngưỡng.
**Đổi thông tin trao mỗi round để cải tiến distillation không đề xuất:** B13–B18 đã thử nội dung Q, dạng loss, nguồn ngoài fit;
tín hiệu đo trên train in-sample (precision .99) quá ít thông tin, đo ngoài fit thì đắt và B18 đã FAIL.

**Owner 12/11 sau B19:** thử cả N1 (teacher DAGSNet giống hệt model xe, như proxy ở `fl_dagsnet_veremi_v2`; cân nhắc **không train
proxy bằng CE** vì tốn compute) và N2 (GBDT liên kết data-space). **Luật khóa N2 trước khi chạy:** GBDT histogram với mọi xe tham gia
và bin chung cho đúng cây của HGB centralized ⇒ `scripts/diag_gbdt_ceiling.py` (HGB 200 iter, 31 lá, 255 bin trên 1,48M dòng train
200 xe sim) là trần. Chỉ tiếp N2 khi macro fold val ≥ **.924** (CBB sim C r20 .9146/.9129 + .010).

**N1 đối chiếu lịch sử (12/11, trước khi code):** owner chọn proxy train ở server, nhưng cơ chế này **đã chạy ở P20**
(C8 proxy_bank, C9 proxy_stream, C10 hybrid, C14 proxy_weights; [P20](../../archive/docs/research/results-p20.md) §3):
proxy DAGSNet W1 tại server distill từ ensemble model riêng trên bank synthetic có test F1 .790/.798 trong khi student .856/.855 ở r20;
C9 (xe query proxy trên synthetic mới) pooled −.021/−.020. Proxy học trên synthetic **yếu hơn model riêng trên dòng thật**.
Proxy train trên dòng thật cũng đã có: NILM (DML, CE cả hai) client r20 .876, MobiHet-proxy (DML 3-bit) r50 .8953/.8995 ≈ Core.
Real-row KD từ teacher (Q, NCKD) nhiều lần hại TD/lớp vắng. ⇒ N1 chưa có biến thể nào khác cơ chế đủ để launch; chờ owner.

**Kết quả N2 (12/11, [diag.json](../../runs/n2-gbdt-ceiling/diag.json)): FAIL luật khóa → dừng.** HGB centralized trên 1.476.064 dòng
train 200 xe sim (= GBDT liên kết đủ xe, bin chung): fold val macro **.8065** (acc .884) ≪ .924 và ≪ CBB sim .9146/.9129.
Train macro .818 ⇒ cấu hình đã khóa còn underfit, nên đây chưa chứng minh mọi GBDT đều kém. Nhưng một model **toàn cục** chấm kém ở lớp
phụ thuộc ngữ cảnh xe nhận (suddenStop .40, lớp11 .50, TD .20 so CBB riêng .99/.86/.38), và GBDT lớn hơn chỉ làm byte tăng thêm:
cấu hình này đã ≈ 16 cây × 30 split × 77 feature × 255 bin × (g,h) fp32 ≈ 75 MB/xe mỗi iteration nếu gửi dense, so CBB 13 KB/50 round.
⇒ N2 không tương thích băng thông hạn chế và chưa có tín hiệu utility.

Không mở lại họ ở §4 bằng tên mới; không cộng gain các arm để dự báo combo. Việc còn thiếu không thuộc F1 nhưng là yêu cầu bắt buộc:
**model trộn kiến trúc full-scale chưa từng chạy** (sim b4 W1/W0.25 không sụp). Seed 41 đã xong 13/11 (CBB .9077).
Giảm compute tại xe đang chạy ở [B20](screening-compute-b20.md).

## 6. Checklist trước khi khóa một thí nghiệm mới

1. Ghi **bằng chứng → cơ chế → lợi kỳ vọng → dấu hiệu bác bỏ**, chỉ ra họ ở §4 mà nó khác và khác ở đâu.
2. Gate macro ≥ +.002 từng seed, guard phụ report-only (owner 11/11); endpoint cố định; so control **cùng round, cùng lịch LR**.
3. Thang sim 200 xe cos20 hai seed → pilot full cos50 r20 → r50 hai seed. Sim đúng chiều với KM (+.0057 → +.0085) nhưng
   prior Core full +.003 không lặp lại ở CBB sim: không suy gain qua recipe khác.
4. Báo đủ 10 metric, byte (bootstrap + hai chiều), compute ×F; đọc toàn CM, đặc biệt benign↔TD.
5. Đọc [LESSONS](../../LESSONS.md) mục compile/resume và [operations lessons](../operations/lessons.md) trước khi sinh notebook.
