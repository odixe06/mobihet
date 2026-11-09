# Bài học kỹ thuật còn hiệu lực

Đọc sau [CONTEXT](CONTEXT.md); job sống ở [continue](docs/operations/continue.md), vận hành ở
[operations lessons](docs/operations/lessons.md). Bản đầy đủ theo batch đã lưu [snapshot](archive/docs-20261107/before/LESSONS.md);
không nạp lịch sử đó đầu phiên. Quyết định hiện hành ở [proposal](docs/research/proposal.md).

## Dữ liệu và đánh giá

- 77 đầu vào đã scaled, thứ tự/label mapping cố định; không normalize lại. Đọc parquet qua `ParquetFile` để tránh hive schema mismatch.
  Cờ NaN/quan hệ vật lý trên scaled values không được tự hiểu như raw values; projection đúng flag chưa chứng minh utility.
- Train FL là train+val local; own-test theo thời gian. Train score là in-sample, không independent validation.
- Personalized evaluation: mỗi xe chấm test của nó, gộp CM đủ support. Macro/xe chỉ tính lớp có trong y_true;
  pooled đủ16lớp. Luôn đọc p10/low256/rare/absent và từng lớp cùng pooled.
- Một epoch không cùng số bước giữa xe; floor chỉ synthetic vẫn giữ một epoch thật. Không bỏ tail batch/client để tăng tốc.
- Headline endpoint đã khóa; không tune/best-round bằng official test. Test/panelval đã đọc lặp, không gọi fresh confirmation;
  hai seed hoặc spread giữa session không là significance/CI.
- Thang scale: sim 2 seed → pilot full-scale r20 (cosine50, nối được) → r50 hai seed; PASS tự lên bậc kế ([CONTEXT](CONTEXT.md)),
  luật bậc kế khóa trước push; FAIL thì dừng nhánh, không kéo dài/tune. So control **cùng round** và audit trước quyết định.
- v1 khác feature/client/protocol; không chuyển gain/ngưỡng hoặc causal claim sang v4/5.343xe.

## Compile, số học và resume

- Compile gate phải kiểm backward trên real data, finite BN/weights/probabilities và applied/skipped steps, không chỉ forward/loss.
  `amax` maxpool từng tạo NaN backward AMP; dùng implementation đã kiểm. Fixture padded slots cũng phải finite.
  Smoke sim local dùng **200 xe**: 60 xe làm compile gate NaN ngay cả khi tắt mọi knob (fixture, không phải lỗi code).
- Slice batch `.contiguous()` để giữ stride/static train shape, tránh recompile/fallback eager. Local sm86 không chứng minh T4sm75/2rank.
- Mỗi client phải có applied step; skip global nhỏ vẫn có thể làm xe ít mẫu mất phần lớn bước. Guard≤1%/finite không phải sửa root cause overflow.
  Không gọi tất cả skip là warm-up hoặc bỏ client/update để qua gate.
- CPU/CUDA sum/normalization có rounding khác; packet fp16 có thể khuếch đại chênh. Kiểm stored body/aggregate và bound theo phép toán;
  không nới atol chung hoặc train lại vì auditor false FAIL.
- Resume cần cả private state/teacher, method/driver generator và torch CPU/CUDA RNG của **hai rank cùng round**.
  Snapshot nhỏ không phải checkpoint. Local optimizers reset mỗi round theo protocol.
- Crash sau ghi rows nhưng trước state: rollback rows về checkpoint, không nhân đôi history. Budget stop phải đồng thuận rank;
  `COMPLETE` thiếu round là partial, không training/utility FAIL.
- Bit-exact decode/local replay không suy ra whole T4 trajectory bit-exact. Giữ nguồn frozen và ghi rõ phạm vi từng proof.
- Lossless CDF: diff trên grid float64 rồi castfloat32 như control; diff float32 đã làm tròn từng gây lệch mass.

## Teacher, loss và thí nghiệm

- `Q[y_true]` là target theo lớp, không classifier theo input. Argmax đúng, entropy thấp, fit KL/synthetic agreement không chứng minh generalization.
  Q thu r chỉ tác động từ r+1; không gán tác động lệch round.
- CEreal dùng LA, KD dùng raw logit có chủ đích. Gradient logit âm/norm/cosine riêng không chứng minh parameter gradient tổng hoặc nhân quả utility.
- Giữ loss/outer weight/target/codec khi isolate một cơ chế. Không cộng gain các ablation để dự báo combo.
- Soft class Q có utility; nội dung off-diagonal có ý nghĩa trong replay. Không kết luận teacher vô ích vì gần nhãn cứng ở cuối run.
- realKD có tradeoff khác nhau theo lớp/cohort; lớp vắng và benign/timeDelay không được bảo vệ chỉ bằng gate lớp đúng/margin.
  Live contrast gồm feedback; muốn tách nội dung/tiếp nhận cần replay cùng teacher.
- Bank tốt ở r20 có thể chỉ tăng tốc; [F50](docs/research/results-f50.md) đã bỏ bank. Prior tăng pooled nhưng
  [Y50](docs/research/results-y50.md) vẫn FAIL do timeDelay/absent. Giữ verdict/gate, không tune sau test.
- Không gọi timeDelay giới hạn tuyệt đối bởi77feature: centralized vẫn tốt hơn. In-sample gap không tách được overfit và temporal shift.

## Generator (đã đóng băng KM16 w0 .5, codec gốc) — bài học còn dùng

- Oracle dùng pool dòng thật, chỉ là trần ngoài hợp đồng FL. Fidelity (pairTV, singletonTV, valTV, TSTR, ESS, probe LL) **không thay utility FL**:
  pilot20 standard/forest có TSTR/pairTV tốt vẫn own-test kém; chi tiết pair/forest/quantile ở archive.
- Probe LL chỉ là **bộ lọc loại**, so trong cùng họ nhiễu: LL tụt mạnh báo hại (K8 −4 nat → −.002), LL tăng không báo lợi
  (EM1 +6,7 / EM2 +9,6 / whitening +2,3 nat → −.0005/−.0011/−.0028). Codec lite6 kiểm lại ở full-scale r20 seed43 (LT6S43) dưới luật TB r16–r20
  **PASS vững** (Δmacro −.00005, p10 −.0008, absent −.0080); LT8 cũng PASS (−.0001) → xác nhận lite6 từng trượt ở r20 một round là do dao động nhiễu.
- Lợi KM nằm ở **tương quan riêng từng cụm** (tâm cụm + covariance chung ≈ +.000); phổ covariance cụm phẳng → low-rank không giảm byte.
- Sim 200 xe dự báo đúng chiều full-scale (+.0057 sim → +.0075/+.0069 pilot, +.0085/+.0086 r50). Sim b9/b10 (nhiễu seed ±.0006):
  K32, pop, part, EM, whitening, temp, benign share, gap-fill, Q synthetic đều ≤ 0 → **KM16 là trần cục bộ**; thêm dòng cho thống kê (pop) không giúp.

## Synthetic và teacher

- Synthetic là tri thức toàn mạng có ích cả ở xe lớn: bỏ synthetic (A256–A4096) hại đơn điệu; tăng trọng số synthetic (λ2/λ4) cũng hại.
  Twin phản thực trên họ trường chung (acl 1↔6, spd, pos) làm lẫn cặp lớp.
- Tỷ lệ/nhiệt synthetic chỉ **dời ngưỡng** benign↔timeDelay (benign 19%: timeDelay P +.138 R −.070 F1 −.045), không thêm tách biệt.
- Q phải đo trên **dòng thật**: Q từ synthetic làm suddenConstantSpeed R −.278 (→ zeroSpeedReport). Q và floor `needed` ở r50 (Y50).
- Ưu tiên synthetic cho lớp/cụm xe thiếu: absent +.084 nhưng macro −.0045 (FP lớp hiếm). Absent tăng không thành macro.
- Sim b11 (09/11, 2 seed, đều FAIL): Q mềm hơn (Q^½) −.004 macro, timeDelay −.009/−.014 → độ sắc Q đo trên dòng thật là đúng mức;
  Q theo (lớp, cụm KM) ≈ 0 → cùng K10 bucket8, **đóng họ "Q theo vùng input"**. EMA trọng số β .8: precision lớp tấn công/timeDelay tăng
  nhưng low256 −.004/−.037 (xe ít bước chậm theo) → EMA không miễn phí cho xe nhỏ.
- Giảm synthetic dòng/bước (G1 full-scale r20, 09/11, 48/32 dòng) **FAIL cả hai arm** (SR48 macro −.0016, low256 −.0118; SR32 macro −.0041,
  trượt 6/7 gate) → tín hiệu synthetic 64 dòng không bão hòa ở xe nhỏ; giữ nguyên 64 dòng, không cắt giảm dòng để tiết kiệm compute.

## Đọc guard và nút thắt

- Lấy mẫu 10% xe: macro −.0014 qua non-inferiority nhưng p10 −.028 → luôn đọc guard p10 cạnh macro.
- p10 ở sim 200 xe dao động ±.01–.03 giữa seed (b12) → gate p10 "tăng ≥ +.006 mỗi seed" gần như không qua được với 2 seed;
  liều compute cần ≥ 3–4 seed hoặc gate trên trung bình seed, khóa **trước** run.
- Liều compute (b12): floor synthetic-only có điểm gãy ≈48 bước (+8,2%: macro +.0013, absent +.065); thêm synthetic/bước kém hiệu quả gấp ~6×.
- suddenConstantSpeed (lớp ít nhất) là **canary** của mọi nhiễu loạn generator; đọc cùng timeDelay.
- Guard absent ít dòng: full-scale 1.555 dòng dao động ±.05 giữa hai round liền; sim ±.03 giữa arm không hiệu ứng → báo thêm trung bình nhiều round.
- Gap KM50 r50 → APFed (.9077 vs .9119): benign recall .850/.956, FP benign→tấn công ngoài timeDelay 101 k/45 k, precision lớp 7/11/6/1 thấp hơn;
  KM **hơn** APFed ở timeDelay. Nút thắt = tổng quát hóa benign tương lai của model riêng.
- FP benign→tấn công **không phải lệch prior/ngưỡng**: hiệu chỉnh prior local lúc suy luận (PH, b11) đổi precision benign ≤ .0006
  trong khi absent −.31/−.34 ở α 1. Mọi knob ngưỡng/prior/tỷ lệ (Y50 prior, SB, PH) chỉ đổi lớp vắng/timeDelay lấy pooled.

## Phiên sender (hướng gộp quyết định)

- Dataset trộn 15 world/xe nhận: benign mỗi lần nhận được rải ngẫu nhiên vào các world, tấn công nằm ở world của nó → phiên theo (xe, pseudonym)
  chỉ 16,8% dòng thuần nhãn (twin benign trộn tấn công); **phiên phải khóa (xe, world, pseudonym)** → 99,1% thuần, nhãn trùng message trước 99,7%.
  Không dùng `sender_id` thật (nối pseudonym Sybil = lộ nhãn). `ses_*`/`win_*` hiện tính theo (run, xe, alias) không có world.
- Gộp phiên (b12) chỉ +.0018 macro dù trần lớn: lỗi model tương quan trong phiên; cộng log-prob pha loãng tấn công gián đoạn
  (feignedBraking −.028) và lớp vắng (absent −.028). Gộp kiểu trung bình không đủ; cần quy tắc giữ bằng chứng tấn công cục bộ.
- v3-raw ↔ v4-scaled khớp vị trí 1:1 (cùng 410 file/split, cùng thứ tự dòng) → định danh lấy theo vị trí, không cần join.

## Communication và privacy

- Chỉ thống kê gộp/xác suất, không dòng/model/BN/gradient. Báo đủ bootstrap + hai chiều + Q; sparse uplink không là vector dày SecAgg.
- Moment theo ô của **một xe** không riêng tư khi ô nhỏ (510 ô xe×lớp 1 dòng; ô cụm KM nhỏ hơn): chỉ SecAgg che; chưa triển khai SecAgg/DP
  (owner 09/11 hoãn xử lý). Knob mới theo ô phải có ngưỡng count tối thiểu.
- Schema cho phép model khác nhau chưa là bằng chứng model trộn full-scale đã chạy.
- Byte KM50 (đọc lại code 09/11): uplink dense 256 ô + class moments fp32 16 lớp dù xe trung vị có 8 lớp; downlink gửi mean/Cholesky
  lớp không dùng khi bật KM; **đếm thiếu một lượt tâm** (iters lượt đếm, iters+1 lượt dùng); tâm mô phỏng fp64 nhưng tính fp16
  (đổi cụm 0,006 % dòng). Codec lossless (sparse + zlib, bỏ gói không dùng, CDF lossless) cắt ≈38 % bootstrap không cần train.
- Không tính bootstrap, Core/KM chỉ 1.088 B/round (54 KB/50r) < mọi phương pháp prototype; Q ở Core/KM stream nằm trong
  `Base._round_private` (`.half()` trong methods/mobihet.py), **không** qua `mobihet_study_wire.roundtrip` (chỉ biến thể bank/mixed):
  patch roundtrip không đổi gì — smoke phải đọc `bytes_up/down` để thấy knob có tác dụng.
- Codec study full-scale r20 (09/11): CH6 (Cholesky 6-bit), CM8 (moments int8), CT8 (tâm int8), LY0/LY2, LT8 và LT6S43 đều **PASS**.
  Q stream: `q_bits=8` (QQ8) **PASS** (giảm −71% byte mỗi round từ 1.088 B → 319 B), nhưng `q_bits=4` (QQ4) **FAIL** (macro −.0019, low256 −.0104)
  → giới hạn nén Q là 8-bit, 4-bit bị quantization noise làm hỏng xe nhỏ. Bootstrap nén xuống còn **1,15 MB** unicast (−68%) ở LT6.

## Giới hạn audit hiện tại

Audit source/config/LR/IDs/support/CM→10metric/snapshot/teacher/finite từ artifact thật. Dataset chưa có full-content fingerprint;
chưa prediction→CM replay/full private-bank reconstruction toàn mạng. Không dùng formula test trên random tensors thay kiểm stored values.
Các thất bại/runtime đã biết và nguồn kết quả: [research index](docs/research/README.md), [generator catalog](docs/research/generators.md).
