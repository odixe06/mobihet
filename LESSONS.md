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
- B14 hybrid LG (11/11): ô lớn là fp16 log-score, ô<8dòng là Q8→fp32 log/center. Không áp bound LG thuần cho cả hybrid:
  dùng tổng bound từng phép toán theo tỷ trọng count/lớp. Replay GPU fallback trên artifact thật khớp aggregate đúng0/r1+r2;
  CPU lệch≈1e-9 không là training FAIL. Giữ strict bound tại lớp không fallback; regression phải còn bắt aggregate corrupt.
- Resume cần cả private state/teacher, method/driver generator và torch CPU/CUDA RNG của **hai rank cùng round**.
  Snapshot nhỏ không phải checkpoint. Local optimizers reset mỗi round theo protocol.
- Crash sau ghi rows nhưng trước state: rollback rows về checkpoint, không nhân đôi history. Budget stop phải đồng thuận rank;
  `COMPLETE` thiếu round là partial, không training/utility FAIL.
- Bit-exact decode/local replay không suy ra whole T4 trajectory bit-exact. Giữ nguồn frozen và ghi rõ phạm vi từng proof.
- Lossless CDF: diff trên grid float64 rồi castfloat32 như control; diff float32 đã làm tròn từng gây lệch mass.

## Teacher, loss và thí nghiệm

- Owner11/11: SAM hiện tiếp tục thử, nhưng phải báo **CBB+SAM** riêng/compute; gain không tự chứng minh cơ chế FL tốt hơn.
  So cải tiến truyềntri thức dùng controlCBB khôngSAM. Sampler/optimizer/pretraining generic cũng cần phân biệt tương tự.
- [B15SAM](docs/research/screening-sam-b15.md) đãđóng: audit4/4r20PASS, utilityFAIL cả2seed.
  Macro+.00053/+.00120 dưới+.002; p10−.0071/−.0102, low256−.0071/−.0198, TD−.0069/−.0014,
  dùbenignF1+.00545/+.00505 vàtrainelapsed+58%. TDprecision tăng nhưng recall giảm; accuracy/weightedF1 tăng không đủ.
  Không pilot/tune/extend; đây là verdict adaptationrho.05, không proof mọiSAM vôích.
- ClassQ là moment1 theo realtrueclass; meanKL từngsyntheticrow là ràng buộc mạnh hơn KL vào meanprediction nhóm.
  Jensen/toycounterexample chỉ phân biệt objective, không chứng minh utility. Betweenclientvariance có thể tính từ Qclient sẵn;
  withinclient/per-rowvariance cần thêm moment2, không khôi phục được từ Qmean hoặc meanlog.
- Cross-fit rowdisjoint không tự là time/eventindependent: attack–benign twin quaworld chia sẻ ngữ cảnh.
  Coverage99,8% theo countsplit cân bằng không chuyển sang groupedmessage/blocktime split nếu chưa đếm lại.
  [Ràtri thức11/11](archive/docs-20261113/research/knowledge-transfer-review-20261111.md) giữ nguồn/code/math và trạng thái chưautility.
- Rà 10/11: sqrt-count Q **đã chạy** trên recipe M20 cũ (+.00148/+.00116 macro, benign giảm), postepoch Q không tăng;
  synthetic hard+NCKD đã có tradeoff W20. Không gọi chúng là ý tưởng chưa thử hoặc suy gain sang CBB.
  [Rà module](archive/docs-20261113/research/pipeline-module-review-20261110.md) chỉ rõ evidence và đường assert/codec; b13 là adaptation có control mới.
- [B13](docs/research/screening-module-b13.md) trên CBB (11/11, 2seed): SQ, reverse/symmetric KL, 4JS, CEreal smoothing .05 đều FAIL.
  Reverse KL giảm FP benign nhưng td −.025/−.020, p10 −.057/−.026; symmetric KL cũng đổi td lấy benign.
  4JS benign+td tăng cả2seed nhưng macro chỉ+.0006/+.0008, không qua gate; không tune/kéo dài nhánh FAIL.
  Gain sqrt-count recipe cũ không chuyển sang CBB. Absent sim chỉ191dòng, không dùng như guard chắc.
- [B14](docs/research/screening-uplink-b14.md), 8/8run audit đủr20: same-class interpolation .5 (MX), geometric pool meanprob (GP),
  mean centered logprob (LG) **FAIL cả2seed**. MX macro−.01581/−.01588, lớp11 precision−.232/−.244, hại cả xe lớn;
  cùng nhãn không bảo đảm nội suy77feature giữ ranh giới. Không quy riêng nguyên nhân cho flag/BN từ CM.
  GP/LG thêm byte nhưng macro chỉ−.00031/+.00088 và+.00083/−.00020; LG teacher r19 diagonal≈.994,
  239/256ô Q8zero mà không đủ gain. Confidence/meanlog không là reliability ngoài fit; không tune/kéo dài nhánhFAIL.
  [Phân tích](archive/docs-20261113/research/b14-analysis-20261111.md) giữ nguồn CM/cohort/teacher và ưu tiên kế tiếp.
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

## Generator (đóng băng KM16 w0 .5; trong CBB cùng lite6 + QQ8) — bài học còn dùng

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
- Sim b14 chỉ9xe low256; control cùng seed b13→b14 low256 thay+.0176/−.0054 dù macro chỉ−.00035/+.00028.
  Seed ghép cặp không bảo đảm wholetrajectory bit-exact; spread mô tả giữa session không là CI.
  R1 GP/LG đã khác C khi chưa tiêu thụ teacher: không gán toàn bộ chênh trajectory sớm cho aggregation.
- FLOPs CBB (đo 12/11, FlopCounterMode, quy ước `fl_dagsnet_veremi_v2/scripts/cost.py`: FLOPs client / dòng thật / round): mỗi bước
  fwd+bwd 64 thật + 64 synthetic = 5,97×F/dòng thật; floor16 thêm 3,5% bước (607.110 vs 586.405) ⇒ toàn mạng **≈6,09×F ≈ 27 MFLOP**, không phải 4,00×F.
  Q tái dùng logit train (0 FLOP); bootstrap KM/Lloyd + sinh synthetic < 0,01×F. Synthetic chỉ train head (backbone no_grad) ≈4,16×F.
- B20 (12/11, sim 2 seed): synthetic chỉ train head (backbone no_grad) **FAIL** macro −.016, absent −.21/−.25 ⇒ tri thức lớp vắng
  phải vào backbone. 16/64 dòng full + 48 head-only (SM) −.002/−.004; 32 dòng từ nửa lịch (SD) PASS −.0006/−.0011.
  FLOPs giảm ≠ wall time ở sim 200 xe (t_train SD = C): báo FLOPs đo, không suy từ t_train.
  Full-scale 5.343 xe cũng vậy: SM42P −24% FLOPs nhưng t_train 3,57 h vs CBB 3,61 h (−1%) — T4 launch-bound; lợi FLOPs chỉ thành
  thời gian trên thiết bị compute-bound (CPU/edge), chưa đo. Knob đếm dòng (SD) không phụ thuộc kiến trúc; head-only (SM/SH) cần tách head riêng từng model.
- SD full-scale r50 (13/11, audit PASS hai seed): macro .9072/.9081 vs CBB .9074/.9081 (Δ −.0002/−.0001) ở 5,32×F; p10 −.0068/−.0041
  (report-only); `t_train` −4%/−10%. Nửa sau lịch cosine thừa một nửa dòng synthetic; nửa đầu thì không (G1).
- Cohort theo kích thước xe (13/11, `scripts/analyse_b20_cohort.py` → [cohort.txt](runs/b20-cohort/cohort.txt), CM full-scale; SM42 r50 đọc
  trước audit batch): 600 xe ≥ 16.384 dòng giữ 65% dòng thật / 63% bước synthetic; 3.820 xe < 4.096 dòng chỉ 14,6% bước.
  Mất macro của mọi knob giảm synthetic **dốc theo kích thước**: macro riêng cohort < 256 dòng vs ≥ 16k — SD r50 −.005/−.009 vs
  **+.0004/+.0005**; SM42 r50 −.012 vs −.0007; SR32 r20 −.026 vs −.0023. Ghép phản thực “chỉ xe ≥ 4.096 dòng dùng arm”:
  SD +.0001/+.0003, SM42 −.0009 (cả mạng −.0017). ⇒ FLOPs nằm ở xe lớn, thiệt hại nằm ở xe nhỏ: knob giảm liều phải gate theo
  số bước local, không áp đều. Ghép phản thực bỏ qua ghép nối qua Q, **không phải run**.
- SM (16 full + 48 head-only) ≈ SR32 ở r20 seed42 cả chi phí synthetic (1,56 vs 1,55×F) lẫn Δmacro (−.0037 vs −.0041, control khác codec)
  ⇒ 48 dòng head-only chỉ đáng ≈ 16 dòng full; knob đếm dòng (không phụ thuộc kiến trúc) đạt cùng điểm chi phí/utility.
  Giảm gradient synthetic làm benign tăng (SD +.0017/+.0016, SM42 +.0154) đổi TD (SM42 −.024): vẫn trục benign↔TD.
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
- Combo codec (10/11): ghép lite6 + QQ8 **giữ được** đến r50 (CBB Δmacro −.0003 cả hai seed vs KM50; CQ8 −.0005/−.0003 vs Core);
  bỏ cả Lloyd (LY0) khi ghép QQ8 thì FAIL benign (CBA) dù từng knob riêng PASS → không cộng PASS từng knob thành PASS combo, phải chạy combo.
  Byte TB 50 round thấp hơn byte đo ở r20 (265 vs 319 B) vì QQ8 chỉ gửi lớp có mặt và round đầu lớn hơn: báo byte theo đúng số round.

## Giới hạn audit hiện tại

Audit source/config/LR/IDs/support/CM→10metric/snapshot/teacher/finite từ artifact thật. Dataset chưa có full-content fingerprint;
chưa prediction→CM replay/full private-bank reconstruction toàn mạng. Không dùng formula test trên random tensors thay kiểm stored values.
Các thất bại/runtime đã biết và nguồn kết quả: [research index](docs/research/README.md), [generator catalog](docs/research/generators.md).

## Audit lịch sử trước B16 — 11/11/2026

- **V20 đã thử validation thời gian** full5343/hai seed; không gọi temporal validation là chưa thử. V20 chưa dùng OOF/val quality để weight teacher. Gap train→val không tách overfit/temporalshift/featureoverlap.
- **P20/P29 classmean TARGET ≠ KL-of-mean STUDENT**: code cũ vẫn mean per-rowKL. `F.kl_div(reduction="batchmean")` cũng không là KL trên mean prediction. P30 inputfit teachermean+output×input moment khác output×output moment.
- KL-of-mean≤meanKL bởi Jensen; giảm constraint/dose không tự là tăng tri thức. Cần dose comparator, guards và utility; không lấy giảm loss/Jensen gap làm PASS.
- Nguồn legacy thiếu rawcode/JSON: ghi reported historical audit, không gọi tái audit artifact. “Chưa tìm thấy” chỉ trong retained local scope, không chứng minh novelty.
  [Audit v4](archive/docs-20261113/research/history-audit-20261111.md), [legacy](archive/docs-20261113/research/history-audit-legacy-20261111.md).

## B16 reception — 11/11/2026

- **KL-of-mean student (MK) và .5rowKL (HK) FAIL cả2seed**, actual6/6r20 audit PASS.
  MK Δmacro−.001855/−.000390,p10−.026956/−.017892,benign−.004288/−.004354;
  HKmacro+.000008/+.000365 không đủ+.002. Không pilot/tune/group-size sweep/extend.
- MK r20 meanKL chỉ≈30%rowKL trên cùngforward/Q nhưng utility giảm. ClassQ là mean không suy ra chỉ ép mean student sẽ tốt hơn;
  pointwise constraint có thể là regularization hữu ích. Đây là giới hạn adaptation, không bác mọi distributionalKD.
- GiảmFPbenign→attack ngoàiTD180/123row vẫn không tăngbenignF1: benign→TD thêm694/636row.
  TDrecall tăng đổi precision/benign và xeítmẫu. Luôn đọc toànconfusion, không chọn riêng cell thuận.
- R1T4 CM MK/HK kháccontrol trướcsoftQ dù CPUr1exact PASS; Δmacro r1 có thể tới−.0026.
  Pairedseed không là commoninitialtrajectory. Không gán toàn bộ laterdelta cho loss hoặc rerunFAIL để tìmwin.
  [B16 result](docs/research/screening-mean-b16.md), [r1/Jensen](runs/sim-kaggle-b16/reception-diagnostics.md).
- Nested temporalprobe80/20 trên simfoldtrain: eligiblepopulation≥8row/cảphía chỉ98,37%, class11 81,82%,
  thấp hơn countbalancedhalf99,86%/96,65%. PriorTV median .159, p90 .381; không dùng precision thô làmqualityweight
  mà không xử lý prior/missingclass. [Nguồn thật](archive/docs-20261113/research/crossfit-feasibility-20261111.md).
- Nestedholdout trong originalfoldtrain đã góp vào fixedv4scaler. Giữ preprocessing chung, chỉ gọi model-out-of-fit
  **conditional on fixed preprocessing**, không claim toànpipeline độc lập. Shadowgenerator/Q/BN/weights vẫn phải loạiheldout.

## Macro-first / kéo dài pilot — owner11/11

- Owner tạm bỏ utilityguard phụ (p10/low256/TD/benign/absent), ưu tiênmacro; vẫn báo tradeoff và audit source/round/finite/CM.
  Giữmốcgain+.002/hai seed, không đổi verdict cũ. Scope mới có thể đánhgiá lại nhánh bịguardveto.
- Y50prior **lúc train** khác PHprior **lúc inference** b11: phải dịch syntheticlogit và cách đoQ đồngbộ; dịch một phía đếmprior sai.
  Y50macro+.003345/+.002890, cuốipilotgain còn giữ. W20realNCKD+.002461/+.002229 nhưng relativegain co nhẹ r15→20.
  Đều Core/bank cũ, không phải CBB; chỉ là evidence adaptation, không dự báo APFed/r50.
- Arm macro còn tăng chưa đủ để kéo dài: cần relativegain vớicontrol và lịchLR. Cos50r20LR≈6.76e-4; cos20r20LR1e-5.
  Bank F50 là phảnchứng: gainr20 mất/r50âm. Không nối cos20 thành protocolcos50 hoặc ngoại suy tuyến tính.
- Owner chốt chỉ cần không trao realrow, thống kê/xác suất được phép; không tự diễn giải thành yêu cầu SecAgg/DP hoặc pause.
  Privacybootstrap vẫn có rủiroônhỏ, không gọi “không uploadraw” làguaranteekhôngsuyrafeature.
  [Review](archive/docs-20261113/research/macro-first-review-20261111.md), [B17](docs/research/screening-macro-b17.md).

## B17 — macro-only vẫn không đủ, 11/11/2026

- Pprior-coordinate vàNRreal-NCKD trênCBB:6/6r20/auditPASS, **FAILmacro cả2seed**.
  Δmacro P−.000658/+.000192, NR−.000164/+.001630; TB16–20 Pâm2seed, NR−.000318/+.001401.
  Không guardphụveto: bỏguard không tự làm các adaptation này thànhwin. Khôngpilot/tune/extend/combo.
- Pbenign+.00834/+.00998 nhưngTD−.03485/−.02563,class11−.00998/−.00637; TD→benign thêm645/590row.
  TD vẫn là1/16 macro, bỏguardTD không bỏ lớpTD khỏimacro. Accuracy/weightedF1 tăng không thay pooledmacro.
- Gain Y50Core/W20bank không chuyển trực tiếp sang CBBsim: recipe/population/schedule khác;
  không kết luận riêngKM đãbãohòa hayQ8lànguyênnhân từ comparisonkhôngmatched.
- NRr1T4 chưaQ vẫn lệchcontrol dùCPUexact; counterNRgateđúng/shiftPđúng không làwholetrajectoryproof.
  Pđổi hardCE từr1 nên r1khác làexpected, không gán nó numericalbug.
  [Kết quả](docs/research/screening-macro-b17.md), [10metric](runs/sim-kaggle-b17/report-r20.md).
- Diagnostic V20hardCM thật: tại cùngtrainbinaryprior, trainP* không xếp hạng futureP* tốtđồngđều
  (V4/V5 accMultρ .055/.027, feignedBraking .045/.002 ởmin32); supportthreshold đổi population,
  TDρ từ−.113/−.127(min32) thành+.042/+.050(min8). Không chọnmincount đểtạo narrative.
  Count cũng cótínhiệu cho vài lớp; countweightedQ làpopulationmean, qualityweight làđổiestimand,
  không “sửa bug” và chưautilityproof. HardCMkhôngkhôiphụcsoftQ hoặc cross-recipientquality.
  [Nguồn/scope](archive/docs-20261113/research/crossfit-feasibility-20261111.md).

- Diagnosticexpertquality: không loại xe khôngpredictclassc khỏi nhómeligible chỉvì precisionundefined;
  khóa zero convention và eligibilitytheotruesupport, báo sốxe. Bỏabstainingexpert sẽ bias qualityranking.
- Cache v4 giữ **fileorder trongclient, khôngtimeorder** (`core/data.py`): nestedtemporalpartition cầnactualrcvtime/sidecaridentity;
  không lấy80%đầucache rồi gọilàearlytime.

## Khảo sát nguồn teacher / recipient — 11/11/2026

- [Diagnosticrecipient](archive/docs-20261113/research/teacher-source-survey-20261111.md) dự báo hardoutputfrequency cóđiềukiện trueclass,
  donor/recipientclient táchbằnghash, V20train/internalval. Futurepeerforecast theo8domain tăngBrierquality cả16class;
  infitforecast gần0/âm ở2armmixedinput. TrainCM làpostepochhardCM, không onlineQ CBB.
  Không gọi forecastgain làmacroF1 hoặc cross-modeltransfer.
  Futurepeer dùng validationlabels tươnglai nên **diagnostic only**, không làm trainteacher.
- Full5343support không chuyển thẳng sangsim200:8domain có1nhóm3xe/0rowclass11 ởsim.
  Groupfallback/concentration phải tính lại. `run=test` làmetadatasimulation, không phép dùng officialtestrow.
- CRAD “precision” = inversevariance của classaccuracy (recall), **không PPV**.
  Cùngn,accuracy .05/.95 cócùngweight; consensus cóthể loại specialist. Đã sửa ambiguity ởtransferreview.
  CrossfitKDpaper giảđịnhiid/same-input, không đảm bảo classmeanQ trên KMsyntheticinput khácx.
  [Nguồn và giới hạn](archive/docs-20261113/research/teacher-source-survey-20261111.md).
- Đổi nguồnQ vàweightquality là2estimand khác. Muốn kiểm ngoài fit cần cùngshadowweights/evalmode,
  matchedclient×classprobecount/countweight/codec, mainfulltrain và ineligiblefallback nhưnhau.
  So trực tiếp shadowheldout với onlineCBB không tách được nguồn/evalmode/producer ítdata.

## B18 matchedsource — đóng12/11/2026

- [B18](docs/research/screening-source-b18.md) 2seed×shadow/C/E/H, đủ8phase20round/source/CM/wireauditPASS.
  Hngoàifit **FAILmacro cả2seed**: H−C−.001066/−.001132, H−E−.001504/−.000636;
  guardphụreport-only, khôngpilot/tune/extend/combo. E−C+.000438/−.000496 không ổnđịnh.
- Outsidefit đo được lỗi thật nhưng không tựlà targettốt đểtransfer qua syntheticclassmean.
  ActualteacherH r19 có Q[TD,benign] .657/.647 >Q[TD,TD] .311/.322, packet/aggregateauditkhớp.
  MainbenignF1+.00814/.00817 nhưngTD−.04715/−.03805; TD→benign+783/+686.
  Không đổlỗi rounding hoặc bỏTD khỏimacro; sourcequalitystat và contentKDtarget làestimandkhác.
- Mainr1CMC/E/Hkhớpđúng2seed, matchedprobe291220row mỗi phía/cùngshadow/evalmode/countweight;
  utilityFAIL khôngdo thiếu matchedcontrol. Vẫn không wholetrajectory/privategradientproof hoặc eventindependence.
- RawmeanQ/hardCM không khôi phụcper-row softBrier/NLL/variance. Chỉ báo quantities đãđo;
  probe subsets cózero-rowclient thì pooled10metric tính trênpooledCM, không summaryclientrỗng.
- Bootstrap2.37845MB/xe và~723B/round gồmshadow/internal/probe/main; measuredtrain~1.92×C cócả2probe.
  Không gọi ngoài fit miễnphícommunication/compute hoặc ngoại suyFLOPs từ elapsed.

## Rà bão hòa CBB — 12/11/2026

- Phẳng r40–50 của CBB/CQ8 là do **cosine LR anneal**, không phải model hết học: ΔF1/ΣLR r10–20 ≈1,1, r40–50 ≈1,7–2,0; train loss còn giảm.
  Không đọc “curve phẳng ở r50” thành bão hòa; cũng không ngoại suy gain khi thêm ngân sách mà chưa chạy (baseline cũng tăng).
- Thiếu hụt macro CBB theo lớp: **TD 44%**, benign 11%, lớp11 10%, lớp7 9%. Gain macro lớn phải chạm TD; knob prior/ngưỡng chỉ dời benign↔TD.
- Gộp phiên b12 (suy luận) là hướng duy nhất trong lịch sử đạt ≥+.002 macro cả hai control; bị loại vì guard p10 và owner đóng 09/11.
  Mất ở feignedBraking −.029 (tấn công gián đoạn bị pha loãng). [Sổ tay](docs/research/method-building-brief.md).
- A1 (12/11): posterior của generator KM16 trên chính dòng synthetic gần one-hot (99,1% dòng `p_gen(y|x)≥.9`); mixture 16 lớp
  gần như tách rời trong 77 chiều → nhãn mềm theo dòng từ generator gần như không đổi target. Bayes-classifier generator trên dòng thật
  (mid-rank) macro .873–.880, thua CBB gần mọi lớp; lớp generator phân biệt kém nhất trùng lớp yếu CBB (TD, benign, 7, 11, 2).
  Chấm dòng thật dưới copula phải dùng mid-rank như lúc fit; PIT chính xác tại atom cho kết quả sai hoàn toàn (acc .02).
- B19 (12/11): thêm thành phần mixture ở vùng dòng thật bị generator nhầm (56 cặp, không lượng tử) chỉ +.0030 macro Bayes-generator
  trên fold val, benign/lớp11↑ nhưng TD/lớp7↓ — cùng kiểu đổi benign↔TD. Sửa generator theo nhầm lẫn không thêm phân tách TD.
  Lượng tử lite6/int8 giảm macro Bayes-generator .8926→.8802 (độ khớp, chưa phải utility). [B19](docs/research/screening-boundary-b19.md).
- N1/N2 (12/11): proxy DAGSNet train ở server trên synthetic **đã thử ở P20** (C8/C9/C14, proxy .79 < student .856) — tra lịch sử trước
  khi khuyến nghị. HGB toàn cục trên dòng thật 200 xe sim (trần GBDT liên kết) chỉ macro .8065 vs CBB sim .914: model toàn cục kém
  ở lớp phụ thuộc ngữ cảnh xe nhận (suddenStop .40, lớp11 .50); personalization là nguồn lợi chính trên dataset này.

- B21 full-scale pilot13/11: T4 artifact PASS nhưng utility FAIL (Δmacro TB r16–20 −.0016 < −.0015), dù endpoint−.0013. Không dùng verdict B20 PASS_TRAJECTORY thay luật B21; T3/T4F PASS đúng luật B21.
