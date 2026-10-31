# M — teacher và cá nhân hóa distillation, 20 round

Owner 31/10 yêu cầu tiếp tục thử nghiệm, kiểm khả năng sửa lỗi qua round; sau đó chốt **20 round chẩn đoán**, không 50.
Owner bổ sung ưu tiên xe ít dữ liệu/thiếu lớp, nhận thông tin toàn mạng theo nhu cầu từng xe.
Nền: [L đã audit](../../archive/docs/research/results-l10.md), [nguyên nhân](../../archive/docs/research/root-cause-experiments.md), [protocol](protocol.md).

## Phạm vi chẩn đoán

- Train mới r1–20, cosine vẫn theo 50 round. Full 5.343 xe, DAGSNet W1, mọi xe mỗi round, 1 epoch, batch64,
  AdamW reset mỗi round, clip1/fp16, seed42/43. Không sửa protocol chung, không resume từ snapshot nhỏ.
- Không proxy neural. Q là bảng xác suất 16×16, cố định trong một round, cập nhật sau round cho round kế tiếp.
  Không raw local row/feature/nhãn từng mẫu/private weights trong thông điệp phương pháp.
- Generator copula/dose64, global prior/bootstrap, LAreal, codec mean_fp16, fixed loss coefficient1 giữ nguyên.
- Đánh giá chính: test tương lai của chính từng xe. Mốc cuối r20, không best-round; báo đủ10metric,16lớp,
  mean/p10/median/min, p90−p10/std giữa xe và từng cohort theo traincount. Chỉ chẩn đoán, chưa benchmark/chốt thắng.
- So cùng seed/cùng source: Δr20 vs augmentation, noReal và gate control. Mốc r10 chỉ đối chiếu L, không lấy baseline
  G4 r10 làm cổng cho endpoint r20. Cổng L tại r10 giữ nguyên; không suy là cổng r20.
- Đọc dấu hiệu utility: ΔF1r20 và Δp10 cảhai seed vs control trực tiếp; cá nhân hóa cần xem meanΔxe<256
  so uniform-control và augmentation. Báo tỷ lệ xe tốt/kém hơn. Không áp cổng PASS/FAIL mới hoặc kiểm định significance;
  khoảng cách thu hẹp do xe mạnh suy giảm không phải cải thiện mong muốn.
- Đọc ΔF1(r20−r10), Δ(r20−r15), slopeOLS r11–20, đường meanF1/p10/xe<256; train/test tại các mốc cố định.
  Không yêu cầu monotonic từng round, không chọn hệ số hoặc stopping theo test. Batch này không chứng minh sau r20.

## Bằng chứng → cơ chế → dấu hiệu bác bỏ

1. L adaptive làm giảm F1; fixed1 cải thiện cảtrain/test. Giữ scale1, không tự tăng KD cho xe nhỏ.
2. Gate chỉ chặn realKD lớp teacher sai; synthetic vẫn dùng Q sai. Fallback sửa riêng kênh này, không hardcode lớp13.
3. Q online trộn trạng thái đang học/BNtrain/dropout. Postepoch eval có thể cho bảng phản ánh model cuối epoch rõ hơn;
   đây là can thiệp timing+mode gộp, không tự gọi là sửa staleness. Nếu tốt mới tách timing/mode.
4. 848xe<256 chiếm15,871%client nhưng0,359%trainrows; teacher gộp theo mẫu chưa chắc hợp từng xe/lớp.
   Tổng absent-local chỉ0,289%test, cần báo riêng cohort; không mặc định thiếu lớp là nguyên nhân chính toàn mạng.
   Scan cohort31/10: nhóm<256 có median8/16lớp, absent-local1555/34827test=4,465%; rare1–16mẫu9776testrows.
   Bằng chứng `runs/m-study/support-cohorts.json`; không dùng kết quả test để chọn client chẩn đoán hoặc hệ số.
   L hardfixed/gate tốt hơn softfixed/gate trên nhóm nhỏ: đây là bằng chứng ngược với giả thuyết cứ tăng teacher
   là giúp xe ít mẫu. Localρ là đối chứng kiểm giả thuyết owner, chưa phải cơ chế nên triển khai.
5. L6/L7 còn tăng F1 giữa r5→r10; chưa có đường v4 của cấu hình này sau r10. Chững r20 hiện là quan sát owner
   cần đo, chưa phải kết luận từ batchL. Không hứa train càng lâu luôn tăng độ chính xác.

## Các cell, mỗi cell hai seed

| cell | teacher collection | synthetic target | realKD | mục đích |
|---|---|---|---|---|
| M0/M1 | không Q | onehot | tắt | augmentation control, không learned packet/r |
| M2/M3 | online | Q | tắt | softfixed/noReal control L0/L1 kéo tới20 |
| M4/M5 | online | Q | gate | softfixed/gate control L6/L7 kéo tới20 |
| M6/M7 | online | fallback | gate | sửa riêng synthetic teacher sai |
| M8/M9 | postepoch eval | Q | gate | thay riêng thời điểm/mode tạo Q |
| M10/M11 | online, server √n_kc | Q | gate | giảm chi phối của xe nhiều mẫu theo lớp |
| M12/M13 | online | fallback + uniform mixture | gate, uniform weight | control cho cá nhân hóa |
| M14/M15 | online | fallback + local support mixture | gate, local support weight | cá nhân hóa theo lớp/xe |

R1 mọi cell vẫn CEreal+CEsynthetic hard, trước khi có teacher. M0/M1 không thu/gửi Q. Postepoch chỉ đổi Q dùng từ r2.
M6/M7 vsM4/M5 tách fallback; M8/M9 vsM4/M5 tách collection; M10/M11 vsM4/M5 tách aggregation.
Chưa chạy kết hợp postepoch×fallback; không cộng gain các cell.

## Công thức khóa trước launch

Q[y] dùng xác suất logit chưa LA như L; CEreal dùng logit adjustment hiện hành. Mỗi phương pháp giữ denominator
realKD chia mọi dòng real; class gate nhân0 khi argmaxQ[y]≠y.

- **Fallback:** synthetic nhãn y dùng Q[y] nếu argmaxQ[y]=y; nếu sai dùng onehot(y). Các lớp còn lại không đổi.
- **Uniform/local support:** đặt ρ=1/2 cho uniform; local đặt ρ(k,y)=64/(64+n_train(k,y)). Số64 lấy batch chung,
  không quét trên test. Nếu teacher argmax sai, ρ hiệu dụng cho synthetic bằng0.
  Synthetic target = ρQ[y]+(1−ρ)onehot(y), KL(target||p_synth); realKD nhânρ(k,y) và gate hiện hành.
  Outer loss coefficient vẫn1. Lớp thiếu local:ρ=1; lớp nhiều mẫu:ρ gần0. Không gửi thêm n_train mỗi round:
  count đã có tại client và bootstrap; không cần server biết sample khó nào.
- Uniform vs local thay riêng quy tắcρ; cảhai giữ cùng mixture/fallback/realKD. Đây là gói cá nhân hóa hai kênh
  distill, chưa tách riêng tác động real/synthetic. Không khẳng định đã tăng KD hơn controlQ100%; cơ chế giảm
  lực teacher ở lớp đủ dữ liệu, giữ lực ở lớp ít/không có dữ liệu, đồng thời có nhãn cứng synthetic.
- Postepoch: một full local real forward với model cuối epoch, frozen runningBN/dropoutoff, softmax chưa LA.
  Mỗi client gộp theo true class, encode meanfp16; server vẫn gộp n_kc, downlinkfp16+rowrenorm. Không raw transfer.
  Báo q_collect_seconds riêng, nằm trong t_train; không cộng hai lần tổng thời gian.
- **Server √n_kc:** Q[c]=Σ_k sqrt(n_kc) decode(meanprob_kc) / Σ_k sqrt(n_kc), chỉclient có n_kc>0 đóng góp.
  Count đã có trong bootstrap; không gửi thêm private representation/weights/mẫu. Codec/packet không đổi.
  Giả thuyết tăng đại diện xe nhỏ, nhưng xe ít mẫu có thể tăng nhiễu; bác bỏ nếu tail không tốt hơn hoặc pooled giảm.

Bác bỏ khi Q đẹp hơn nhưng utility/tail không hơn control; fallback chỉ đổi benign falsepositive lấy recall mà pooled
xấu đi; localρ không hơn uniform/tail; hoặc train tăng/test đứng yên dù teacher tiếp tục thay đổi.

## Đo và kiểm

- FulltestCM/10metric và smallQsnapshot mỗi round; diagnostic fulltrainCM/probe34xe cố định ở1/2/5/10/15/20.
  Probe giữ teacherQr−1, clone privateweights, fp32/trainBN/dropout0; không gọi là actualAMPgradient.
- Thêm Q true-probability/entropy, rejection count và L1 thay đổi theo round; localρ trung bình trong probe.
  Báo benign↔timeDelay, Sybil, cohort traincount và absent/rare local class support.
- Owner chọn thêm chẩn đoán lớp từ xe khác, sau đó nhấn mạnh own-test vẫn chính. Chẩn đoán phụ chỉ34xe đãchọn theo
  traincount×scenario, tại1/10/20: bộ đánh giá mô phỏng lấy pool test cốđịnh tối đa256mẫu/lớp từ xe khác, cùng pool
  giữa phương pháp/seed, loại sourceclient trùng xe đang chấm; ghi đủ support/hash. Không dùng pool này để train,
  bootstrap hoặc Q, không truyền nó qua kênh FL. Không gọi đây là public data được phép chia sẻ khi triển khai.
- Train+val vẫn đãfit; chưa independentvalidation. Officialtest đã được xem qua nhiều batch, không gọi fresh confirmatory.
  Aggregate statistics chưa có DP/secureaggregation guarantee; không đồng nhất không rawsharing với bảo đảm riêng tư hình thức.
- Test công thức target/gradient độc lập, postepoch fullsupport và đối chiếu forward từng client,
  bảo toàn weights/BN/RNG, decoded codec; CPU2round legacy bitexact với frozenL.
- Realdata smoke W1 compiledAMP+strictbankrebuild/CM metrics; localsm86 không thay remote2×T4 gate.
- Dự kiến~4–5h/run, cap6h với margin quota ≥6,9h/account,16run=96hcap. Đọc quota live/slots và setup/compile/eval/stateIO
  trước push, notebook private và sourceoverlaySHA. Nếu run budget-stop<20 chỉ báo partial, không tự restart/resume.
- Với Q:1088B/xe/round, bootstrap3311744B ⇒3333504B/xe/20round. Augmentation3311744B/xe.
  Không gọi accounting bootstrap là end-to-end network/secureaggregation proof.
- Poll status exactversion nền1200s bằng codex-long-jobs, không modelpolling lúc chờ. Completion chỉ là tín hiệu collect/audit.

Manifest thực thi: `runs/m-study/batch.json`; quota/preflight/source/artifact được ghi cùng thư mục. Trạng thái chạy xem
[bàn giao sống](../operations/continue.md). Không Git/DVCpush, không cleanup hoặc launch50round.
