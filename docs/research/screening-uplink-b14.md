# Uplink và synthetic interpolation b14 — khóa trước chạy 11/11/2026

Quyền owner/split/thang scale như [b13](screening-module-b13.md): ưu tiên performance, cho tăng compute,
giảm byte tối đa chưa cap cứng, hỗ trợ kiến trúc khác, experiment chỉ **DAGSNet W1**.
B13 đã FAIL cả5nhánh; không tune/kéo dài SQ/RK/SK/JS/LS. CBB vẫn nền.

## Evidence → mechanism → falsifier

Mean probability theo true class làm mất phân bố confidence giữa dòng/client. B13 đổi loss không đủ gain;
đây chỉ là lý do kiểm kênh thông tin khác, không causal proof Q gây gap. [Rà module](../../archive/docs-20261113/research/pipeline-module-review-20261110.md)
nêu synthetic còn giúp chống overfit, bank/Qregion từng FAIL. B14 giữ Q **theo lớp**, không tạo bucket hoặc input-bank.

| arm | cơ chế và bằng chứng | kỳ vọng cần kiểm / bác bỏ |
|---|---|---|
| C | CBB nguyên gốc, Q8 arithmetic mean | control mới cùng source/batch/seed |
| MX | synthetic input thay bằng `.5*x_s + .5*x_r` khi batch/client có **cùng true class**; giữ Q[y] và số slot | nối vùng synthetic với real local, giảm memorization; bác bỏ khi macro không đạt hoặc td/p10/rare giảm. Khác twin phản thực cũ: không đổi họ feature hoặc lấy benign khác nhãn |
| GP | upload fp16 **mean probability** cho ô ≥8 dòng, server logpool → Q8 | thử geometric aggregation của confidence với codec cùng bitwidth LG; có thể đàn áp offdiagonal nhiễu. Bác bỏ khi chỉ sharpening/dời td hoặc mất xe nhỏ |
| LG | upload fp16 **mean centered log-probability** trên dòng thật cho ô ≥8 dòng, server linear sum → Q8 | thông tin về confidence trên từng dòng khác log(mean probability); kiểm macro và FP benign. Bác bỏ khi tail/noise làm td/rare/p10 giảm hoặc không có gain |

Không nhánh nào gửi weight/hidden/BN/parameter gradient; không thêm model phụ. Tất cả chỉ cần đầu ra16lớp
và input77chiều chung; không cần aligned hidden. Logit scale giữa kiến trúc vẫn là rủi ro calibration,
chưa có bằng chứng model trộn full-scale từ b14 DAGSNet-only.

MX là adaptation same-class interpolation, ý tưởng convex combination theo
[mixup](https://arxiv.org/abs/1710.09412); không reproduce cross-class mixup hoặc coi scaled flags/atoms nội suy là vật lý hợp lệ.
Ghép uniform trong real rows hợp lệ **cùng lớp của batch hiện tại**, không tìm dòng khác xe hoặc val/test.
Thiếu lớp/floor-only/padded → synthetic gốc hữu hạn. Không đổi CEreal/LA/target/loss/64slot/floor16.
Real rows và thủ tục thu Q giữ nguyên; **giá trị Q không bất biến** vì BN chung đổi theo synthetic mới.
Giữ Q[y] cho input nửa-real là soft supervision gần vùng real, có rủi ro như realKD cũ; kiểm td/absent/c11.

## GP/LG chính xác

`p_ki = softmax(z_ki)` trên real row i trước optimizer update như CBB;
`center(v) = v - mean_output(v)`; count `n_kc` từ bootstrap, `N_c=Σ_k n_kc`.

- GP ô `n>=8`: `u_kc = fp16(mean_i p_ki)`; receiver dùng `center(log(max(u,1e-6)))`.
- LG ô `n>=8`: `u_kc = fp16(mean_i center(log p_ki))`; receiver dùng `u` trực tiếp.
  `log p` tính bằng log_softmax hữu hạn; center xóa offset chung. Accumulate fp32, không accumulate fp16.
- Ô `0<n<8` cả GP/LG giữ **Q8 mean probability** hiện có; receiver chuyển thành centered log sau clamp1e-6.
  Không gửi mean-logit mới từ ô một dòng; không bỏ count nhỏ khỏi denominator. Ô vắng weight0.
- `score_c=Σ_k n_kc*decoded(u_kc)/N_c` float64; cast fp32, softmax16 → round(p*255) → normalize Q8.
  Không intermediate fp16 downlink; GP và LG cùng đường này. Q r chỉ tác động từ r+1.

Trước codec/clamp, `center(log softmax z)=center(z)`, và `log E[p_j] - E[log p_j]>=0`.
LG phạt confidence không ổn định khác nhau từng output; không chỉ uniform temperature. Đây là class-conditional
summary từ **inputs khác nhau**, không product-of-independent-experts trên cùng input hoặc Bayesian consensus.
Clamp/codec làm contrast không phải exact Jensen identity. GP16 giữ cùng bitwidth LG để giảm confound GP8 zero tail;
khác precision của hai dạng biểu diễn vẫn thuộc adaptation, không gọi pure-information causal proof.
Primary utility: mỗi GP/LG vs **CBB cùng seed**; LG−GP chỉ đọc cơ chế, không đổi control/gate.

## Wire và audit

GP/LG uplink: header32B + present mask2B + eligible mask2B + codec flag1B + body.
Body = eligible fp16 rows16cột (32B/ô) rồi small Q8 rows (16B/ô); chọn raw/zlib6 ngắn hơn.
Counts được cached từ bootstrap, không truyền lại. Downlink: header32B + flag1B + raw/zlib toàn bảng Q8.
Trần hai chiều **838B/round/xe** khi đủ16ô lớn, chưa nén; 41,9KB/50r ngoài bootstrap.
Bootstrap giữ KM16/lite6; báo byte **thực** từng arm, không suy byte full từ sim.

`methods/mobihet_uplink.py` kế thừa CBB. Path GP/LG tái dùng epoch/Q8 simulator của parent để giữ loop/guards;
teacher arithmetic tạm và byte estimate parent **không tiêu thụ/truyền**, thay bởi teacher/payload GP/LG trước round signature.
Đây là implementation reuse, không cộng thêm packet arithmetic vào chi phí.

`uplink_audit=true`: lưu **frame bytes thực**, offsets, client IDs/counts, aggregate trước softmax, Q/downlink2rank mỗi round.
`scripts/audit_uplink.py` decode tất cả frames, kiểm count/mask/ngưỡng8, weighted sum, Q8/hash/byte và source/config audit trước verdict.
Local smoke cũng replay packet thật, không chỉ random tensor test. Envelope ban đầu: ≤1e-10 LG truyền fp16,
≤1e-5 cho fp32 log/center CPU vs GPU. **Sửa bug auditor 11/11 00:43:** LG có ô Q8 fallback cũng chạy fp32 log/center,
nên bound từng lớp = `(1-f)*1e-10 + f*1e-5`, `f=count_fallback/N_class`, không áp1e-10 LG thuần cho cả hybrid.
Lớp không fallback vẫn ≤1e-10; GP giữ≤1e-5. Không đổi training, codec, HP hoặc gate utility.
Q8 chỉ cho phép ambiguity khi nằm trong
interval probability±1e-6 quanh rounding boundary. Không blanket ±1integercode hoặc nới gate sau lỗi.
Đây là bounded numerical audit; không proof whole T4 trajectory bit-exact/private-weight replay.

## Luật khóa trước train

**8 kernel = C/MX/GP/LG × seed42/43**, 200xe fold train→val, cosine20/r20, DAGSNet W1, protocol như b13.
Cap2.75h/kernel, whole batch cap22kernel-giờ +15% margin; quota/slots sống theo owner trước push.
Local trước: unit/default2round bit-exact, mask/floor, decoded frames → teacher/state; smoke real200xe W1 compile2round
bật LG+MX **chỉ kiểm đường code**, không utility combo. Hai T4 mỗi arm vẫn compile/backward gate riêng.

Giữ nguyên gate **b13**, không hạ threshold sau b13 FAIL: từng seed endpoint r20 Δmacro≥+.002;
Δp10≥−.006, Δbenign≥−.0005, Δtd≥−.0035; Δabsent≥−.01 nếu≥2000dòng. Audit source/config/HP/LR/CM→10metric,
finite/skip≤1%/mọi client applied và wire audit PASS trước verdict. Low256/rare/c11/P/R từng lớp/TB16–20 báo đủ.
PASS hai seed → pilot20cosine50; pilot/r50 gate **y hệt b13**, control CBB cùng seed/round; nhiều nhánh PASS lên riêng.
FAIL dừng, không tune lambda/mode/mincount/clamp/temperature hoặc kéo dài nhánh FAIL.

Worker `SIM_BATCH=b14 scripts/run_b13_pipeline.py`: source-lock → unit/local+wire → quota → push → collect1200s → audit → verdict/report.
Giữ exact owner/slug/version trong `runs/sim-kaggle-b14/batch.json`; stage/job thật chỉ ở [continue](../operations/continue.md).
Không official test ở screening. Chưa utility mới khi khóa tài liệu này.

## Phục hồi preflight local — 11/11 00:43

Job đầu dừng **trước quota/push** ở auditor, không phải training/utility FAIL.
Smoke W1/200xe/compile đủ2round; CPU replay lệch mean-score tối đa3.78e-10/r1 và1.15e-9/r2.
Chỉ thay fallback log transform sang GPU khi replay packet **cũ** → khớp stored aggregate **đúng0 cả2round**;
lớp không fallback cũng lệch0. Count/mask/frame/header/Q8/hash/byte không đổi.

Failure/source giữ `runs/sim-kaggle-b14/recovery-20261111/`; `diagnosis.json` giữ SHA wire và số đo replay;
fixture6,7KB ở `tests/fixtures/uplink-fallback-r1.npz`, regression trướcfixFAIL đúng3.78e-10,
saufixPASS; test còn reject aggregate corrupt ở cả lớp có/không fallback. Tổng30tests PASS.

Phục hồi bằng `--reuse-local-smoke`: kiểm training overlay và HP khớp nguyên bản, CM→10metric/state finite/compile/wire
trên artifact cũ; giữ `source-lock.json` ban đầu, thêm `source-lock-recovery.json` cho worker/auditor sửa.
Ghi toàn bộ SHA smoke trong preflight. **Không train lại smoke** vì checker lỗi; chỉ sau các gate này mới query quota sống và push.

## Backlog sau b14

Cross-fit teacher/reliability vẫn cần nguồn đánh giá thật ngoài fit và kế toán model/state/compute; không đổi entropy train thành reliability.
LOO hạ ưu tiên: [count diagnostic b13](screening-module-b13.md) full maxselfshare0,46% vs sim9,01%; count bound chưa là utility proof.
Không khẳng định đã cạn không gian cải tiến.

## Kết quả đóng 11/11/2026 — 8/8 đủ r20, mọi ứng viên FAIL

Collector phục hồi `watch-recovery.sh` hoàn tất 04:21; kiểm lại trực tiếp 05:11:
source/base/overlay/config/panel/LR/20CM → đủ10metric PASS; hai T4/rank compile PASS,
skip0, không client mất toàn bộ update. GP/LG replay packet thật đủ20round/2rank PASS.
Sai số mean log-score tối đa GP≈3,97e-7, LG≈7,69e-9, trong bound đã ghi trên.
Đây không phải whole-private-weight replay. Controller cũ có state `running` stale sau reboot;
PID không còn, không bật lại train. Marker [DONE](../../runs/sim-kaggle-b14/DONE) và artifact là bằng chứng completion.

| nhánh | Δmacro s42 / s43 | Δbenign s42 / s43 | ΔtimeDelay s42 / s43 | verdict |
|---|---:|---:|---:|---|
| MX | −.015809 / −.015881 | −.005403 / −.004517 | −.009021 / +.001798 | FAIL cả2; macro/p10/benign, thêm TD s42 |
| GP | −.000307 / +.000881 | −.000840 / +.000365 | −.004041 / +.009206 | FAIL cả2; macro, thêm benign/TD s42 |
| LG | +.000834 / −.000199 | −.001349 / −.000025 | +.007066 / +.003443 | FAIL cả2; macro, thêm benign s42 |

Control macro s42=.913491, s43=.913456. Không nhánh nào đủ gain+.002 **từng seed**;
không pilot, không tune/mix các nhánh FAIL. CBB giữ nguyên.
Δmacro TB16–20: MX −.015715/−.015254; GP −.000309/+.000975;
LG +.000589/+.000182. TB chỉ mô tả, không thay endpoint/gate.

- MX: suddenConstantSpeed F1 −.1021/−.1088, precision −.2317/−.2443 trong khi recall +.0510/+.0526.
  Riêng lớp này chiếm40–43% tổng giảm macro; FP zeroSpeedReport→lớp11 tăng209/236 dòng,
  constantSpeedOffset→lớp11 tăng110/103. FP benign→attack ngoàiTD tăng464/556.
  Xe ≥4096dòng mean F1 −.01546/−.01612; không chỉ vấn đề xe nhỏ.
  Cùng nhãn không bảo đảm đoạn thẳng trong77feature scaled giữ đúng ranh giới; BN và target cũng là confound,
  không suy ra riêng feature flags hoặc BN là nguyên nhân đã chứng minh.
- GP/LG: FP benign→attack ngoàiTD chỉ giảm71/13 và72/39 dòng, nhưng benign→TD tăng110/83 và304/131.
  LG tăng TD recall nhưng giảm precision cả2seed. Không giải được gap benign-generalization lớn đã thấy ở full-scale.
- LG teacher r19 (tiêu thụ tại r20) diagonalmean=.99436/.99387, entropymean=.01894/.02005nat,
  **239/256 ô zero sau Q8**; GP≈.98107, entropy≈.08,217/218ô zero. Không dùng confidence làm reliability.
  Đây là hai live trajectory khác nhau; không kết luận định lượng LG−GP chỉ do Jensen gap/độ sắc.
- GP/LG tốn≈615/545B hai chiều/round vs CBB≈306B ở **sim này** (~2,01×/1,78×), không đủ utility.
  Bootstrap mọi arm1.196.032B/xe; tính cả bootstrap, tổng20round chỉ tăng≈0,51%/0,40%.
  Tăng byte mỗi round không là nút thắt duy nhất; cần thông tin có ích ngoài mẫu.

Nguồn: [10metric/byte](../../runs/sim-kaggle-b14/report-r20.md),
[verdict](../../runs/sim-kaggle-b14/verdict-r20.json), [CM/cohort/teacher](../../runs/sim-kaggle-b14/diagnostics.md),
[JSON chi tiết từng lớp](../../runs/sim-kaggle-b14/diagnostics.json).
[Phân tích lại và ưu tiên kế tiếp](../../archive/docs-20261113/research/b14-analysis-20261111.md).
