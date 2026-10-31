# V2 — kết quả kiểm khả thi, cập nhật2026-10-11

Đợt đang chạy. A đã thu/verify; B/C chưa có artifact terminal.
[Report đủ10metric + confusion/F1 lớp](../../methods/shared-comparisons/builds/countsketch-subspace-ratio/results/report.md),
[manifest/cổng khóa](../pilot-v2.md), [ledger/source/quota](../../methods/shared-comparisons/builds/countsketch-subspace-ratio/results/rebuild.md).

## A — đóng adaptation CountSketch hiện tại

Hai kernel v1: `minhtran0601/mobihet-v2a-s42-20261011-v1`, `dinhan25/mobihet-v2a-s43-20261011-v1`.
Actual Torch2.10.0+cu128,2×TeslaT4; parent real-DAGS deterministic preflight cả2GPU PASS.
20client,1.000.369train rows,66feature,holdout nội bộ old/group65.536rows; r20 headline private/gp.

| arm | seed42 group macro-F1 | seed43 group macro-F1 | hoàn tất |
|---|---:|---:|---|
| local |0,464340|0,460972|20/20 cả2|
| proxy own |0,469440|0,467270|20/20 cả2|
| proxy dense |0,661926|0,645405|20/20 cả2|
| sketch small noEF |0,210803|0,205401|20/20 cả2|
| sketch medium noEF |0,270082|0,240748|20/20 cả2|
| sketch small EF |N/A|N/A|4/20 và3/20; lỗi|
| sketch medium EF |N/A|N/A|6/20 cả2; lỗi|

Dense−own≈+0,1925/+0,1781: tri thức proxy full có ích. Codec sketch adaptation này không giữ lợi.
Small/medium EF **FAIL hai seed**, không mở contact. Không đổi headline sang head để cứu cổng.
Không chạy lại A y nguyên: output đã đủ5arm hoàn tất/seed và checkpoint trước lỗi4arm.

## Vì sao kernel ERROR

Hai lớp lỗi độc lập:

1. **Mất ổn định phương pháp:** seed42 small/medium và seed43 medium lỗi `prepare→calibrate→finite`
   vì BN state không hữu hạn. Seed43 small lỗi private loss không hữu hạn tại client0/r4.
   [Chẩn đoán checkpoint](../../../../archive/methods/shared-comparisons/builds/countsketch-subspace-ratio/probes/sketch-diagnostics.json) cho thấy residual tăng trước lỗi:
   seed42 small46→238.090, medium28→16.296; seed43 medium đến18.184.
   Tỷ lệ norm residual mới/target≈17 ở small,≈3–4 ở medium. Decode top-k rồi sketch lại không co;
   không mặc định EF trong sketch sửa được collision. Đây là bằng chứng khuếch đại theo adaptation,
   không chứng minh mọi FetchSGD hoặc EF đều thất bại.
2. **Lỗi parent sau train:** quét key trên toàn working bắt chính `__notebook__.ipynb` do Kaggle sinh.
   Notebook private-inline được chủ dự án cho phép, source/key tồn tại trong private history.
   B/C chưa push đã sửa quét chỉ trên `runs/`,`source/`; A đã push giữ nguyên.

Pull exit0:527/523file,5.577/6.479MiB. Exact source/executed redacted audit, strict mọi checkpoint,
resume/payload/hash chain và prediction→client confusion→10metric đều PASS ở round đã commit.
Sample decisive inference client0/group checkpoint đầu/cuối có đánh giá PASS; không re-infer toàn holdout.
Không gọi4arm lỗi là20round hoàn tất; không điền metric còn thiếu bằng0.

## B — đã đóng seed-based adaptation

Hai kernel v1 đã pull/verify: dense/own1024/seed1024 đủ20round/seed;
K512/own512 không córoundcommit,nonfiniteproxy loss client0/r1 seed42,client17/r1 seed43.
Không phải auth/OOM; không rerun y nguyên. Không đủcheckpoint cho4arm lỗi là bằng chứng fail theorecipe,
không điềnmetric0 hoặcnhận fullround.

| teacher head r20 | seed42 | seed43 |
|---|---:|---:|
| seed1024 |0,307810|0,338032|
| gain vs localprivategp |−0,156531|−0,122940|
| retention dense−own1024 |0,298967|0,373225|

K1024 cólợi soown1024 nhưng không đủquality/retention; K512numericfail.
**B512/B1024 FAILhai seed; đóngprivateKD,ZO/contact.** Không gọiteacher làprivateIDS.
Full10metric/benign/perclass/source/checkpoint theo [report](../../methods/shared-comparisons/builds/countsketch-subspace-ratio/results/report.md).

## C — đủ artifact, FAIL hai seed

C cảhai kernel COMPLETE; đủ8arm×20round, source/weights/resume/payload/prediction→10metric và
sampled decisive inference PASS. Seed43 pull367file/4611MiB/213s; không cần rerun.

| contrast group/private/gp r20 | seed42 | seed43 |
|---|---:|---:|
| FL−diag | −0,016664 | −0,020820 |
| FL−own | +0,001276 | −0,001725 |
| FL−shuffle | −0,009317 | −0,008446 |
| ESS fraction theo lớp, min–max | 0,0090–0,0238 | 0,0070–0,0209 |

**C FAIL; không mở contact.** Không hơn Gaussian hard-label, own hoặc shuffled-address ratio.
ESS thấp dù đã clip/class-normalize; không đủ bằng chứng ratio chở tri thức hữu ích.
Không kết luận ESS là nguyên nhân duy nhất hoặc density-ratio toàn họ thất bại.
Support fidelity và transport bằng MALA là adaptation mới, phải có đối chứng/cổng riêng;
không kéo dài hoặc tune bản C này để cứu cổng. [Report đủ10metric/ảnh](../../methods/shared-comparisons/builds/countsketch-subspace-ratio/results/report.md).

## Bài học cho hướng tiếp

- Giữ đối chứng dense để phân biệt carrier hỏng với tri thức proxy không có ích.
- Phép nén phải kiểm norm/residual ổn định trên update thật, không chỉ linearity/codec/resume.
- Với random-coordinate EF, kiểm phép chiếu không scale và residual trong parameter space;
  ít coordinate có thể ổn định nhưng không đủ sức chứa/hội tụ trong20round.
- Kernel state không thay artifact audit; lỗi parent không xóa kết quả arm hợp lệ.
- Holdout/scaler/simulator và DAGSNet-only: chưa chứng minh R1 hữu hạn hay R2 đa kiến trúc.
