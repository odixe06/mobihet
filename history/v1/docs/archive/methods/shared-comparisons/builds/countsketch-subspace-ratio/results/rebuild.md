# V2 — ledger triển khai và kiểm nghiệm

Ngày2026-10-11. Chủ dự án yêu cầu triển khai kiểm khả thi, xác nhận A cả2cỡ, B FO512/1024,
C τ0,05/cap10/class-normalize/q-BN. [Cổng khóa](../../../../../research/pilot-v2.md).

## Trạng thái

Đang triển khai; **chưa có kết quả pilot Kaggle V2**. Không dùng điểm smoke làm feasibility.

### Tiếp tục22:13UTC+7

A seed42 ERROR nhưng log có5arm đủ20round; EF small dừng sau r4, medium sau r6.
Traceback medium: `prepare→calibrate→finite`, BN state không hữu hạn. Loss proxy/private tăng mạnh trước lỗi.
Parent cuối còn lỗi quét key trong notebook Kaggle tự sinh `__notebook__.ipynb`; đây là lỗi vận hành riêng.
Không rerun A y nguyên; đang pull source/checkpoint/predictions cả hai seed và kiểm partial arm.
Collector chỉ pull `runs/`, `source/`; executed source vào private_push, audit redacted.
B/C chưa push được sửa quét key chỉ trên hai thư mục artifact do driver tạo; exact orchestration test6notebook PASS.
Quota22:15UTC+7: minhtran0601 12,798616h, dinhan25 13,125101h, reserved0.
Đang pushB/C hai seed; source học/manifest không đổi, không tune A. Kết quả W&B chỉ live, chưa artifact-verified.

Seed42 pull exit0:527file,5.577MiB,174s; source/executed redacted audit PASS.
Strict verify5arm×20round, smallEF4round, mediumEF6round PASS, gồm sampled decisive inference các checkpoint eval.
`sketch-diagnostics.json`: small residual46→238.090, maxparameter4.703; medium28→16.296,max182.
Residual/target sau decode ở small≈17, medium≈3–4: decoder/EF trong sketch không co, phù hợp mất ổn định.
Không quy lỗi riêng precision từ assertion BN; đường tăng trưởng đã xuất hiện trước overflow.
B/C đủ4kernel pushv1 22:15UTC+7; source manifest học giữ nguyên. API RUNNING, W&B đúng ID đã xuất hiện.

A seed43 pull exit0:523file,6.479MiB,282s; strict audit5arm×20round,smallEF3round,mediumEF6round PASS.
Seed43 small assertion private loss nonfinite client0/r4; medium finite guard khiBNcalibrate.
A hai seed đã đóng:10arm hoàn tất,4arm numerical failure; cổngsmall/mediumFAIL, không rerun y nguyên.
`report.md`:14run,236row đủ10metric, perclassCSV và confusion/classfigures, pending B/C.
W&B SDK local có service-busy authentication timeout; direct authenticated read-only GraphQL hoạt động,
khớp expected IDs/fingerprint, không thay auth/training. Dùng `monitor_v2.py --http` nếu SDK busy.

B seed42 terminalERROR đã pull312file/3.920MiB/144s,source/executed redacted audit PASS.
K512/own512 log:setup/auth thành công,`torch.isfinite(loss)` failclient0/r1,không córoundcommit/weights.
Đây là numericalfailure,khôngOOM/auth. Không rerun y nguyên. Dense/own1024/seed1024 đủ20round đangverify.
Collector cũ đã nạp trước sửa partial0round nên bịchặn; chạy lạicollector mới chỉthu/audit,khôngreruntraining.
Còn thu seed43; cổngB xét riêng512/1024, không yêu cầu tấtcả5arm đều hoàn tất mới báoK1024.
Code mới: `mh/v2_math.py`, `mh/v2_methods.py`, `mh/v2_run.py`; không sửa notebook/source pilot trước đã PUSHED.
MCP tool Kaggle không có trong inventory phiên; dùng CLI/SDK account-scoped, không đổi credential active.
W&B API verified entity `21522798-uit`, local SDK0,29,0; production phải online trước train.

## Kiểm local

- Math actual codec, tamper/base, cohort5/duplicate, interrupted frame, sketch linearity/decoded residual,
  real DAGSNet seed gradient/finite-difference/reconstruction, class weight normalization/shuffle: PASS.
  PeakRSS1.409MiB, RTX3050, Torch2.13.0+cu130; chưa xác nhận workerT4/Torch2.10.
- Smoke đầu a_small_ef,20×256train/256hold,r1: exit0, peakRSS2.145MiB. Đây là codepath test,
  source sau đó còn hoàn thiện; chưa dùng run này làm audit/resume của source cuối.
- Bộ smoke16arm×3round đã chạy/verify metric; resume A phát hiện residual r1 khác bit do GPU atomic scatter,
  chưa PASS exact-resume. Khóa deterministic+CUDA workspace trong source cuối, kiểm lại; không push trước khi qua.
- Deterministic smoke lần đầu chặn CUDA median-with-indices dù chỉ dùng values. Đổi median sang middle stable-sort,
  top-k stable index tie; giữ đúng công thức median/top-k và ghi trước pilot. Source cuối đã kiểm lại PASS.
- Production basis K512/K1024, batch32local,4bước: finite gradients và CE giảm; peakGPU≈1.115/2.243MiB,
  peakRSS1.642MiB. Không suy thứ tự tốc độ từ probe chưa đồng nhất warmup, không dùng làm F1 feasibility.
- W&B login/init/log/finish online PASS: `https://wandb.ai/21522798-uit/mobihet-v2-preflight/runs/0o5f3fmt`.
  Đây là auth preflight không train, SDK0.29.0; tắt code upload. Notebook mới private-inline theo quyền đã chốt:
  key tồn tại lâu dài trong private version history; publish/share lộ key; rotate phải rebuild/re-push.

## Quota trước chuẩn bị

CLI/SDK read: minhtran0601≈14,35h, dinhan25≈14,63h free; các owner khác17,57–22,07h.
Refresh2026-10-17UTC. Ưu tiên hai owner quota thấp nếu preflight đạt; kế hoạch≤8h/account gồm dư.
Số chính xác/giờ read và quota trước/sau từng push sẽ ghi lúc thực hiện, không suy quota hiện tại từ snapshot này.

## Còn làm

Hoàn thiện verifier/notebook/W&B; driver smoke, production-basis small probe, resume/tamper;
freeze source/config/fixture manifest; notebook2×T4 preflight; pilot U hai seed A/B/C;
pull+verify+report, cổng; chỉ hướng qua U mới được mở contact training với manifest riêng.
Git/DVC giữ local, không tự commit/push/backup. Artifact nặng và private_push ignored; chưa backup.

## Preflight cuối trước Kaggle

- Source deterministic cuối:16arm×3round smoke PASS; A/B/C stopr1→resume r2–3 weights/state bitwise bằng uninterrupted, tamper bị từ chối. PeakRSS2.479MiB.
- Six notebook A/B/C×seed42/43: private metadata, Internet, image digest, auth key boundary, exact7/5/4job orchestration fake-process PASS.
- Parent notebook kiểm DAGSNet thật, seed-gradient/sketch/bincount deterministic trên cả2T4 trước train; chưa có bằng chứng T4 cho tới worker chạy.
- Source/fixture manifests frozen trong `papers/v2/notebook/` và input2owner; private input dataset create exit0, đang chờREADY.
- `quota-before.json` read11:45UTC/18:45UTC+7: minhtran0601 14,351889h, dinhan25 14,634846h, reserved0 cả2.
- W&B32expected IDs ở `wandb-runs.json`; status vẫn chưa verified tới khi API trả đúng config fingerprint.
- Không thay mh source/manifest đã freeze. Raw executed notebook có key chỉ tải vào private_push; audit copy phải redacted.

## Kaggle A đã PUSHED v1

- Input private2owner READY trước push.
- 2026-10-11 11:56UTC/18:56UTC+7: `minhtran0601/mobihet-v2a-s42-20261011-v1` và `dinhan25/mobihet-v2a-s43-20261011-v1`, mỗi kernel7arm×20round dự kiến.
- Source/config từ secret-free `notebook/v2a-s{42,43}-20261011-v1/`; bản key trong ignored private_push, không sửa sauPUSHED.
- Đang chờ worker/W&B để xác nhận actual2T4, deterministic preflight, tiến độ; chưa có kết quả pull/verify V2.
- B/C notebooks đã dựng/validate, source cùng snapshot; chưa PUSHED. Cần thu A rồi chạyB/C và cổng độc lập.
